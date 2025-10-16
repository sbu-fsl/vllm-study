# Extended experiment runner with multi-GPU scaling support
# Includes detailed logging: model info, GPU usage, input resolution, variable dataset sizes, and epoch timing

import os
import time
import csv
import datetime
import argparse
import traceback
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import datasets, transforms
import timm

# Modify model loop to support varied training/validation set sizes
import random

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

input_shapes = {
    "default": (3, 224, 224),
    "large": (3, 384, 384),
    "xlarge": (3, 512, 512)
}

def imagenet_transform(shape):
    return transforms.Compose([
        transforms.Resize(max(shape[1:]) + 32),
        transforms.CenterCrop(shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_imagenet_train_dataset(shape):
    return datasets.ImageFolder(root='/home/ImageNet-1k/train', transform=imagenet_transform(shape))

def get_imagenet_val_dataset(shape):
    return datasets.ImageFolder(root='/home/ImageNet-1k/val', transform=imagenet_transform(shape))

def append_to_csv(row, csv_filename, headers):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

def extract_activation_functions(model):
    activations = set()
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.Tanh)):
            activations.add(m.__class__.__name__)
    return list(activations)

def get_model_from_timm(model_name, num_classes=1000):
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed loading timm model {model_name}: {str(e)}")

def estimate_batch_size(model_name, usage_ratio=0.9):
    """
    Manually mapped batch sizes for selected models at low (~20%) and high (~90%) GPU usage.
    """
    name = model_name.lower()
    manual_bs_table = {
        'mobilenetv2': (128, 512), 'mobilenetv3': (128, 512),
        'resnet18': (64, 256), 'resnet34': (64, 192), 'resnet50': (32, 128), 'resnet101': (16, 64), 'resnet152': (8, 32),
        'efficientnet_b0': (32, 128), 'efficientnet_b1': (32, 96), 'efficientnet_b2': (16, 64), 'efficientnet_b3': (16, 48),
        'efficientnetv2_rw_t': (16, 64), 'efficientnetv2_rw_s': (8, 32), 'efficientnetv2_rw_m': (4, 16),
        'densenet121': (64, 192), 'densenet161': (16, 64), 'densenet201': (8, 32),
        'vgg11': (16, 64), 'vgg13': (16, 48), 'vgg16': (8, 32), 'vgg19': (8, 24),
        'shufflenet': (128, 512), 'squeezenet': (128, 512),
        'mnasnet': (64, 256),
        'regnetx': (16, 64), 'regnety': (16, 64),
        'convnext_tiny': (16, 64), 'convnext_small': (8, 32), 'convnext_base': (4, 16),
        'vit_tiny': (8, 32), 'vit_small': (4, 16), 'vit_base': (2, 8),
        'deit_tiny': (8, 32), 'deit_small': (4, 16), 'deit_base': (2, 8),
        'swin_tiny': (8, 32), 'swin_small': (4, 16)
    }
    for key, (low, high) in manual_bs_table.items():
        if key in name:
            return low if usage_ratio < 0.5 else high
    return 8 if usage_ratio < 0.5 else 32

def setup_experiment_logger(experiment_name):
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(LOG_DIR, f"{experiment_name}.txt")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='experiment_log.csv')
    args = parser.parse_args()

    csv_headers = [
        'start_date', 'end_date', 'distributed_training', 'model_name', 'run_name', 'batch_size',
        'num_epochs', 'learning_rate', 'input_shape', 'gpu_type', 'total_gpus',
        'epoch_duration_sec', 'max_gpu_memory_mb', 'activation_functions',
        'num_parameters', 'num_classes', 'dropout_rate', 'accuracy', 'val_loss',
        'train_sample_count', 'val_sample_count', 'ERROR'
    ]

    if os.path.exists(args.csv_file):
        os.remove(args.csv_file)

    model_list = [
        'mobilenetv2_100', 'mobilenetv3_small_100', 'mobilenetv3_large_100',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'efficientnetv2_rw_t', 'efficientnetv2_rw_s', 'efficientnetv2_rw_m',
        'densenet121', 'densenet161', 'densenet201',
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
        'squeezenet1_0', 'squeezenet1_1',
        'mnasnet_0_5', 'mnasnet_1_0',
        'regnetx_002', 'regnetx_004', 'regnetx_006',
        'regnety_002', 'regnety_004',
        'convnext_tiny', 'convnext_base', 'convnext_small',
        'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
        'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'
    ]  # selected small to medium models manually
    varied_input_models = {
        'vit_base_patch16_384', 'vit_large_patch16_384',
        'swin_large_patch4_window12_384', 'swin_base_patch4_window12_384',
        'deit_base_patch16_384', 'beit_large_patch16_512',
        'vit_small_patch32_384', 'vit_huge_patch14_384',
        'deit_tiny_patch16_384', 'deit_small_patch16_384',
        'beit_base_patch16_384', 'beit_base_patch16_512',
        'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224',
        'convnext_base', 'convnext_large', 'convnext_xlarge',
        'efficientnet_b6', 'efficientnet_b7',
        'regnety_128gf', 'regnety_160gf', 'regnety_320gf'
    }  # keep full list
    gpu_configs = [1, 2, 4]

    for model_name in model_list[:200]:  # increased model sweep
        shape_key = "large" if "384" in model_name else "xlarge" if "512" in model_name else "default"
        input_shape = input_shapes[shape_key]

        for gpus in gpu_configs:
            for train_subset_size in [500, 10000]:
                val_subset_size = 100
            for num_epochs in [3, 6]:
                for usage_ratio, tag in [(0.2, "lowgpu"), (0.9, "highgpu")]:
                    run_name = f"{model_name}_{tag}_{gpus}gpu_{train_subset_size}train_{num_epochs}epochs"
                    cfg = {
                    'run_name': run_name,
                    'model_name': model_name,
                    'input_shape': input_shape,
                    'num_classes': 1000,
                    'dropout': 0.5,
                    'lr': 0.01,
                    'epochs': num_epochs,
                    'train_subset': train_subset_size,
                    'val_subset': val_subset_size,
                    'world_size': gpus,
                    'csv_file': args.csv_file,
                    'csv_headers': csv_headers,
                    'gpu_type': "Quadro RTX 6000",
                    'gpu_usage_ratio': usage_ratio
                }
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                print(f"Launching: {cfg['run_name']} | GPUs: {gpus} | Train size: {train_subset_size} | GPU usage target: {int(usage_ratio * 100)}%")
                mp.spawn(train_eval_worker, args=(cfg,), nprocs=gpus)

def train_eval_worker(rank, cfg):
    logger = setup_experiment_logger(cfg['run_name'])
    dist.init_process_group(backend='nccl', init_method='env://', world_size=cfg['world_size'], rank=rank)
    torch.cuda.set_device(rank)
    start_time = datetime.datetime.now()
    try:
        print(f"[Rank {rank}] Loading model: {cfg['model_name']}")
        model = get_model_from_timm(cfg['model_name'], cfg['num_classes']).to(rank)
        num_params = sum(p.numel() for p in model.parameters())
        activation_funcs = extract_activation_functions(model)
        batch_size = estimate_batch_size(cfg['model_name'], usage_ratio=cfg.get('gpu_usage_ratio', 0.9))
        model = DDP(model, device_ids=[rank])
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'])
        criterion = nn.CrossEntropyLoss()
        train_dataset = get_imagenet_train_dataset(cfg['input_shape'])
        val_dataset = get_imagenet_val_dataset(cfg['input_shape'])
        train_subset = Subset(train_dataset, list(range(cfg['train_subset'])))
        val_subset = Subset(val_dataset, list(range(cfg['val_subset'])))
        train_sampler = DistributedSampler(train_subset, num_replicas=cfg['world_size'], rank=rank)
        val_sampler = DistributedSampler(val_subset, num_replicas=cfg['world_size'], rank=rank)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler)
        print(f"[Rank {rank}] Starting training with batch size {batch_size} and input shape {cfg['input_shape']}")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(cfg['epochs']):
            train_sampler.set_epoch(epoch)
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                print(f"[Rank {rank}] Epoch {epoch+1}/{cfg['epochs']} - Batch {batch_idx+1}/{len(train_loader)}")
                inputs, labels = inputs.cuda(rank), labels.cuda(rank)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(rank), labels.cuda(rank)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100. * correct / total
        end_time = datetime.datetime.now()
        if rank == 0:
            append_to_csv({
                'start_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_date': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'distributed_training': f"Distributed Training on {cfg['world_size']} GPU(s)",
                'model_name': cfg['model_name'],
                'run_name': cfg['run_name'],
                'batch_size': batch_size,
                'num_epochs': cfg['epochs'],
                'learning_rate': cfg['lr'],
                'input_shape': str(cfg['input_shape']),
                'gpu_type': cfg['gpu_type'],
                'total_gpus': cfg['world_size'],
                'epoch_duration_sec': (end_time - start_time).total_seconds(),
                'max_gpu_memory_mb': torch.cuda.max_memory_allocated(rank) / (1024*1024),
                'activation_functions': activation_funcs,
                'num_parameters': num_params,
                'num_classes': cfg['num_classes'],
                'dropout_rate': cfg['dropout'],
                'accuracy': acc,
                'val_loss': val_loss,
                'train_sample_count': cfg['train_subset'],
                'val_sample_count': cfg['val_subset'],
                'ERROR': ''
            }, cfg['csv_file'], cfg['csv_headers'])
    except Exception as e:
        end_time = datetime.datetime.now()
        if rank == 0:
            append_to_csv({
                'start_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_date': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'distributed_training': f"Distributed Training on {cfg['world_size']} GPU(s)",
                'model_name': cfg['model_name'],
                'run_name': cfg['run_name'],
                'batch_size': 'OOM',
                'num_epochs': cfg['epochs'],
                'learning_rate': cfg['lr'],
                'input_shape': str(cfg['input_shape']),
                'gpu_type': cfg['gpu_type'],
                'total_gpus': cfg['world_size'],
                'epoch_duration_sec': '',
                'max_gpu_memory_mb': '',
                'activation_functions': 'load_failed',
                'num_parameters': 0,
                'num_classes': cfg['num_classes'],
                'dropout_rate': cfg['dropout'],
                'accuracy': 0,
                'val_loss': 'OOM_ERROR',
                'train_sample_count': cfg['train_subset'],
                'val_sample_count': cfg['val_subset'],
                'ERROR': str(e)
            }, cfg['csv_file'], cfg['csv_headers'])
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
