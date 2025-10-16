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

# Import model libraries
import torchvision.models as tv_models
from torchvision import datasets, transforms
from transformers import ViTForImageClassification  # Example for Hugging Face ViT models

##################################
# Setup directories and constants
##################################
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

##################################
# CSV Append Utility Function
##################################
def append_to_csv(row, csv_filename, headers):
    # Open file in append mode, or create if not exists.
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        # If file did not exist, write the header first.
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

##################################
# ImageNet Dataset Setup
##################################
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_imagenet_train_dataset(root_dir='/home/ImageNet-1k/train'):
    return datasets.ImageFolder(root=root_dir, transform=imagenet_transform)

def get_imagenet_val_dataset(root_dir='/home/ImageNet-1k/val'):
    return datasets.ImageFolder(root=root_dir, transform=imagenet_transform)

##################################
# Dummy Dataset (Fallback)
##################################
class DummyImageNetDataset(Dataset):
    def __init__(self, num_samples=10000, input_shape=(3, 224, 224), num_classes=1000):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.input_shape)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

##################################
# Model Loader Functions
##################################
def load_torchvision_model(model_name, num_classes=1000):
    try:
        model_fn = getattr(tv_models, model_name)
        model = model_fn(pretrained=True)
        # Adjust final layer if needed.
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            # Adjustments for models like VGG, MobileNet may be necessary.
            pass
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading torchvision model {model_name}: {e}")

def load_huggingface_model(model_name, num_classes=1000):
    try:
        model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading Hugging Face model {model_name}: {e}")

##################################
# Extract Activation Functions
##################################
def extract_activation_functions(model):
    """
    Traverse the model architecture and return unique activation function names.
    """
    activations = set()
    activation_classes = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh,
                          nn.ELU, nn.PReLU, nn.SELU, nn.GELU, nn.Softplus, nn.Softmax)
    for m in model.modules():
        if isinstance(m, activation_classes):
            activations.add(m.__class__.__name__)
    return list(activations)

##################################
# Batch Size Heuristic Based on Number of Parameters
##################################
def calculate_batch_sizes(num_params):
    """
    Estimate lower and higher per-GPU batch sizes based on the number of parameters.
    Buckets (example heuristic):
      - <25M: (256, 512)
      - 25M-50M: (128, 256)
      - 50M-100M: (64, 128)
      - 100M-200M: (32, 64)
      - >200M: (16, 32)
    """
    if num_params < 25e6:
        return 256, 512
    elif num_params < 50e6:
        return 128, 256
    elif num_params < 100e6:
        return 64, 128
    elif num_params < 200e6:
        return 32, 64
    else:
        return 16, 32

##################################
# Logger Setup for Each Experiment
##################################
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

##################################
# DDP Training & Validation Worker Function
##################################
def ddp_train_worker(gpu, experiment_cfg, world_size):
    logger = setup_experiment_logger(experiment_cfg['NAME'])
    try:
        rank = experiment_cfg['global_rank'] + gpu
        logger.info(f"Initializing process group for rank {rank} on GPU {gpu}.")
        # Make sure MASTER_ADDR/Master_PORT are set (or use tcp://).
        dist.init_process_group(
            backend='nccl',
            init_method='env://',  # Alternatively: init_method='tcp://127.0.0.1:29500'
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(gpu)

        model_source = experiment_cfg['MODEL_SOURCE']
        model_name = experiment_cfg['MODEL_NAME']
        num_classes = experiment_cfg['NUM_CLASSES']
        logger.info(f"Loading model {model_source}:{model_name} on GPU {gpu}.")
        if model_source == 'torchvision':
            model = load_torchvision_model(model_name, num_classes)
        elif model_source == 'huggingface':
            model = load_huggingface_model(model_name, num_classes)
        else:
            raise ValueError(f"Unknown model source: {model_source}")
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])
        optimizer = optim.SGD(model.parameters(), lr=experiment_cfg['LEARNING_RATE'])
        criterion = nn.CrossEntropyLoss()

        # Log the full experiment configuration as backup.
        logger.info("Experiment configuration:")
        for key, value in experiment_cfg.items():
            logger.info(f"  {key}: {value}")

        # Load and subset the training dataset (10,000 images).
        try:
            full_train_dataset = get_imagenet_train_dataset()
            limited_train_dataset = Subset(full_train_dataset, list(range(10000)))
            logger.info("Using limited training dataset (10,000 images) from /home/ImageNet-1k/train.")
        except Exception as ex:
            logger.warning(f"Training dataset load failed: {ex}. Using dummy dataset instead.")
            limited_train_dataset = DummyImageNetDataset(num_samples=10000,
                                                          input_shape=experiment_cfg['INPUT_SHAPE'],
                                                          num_classes=num_classes)
        train_sampler = DistributedSampler(limited_train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(limited_train_dataset, batch_size=experiment_cfg['BATCH_SIZE'], sampler=train_sampler)

        # Record epoch durations (only rank 0).
        epoch_times = []
        if rank == 0:
            torch.cuda.reset_max_memory_allocated(gpu)

        logger.info("Starting training loop.")
        model.train()
        for epoch in range(experiment_cfg['NUM_EPOCHS']):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch+1}/{experiment_cfg['NUM_EPOCHS']} starting.")
            train_sampler.set_epoch(epoch)
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                logger.info(f"Rank {gpu}, Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
                # Stop after processing the limited dataset.
                if batch_idx >= (len(limited_train_dataset) // experiment_cfg['BATCH_SIZE']):
                    break
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            if rank == 0:
                epoch_times.append(epoch_duration)
            logger.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} sec.")
        logger.info("Training loop completed successfully.")

        # Load and subset the validation dataset (100 images).
        try:
            full_val_dataset = get_imagenet_val_dataset()
            limited_val_dataset = Subset(full_val_dataset, list(range(100)))
            logger.info("Using limited validation dataset (100 images) from /home/ImageNet-1k/val.")
        except Exception as ex:
            logger.warning(f"Validation dataset load failed: {ex}. Using dummy dataset instead.")
            limited_val_dataset = DummyImageNetDataset(num_samples=100,
                                                        input_shape=experiment_cfg['INPUT_SHAPE'],
                                                        num_classes=num_classes)
        val_sampler = DistributedSampler(limited_val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(limited_val_dataset, batch_size=experiment_cfg['BATCH_SIZE'], sampler=val_sampler)

        val_losses = []
        val_correct = 0
        total_val_samples = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                if batch_idx >= (len(limited_val_dataset) // experiment_cfg['BATCH_SIZE']):
                    break
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
        val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        val_acc = 100.0 * val_correct / total_val_samples if total_val_samples > 0 else 0.0
        logger.info(f"Validation: Avg Loss = {val_loss:.4f}, Accuracy = {val_acc:.2f}%.")
        logger.info("Validation loop completed successfully.")

        # Only rank 0 writes metrics.
        if rank == 0:
            avg_epoch_duration = sum(epoch_times)/len(epoch_times) if epoch_times else 0.0
            max_memory_mb = torch.cuda.max_memory_allocated(gpu) / (1024*1024)
            metrics_file = os.path.join(LOG_DIR, f"{experiment_cfg['NAME']}_metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write(f"epoch_duration_sec={avg_epoch_duration}\n")
                f.write(f"max_gpu_memory_mb={max_memory_mb}\n")
            logger.info(f"Metrics saved: avg_epoch_duration_sec={avg_epoch_duration:.2f}, max_gpu_memory_mb={max_memory_mb:.2f}.")
        dist.destroy_process_group()
    except Exception as e:
        error_msg = f"Rank {gpu}: {str(e)}\n{traceback.format_exc()}"
        experiment_cfg['ERROR'] = error_msg
        logger.error("Error during training/validation: " + error_msg)
        try:
            dist.destroy_process_group()
        except Exception:
            pass

##################################
# Run Experiment and Logging
##################################
def run_experiment(experiment_cfg):
    logger = setup_experiment_logger(experiment_cfg['NAME'])
    world_size = experiment_cfg['GPU_COUNT']
    experiment_cfg['global_rank'] = 0
    start_time = datetime.datetime.now()
    experiment_cfg['STARTED_AT'] = start_time.strftime("%Y-%m-%d %H:%M:%S,%f")
    logger.info(f"Experiment {experiment_cfg['NAME']} started at {experiment_cfg['STARTED_AT']}.")
    try:
        mp.spawn(
            ddp_train_worker,
            args=(experiment_cfg, world_size),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        experiment_cfg['ERROR'] = f"mp.spawn error: {str(e)}\n{traceback.format_exc()}"
        logger.error("mp.spawn error: " + experiment_cfg['ERROR'])
    finish_time = datetime.datetime.now()
    experiment_cfg['FINISHED_AT'] = finish_time.strftime("%Y-%m-%d %H:%M:%S,%f")
    logger.info(f"Experiment {experiment_cfg['NAME']} finished at {experiment_cfg['FINISHED_AT']}.")
    # Read metrics from file (if present)
    metrics_file = os.path.join(LOG_DIR, f"{experiment_cfg['NAME']}_metrics.txt")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.startswith("epoch_duration_sec="):
                    experiment_cfg['epoch_duration_sec'] = float(line.split("=")[1].strip())
                if line.startswith("max_gpu_memory_mb="):
                    experiment_cfg['max_gpu_memory_mb'] = float(line.split("=")[1].strip())
    else:
        experiment_cfg['epoch_duration_sec'] = ""
        experiment_cfg['max_gpu_memory_mb'] = ""
    # Additional fields for CSV summary.
    experiment_cfg['distributed_training'] = f"Distributed Training on {experiment_cfg['GPU_COUNT']} GPU(s)"
    experiment_cfg['model_name'] = experiment_cfg['MODEL_NAME']
    experiment_cfg['run_name'] = experiment_cfg['NAME']
    experiment_cfg['total_gpus'] = experiment_cfg['GPU_COUNT']
    return experiment_cfg

##################################
# Main Experiment Loop
##################################
def main():
    parser = argparse.ArgumentParser(description="Run distributed experiments with multiple pretrained models.")
    parser.add_argument('--logfile', type=str, default='experiment_log.csv', help='CSV file to store experiment summary logs')
    args = parser.parse_args()

    gpu_type = "Quadro RTX 6000"  # 24GB each
    gpu_counts = [1, 2, 4]
    num_epochs_options = [3, 6]
    learning_rate = 0.01
    input_shape = [3, 224, 224]  # using list for CSV logging consistency.
    num_classes = 1000
    dropout_rate = 0.5
    max_sequence_length = 512
    # Using subset sizes:
    train_steps = 10000  # training subset size
    val_steps = 100      # validation subset size

    # Define at least 30 model entries.
    model_list = [
        # Torchvision models:
        # {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'resnet18'},
        # {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'resnet34'},
        # {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'resnet50'},
        # {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'resnet101'},
        # {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'resnet152'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'vgg11'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'vgg13'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'vgg16'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'vgg19'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'densenet121'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'densenet169'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'densenet201'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'mobilenet_v2'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'mobilenet_v3_large'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'mobilenet_v3_small'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'inception_v3'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'googlenet'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'shufflenet_v2_x1_0'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'mnasnet0_5'},
        {'MODEL_SOURCE': 'torchvision', 'MODEL_NAME': 'efficientnet_b0'},
        # Hugging Face models:
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'google/vit-base-patch16-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'google/vit-base-patch16-384'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'google/vit-large-patch16-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'google/vit-large-patch16-384'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'facebook/deit-base-distilled-patch16-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'facebook/deit-small-distilled-patch16-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'microsoft/swin-tiny-patch4-window7-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'microsoft/swin-small-patch4-window7-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'microsoft/swin-base-patch4-window7-224'},
        {'MODEL_SOURCE': 'huggingface', 'MODEL_NAME': 'microsoft/swin-large-patch4-window7-224'},
    ]

    # Headers for the CSV summary.
    csv_headers = [
        "start_date", "end_date", "distributed_training", "model_name",
        "run_name", "batch_size", "num_epochs", "learning_rate", "input_shape",
        "gpu_type", "total_gpus", "epoch_duration_sec", "max_gpu_memory_mb", "ERROR"
    ]

    experiment_logs = []

    # CSV summary file name.
    csv_filename = args.logfile
    # If the CSV file exists, remove it to start fresh.
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    for model_entry in model_list:
        model_source = model_entry['MODEL_SOURCE']
        model_name = model_entry['MODEL_NAME']
        try:
            if model_source == 'torchvision':
                model = load_torchvision_model(model_name, num_classes)
            elif model_source == 'huggingface':
                model = load_huggingface_model(model_name, num_classes)
            else:
                raise ValueError(f"Unknown model source: {model_source}")
            num_params = sum(p.numel() for p in model.parameters())
            activation_functions = extract_activation_functions(model)
        except Exception as e:
            log_entry = {
                'start_date': '',
                'end_date': '',
                'distributed_training': f"Distributed Training on {gpu_counts[0]} GPU(s)",
                'model_name': model_name,
                'run_name': f"{model_source}_{model_name}",
                'batch_size': '',
                'num_epochs': '',
                'learning_rate': learning_rate,
                'input_shape': str(input_shape),
                'gpu_type': gpu_type,
                'total_gpus': '',
                'epoch_duration_sec': '',
                'max_gpu_memory_mb': '',
                'ERROR': str(e)
            }
            experiment_logs.append(log_entry)
            append_to_csv(log_entry, csv_filename, csv_headers)
            print(f"Skipping model {model_name} due to error: {e}")
            continue

        # Calculate batch sizes dynamically.
        low_bs, high_bs = calculate_batch_sizes(num_params)

        for gpu_count in gpu_counts:
            for bs_label, bs in [('low', low_bs), ('high', high_bs)]:
                for num_epochs in num_epochs_options:
                    experiment_name = f"{model_source}_{model_name}_{gpu_count}gpu_{bs_label}_e{num_epochs}"
                    experiment_cfg = {
                        'NAME': experiment_name,
                        'MODEL_SOURCE': model_source,
                        'MODEL_NAME': model_name,
                        'GPU_TYPE': gpu_type,
                        'GPU_COUNT': gpu_count,
                        'BATCH_SIZE': bs,   # From heuristic.
                        'NUM_EPOCHS': num_epochs,
                        'LEARNING_RATE': learning_rate,
                        'INPUT_SHAPE': input_shape,
                        'NUM_CLASSES': num_classes,
                        'DROPOUT_RATE': dropout_rate,
                        'ACTIVATION_FUNCTIONS': activation_functions,
                        'MAX_SEQUENCE_LENGTH': max_sequence_length,
                        'NUM_PARAMETERS': num_params,
                        'TRAIN_STEPS': train_steps,
                        'VAL_STEPS': val_steps,
                        'ERROR': ''
                    }
                    print(f"Starting experiment: {experiment_name}")
                    try:
                        result_cfg = run_experiment(experiment_cfg)
                    except Exception as e:
                        result_cfg = experiment_cfg
                        result_cfg['ERROR'] = f"Top-level error: {str(e)}\n{traceback.format_exc()}"
                        print(f"Error in experiment {experiment_name}: {e}")
                    experiment_logs.append(result_cfg)
                    print(f"Finished experiment: {experiment_name}")
                    # Map our result dictionary to our CSV header names.
                    csv_row = {
                        "start_date": result_cfg.get("STARTED_AT", ""),
                        "end_date": result_cfg.get("FINISHED_AT", ""),
                        "distributed_training": result_cfg.get("distributed_training", f"Distributed Training on {result_cfg.get('GPU_COUNT', '')} GPU(s)"),
                        "model_name": result_cfg.get("MODEL_NAME", result_cfg.get("model_name", "")),
                        "run_name": result_cfg.get("NAME", result_cfg.get("run_name", "")),
                        "batch_size": result_cfg.get("BATCH_SIZE", ""),
                        "num_epochs": result_cfg.get("NUM_EPOCHS", ""),
                        "learning_rate": result_cfg.get("LEARNING_RATE", ""),
                        "input_shape": str(result_cfg.get("INPUT_SHAPE", "")),
                        "gpu_type": result_cfg.get("GPU_TYPE", ""),
                        "total_gpus": result_cfg.get("GPU_COUNT", result_cfg.get("total_gpus", "")),
                        "epoch_duration_sec": result_cfg.get("epoch_duration_sec", ""),
                        "max_gpu_memory_mb": result_cfg.get("max_gpu_memory_mb", ""),
                        "ERROR": result_cfg.get("ERROR", "")
                    }
                    append_to_csv(csv_row, csv_filename, csv_headers)
                    print(f"CSV updated for experiment: {experiment_name}")

    print(f"All experiment summary logs are saved in {csv_filename}")

if __name__ == "__main__":
    main()
