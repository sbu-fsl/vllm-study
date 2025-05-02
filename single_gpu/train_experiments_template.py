import os
import random
import time
import datetime
import csv
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('logs', exist_ok=True)
gpu_type = torch.cuda.get_device_name(0)

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples, input_shape, num_classes):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randn(self.input_shape)
        y = random.randint(0, self.num_classes - 1)
        return x, y

# ImageNet Transform
def imagenet_transform(input_shape):
    size = max(input_shape[1:])
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Dataset Loader
def get_train_dataset(input_shape, real_imagenet, num_samples, num_classes):
    if real_imagenet:
        return datasets.ImageFolder('/home/ImageNet-1k/train', transform=imagenet_transform(input_shape))
    else:
        return DummyDataset(num_samples, input_shape, num_classes)

def get_val_dataset(input_shape, real_imagenet, num_samples, num_classes):
    if real_imagenet:
        return datasets.ImageFolder('/home/ImageNet-1k/val', transform=imagenet_transform(input_shape))
    else:
        return DummyDataset(num_samples, input_shape, num_classes)

# Logger
def append_to_csv(log_path, row, headers):
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Batch Size Finder
def find_batch_size(model, input_shape, device, max_memory_gb=22):
    bs = 512
    while bs >= 8:
        try:
            dummy_input = torch.randn((bs,) + input_shape).to(device)
            model(dummy_input)
            torch.cuda.reset_peak_memory_stats(device)
            del dummy_input
            torch.cuda.empty_cache()
            return bs
        except Exception:
            bs //= 2
            torch.cuda.empty_cache()
    return 8

# Extract Activation Functions
def extract_activation_functions(model):
    activations = set()
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax, nn.LeakyReLU, nn.Tanh)):
            activations.add(m.__class__.__name__)
    return list(activations)

# Main Training
def train_one_experiment(config, log_file, real_imagenet):
    start_time = datetime.datetime.now()
    try:
        model = timm.create_model(config['model_name'], pretrained=False, num_classes=config['num_classes'], drop_rate=config['dropout_rate'])
        model = model.to(device)

        # Extract true num_classes dynamically
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
            num_classes = model.classifier.out_features
        elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            num_classes = model.head.out_features
        else:
            num_classes = 1000
        config['num_classes'] = num_classes

        activations = extract_activation_functions(model)
        num_params = sum(p.numel() for p in model.parameters())

        batch_size = find_batch_size(model, config['input_shape'], device)
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        train_dataset = get_train_dataset(config['input_shape'], real_imagenet, config['train_sample_count'], config['num_classes'])
        val_dataset = get_val_dataset(config['input_shape'], real_imagenet, config['val_sample_count'], config['num_classes'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        torch.cuda.reset_peak_memory_stats()
        model.train()

        train_start_time = time.time()
        for epoch in range(config['num_epochs']):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        train_end_time = time.time()

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024*1024)
        duration = (train_end_time - train_start_time) / config['num_epochs']

        end_time = datetime.datetime.now()

        append_to_csv(log_file, {
            'start_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_date': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': config['model_name'],
            'run_name': config['run_name'],
            'batch_size': batch_size,
            'num_epochs': config['num_epochs'],
            'learning_rate': config['learning_rate'],
            'input_shape': str(config['input_shape']),
            'gpu_type': gpu_type,
            'epoch_duration_sec': round(duration, 2),
            'max_gpu_memory_mb': round(peak_memory, 2),
            'activation_functions': activations,
            'num_parameters': num_params,
            'num_classes': config['num_classes'],
            'dropout_rate': config['dropout_rate'],
            'train_sample_count': config['train_sample_count'],
            'val_sample_count': config['val_sample_count']
        }, HEADERS)

    except Exception as e:
        print(f"Experiment failed: {e}")
        end_time = datetime.datetime.now()
        append_to_csv(log_file, {
            'start_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_date': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': config['model_name'],
            'run_name': config['run_name'],
            'batch_size': 'FAILED',
            'num_epochs': config['num_epochs'],
            'learning_rate': config['learning_rate'],
            'input_shape': str(config['input_shape']),
            'gpu_type': gpu_type,
            'epoch_duration_sec': '',
            'max_gpu_memory_mb': '',
            'activation_functions': '',
            'num_parameters': '',
            'num_classes': '',
            'dropout_rate': config['dropout_rate'],
            'train_sample_count': config['train_sample_count'],
            'val_sample_count': config['val_sample_count']
        }, HEADERS)

# Headers
HEADERS = [
    'start_date', 'end_date', 'model_name', 'run_name', 'batch_size', 'num_epochs',
    'learning_rate', 'input_shape', 'gpu_type', 'epoch_duration_sec', 'max_gpu_memory_mb',
    'activation_functions', 'num_parameters', 'num_classes', 'dropout_rate',
    'train_sample_count', 'val_sample_count'
]

# Your model list, input_shapes, lrs, dropouts, etc. would be defined per script

