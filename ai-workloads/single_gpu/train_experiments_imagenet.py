# train_experiments_imagenet.py

import os
import random
import itertools
import torch
from train_experiments_template import train_one_experiment, HEADERS, append_to_csv

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('logs', exist_ok=True)
log_file = "logs/log_imagenet.csv"

# Model List (real ImageNet style CNNs)
models_list = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'mobilenetv2_100', 'mobilenetv3_small_100', 'mobilenetv3_large_100',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'densenet121', 'densenet161', 'densenet201',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'regnetx_002', 'regnety_004'
]

input_shapes = [(3, 224, 224), (3, 384, 384), (3, 512, 512)]
learning_rates = [0.0001, 0.001, 0.01, 0.1]
dropouts = [0.0, 0.1, 0.3, 0.5]
epochs_list = [2, 4, 6]
train_samples = [500, 2000, 5000, 10000]
val_samples = [100, 200, 500]

# Main
def main():
    if os.path.exists(log_file):
        os.remove(log_file)

    real_imagenet = True  # Use real dataset

    # Generate all combinations
    for model_name in models_list:
        combinations = list(itertools.product(
            learning_rates, dropouts, input_shapes, train_samples, val_samples, epochs_list
        ))
        random.shuffle(combinations)  # Shuffle for randomness in load

        for lr, do, shape, train_s, val_s, num_epochs in combinations:
            run_name = f"{model_name}_bsAUTO_lr{lr}_do{do}_train{train_s}_epoch{num_epochs}"
            config = {
                'model_name': model_name,
                'run_name': run_name,
                'learning_rate': lr,
                'dropout_rate': do,
                'input_shape': shape,
                'train_sample_count': train_s,
                'val_sample_count': val_s,
                'num_epochs': num_epochs,
                'num_classes': 1000  # Will auto-correct dynamically
            }
            print(f"Running: {run_name}")
            train_one_experiment(config, log_file, real_imagenet)

    print("Finished all ImageNet model experiments!")

if __name__ == "__main__":
    main()
