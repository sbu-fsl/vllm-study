# train_experiments_large.py

import os
import random
import itertools
import torch
from train_experiments_template import train_one_experiment, HEADERS, append_to_csv

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('logs', exist_ok=True)
log_file = "logs/log_large.csv"

# Large/Heavy Model List (Big ViTs, Swin Large, ConvNeXt)
models_list = [
    'vit_base_patch16_384', 'vit_large_patch16_384', 'vit_huge_patch14_224',
    'deit_base_patch16_384', 'beit_base_patch16_384', 'beit_large_patch16_512',
    'swin_base_patch4_window7_224', 'swin_large_patch4_window12_384',
    'convnext_base', 'convnext_large', 'convnext_xlarge',
    'regnety_016', 'regnety_032', 'regnety_040'
]

input_shapes = [(3, 384, 384), (3, 512, 512)]
learning_rates = [0.0001, 0.001, 0.01]
dropouts = [0.0, 0.1, 0.3]
epochs_list = [2, 4, 6]
train_samples = [500, 2000, 5000]
val_samples = [100, 200]

# Main
def main():
    if os.path.exists(log_file):
        os.remove(log_file)

    real_imagenet = False  # Dummy dataset for large models (to avoid huge load)

    # Generate all combinations
    for model_name in models_list:
        combinations = list(itertools.product(
            learning_rates, dropouts, input_shapes, train_samples, val_samples, epochs_list
        ))
        random.shuffle(combinations)

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
                'num_classes': 1000  # Will correct based on model
            }
            print(f"Running: {run_name}")
            train_one_experiment(config, log_file, real_imagenet)

    print("Finished all Large model experiments!")

if __name__ == "__main__":
    main()
