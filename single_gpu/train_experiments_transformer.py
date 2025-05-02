# train_experiments_transformer.py

import os
import random
import itertools
import torch
from train_experiments_template import train_one_experiment, HEADERS, append_to_csv

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('logs', exist_ok=True)
log_file = "logs/log_transformer.csv"

# Transformer Model List (ViT, Swin, DeiT, ConvNeXt)
models_list = [
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_384',
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_384',
    'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window12_384',
    'convnext_tiny', 'convnext_small', 'convnext_base',
    'coat_tiny', 'coat_lite_small', 'levit_128s', 'levit_128', 'levit_192'
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

    real_imagenet = False  # Use Dummy dataset for Transformers

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
                'num_classes': 1000  # Auto corrected after loading model
            }
            print(f"Running: {run_name}")
            train_one_experiment(config, log_file, real_imagenet)

    print("Finished all Transformer model experiments!")

if __name__ == "__main__":
    main()
