#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_experiments_imagenet.py &
CUDA_VISIBLE_DEVICES=1 python train_experiments_transformer.py &
CUDA_VISIBLE_DEVICES=2 python train_experiments_custom.py &
CUDA_VISIBLE_DEVICES=3 python train_experiments_large.py &
wait
