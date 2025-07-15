#!/bin/bash

# This script sets up a Kubernetes job to run a Python script that uses GPU resources.
script=$1
if [ -z "$script" ]; then
  echo "Usage: $0 <script>"
  exit 1
fi

script="$script"
if [ ! -f "$script" ]; then
  echo "Script $script not found!"
  exit 1
fi

echo "Setting up Kubernetes job to run $script"
cp $script gpu.py

kubectl delete configmap gpu-script --ignore-not-found
kubectl delete -f job.yaml --ignore-not-found

kubectl create configmap gpu-script --from-file=gpu.py
kubectl apply -f job.yaml

rm gpu.py
