#!/bin/bash

echo "Aborting ..."

kubectl delete configmap gpu-script --ignore-not-found
kubectl delete -f job.yaml --ignore-not-found

