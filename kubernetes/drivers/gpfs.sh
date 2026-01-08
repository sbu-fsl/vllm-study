#!/bin/bash

helm upgrade --install gpfs-nfs-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  --set nfs.server=130.245.176.57 \
  --set nfs.path=/gpfs/fs1/suny-ibm \
  --set storageClass.name=gpfs-nfs \
  --set storageClass.defaultClass=false \
  --set storageClass.reclaimPolicy=Delete \
  --namespace=kube-storage
