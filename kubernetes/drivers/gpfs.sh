#!/bin/bash

NFS_IP=$1

helm upgrade --install gpfs-nfs-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  --set nfs.server=$NFS_IP \
  --set nfs.path=/gpfs/fs1/suny-ibm \
  --set storageClass.name=gpfs-nfs \
  --set storageClass.defaultClass=false \
  --set storageClass.reclaimPolicy=Delete \
  --namespace=kube-storage
