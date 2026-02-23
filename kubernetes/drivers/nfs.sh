#!/bin/bash

NFS_IP=$1

helm install nfs-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
  --set nfs.server=$NFS_IP \
  --set nfs.path=/workloads/sunyibm/data \
  --set storageClass.defaultClass=true \
  --set storageClass.reclaimPolicy=Delete \
  --set securityContext.runAsUser=3884 \
  --namespace=kube-storage
