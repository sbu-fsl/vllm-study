#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIGURATION
########################################

CACHE_DEVICE="/dev/nvme0n1"   # CHANGE THIS
MOUNT_POINT="/mnt/lmcache"
FILESYSTEM="xfs"

########################################
# REQUIRE ROOT
########################################

if [[ $EUID -ne 0 ]]; then
  echo "Run as root"
  exit 1
fi

echo "==== Installing Required Packages ===="

# Ubuntu / Debian
apt update
apt install -y xfsprogs nvidia-fabricmanager dkms build-essential \
               nvidia-driver-535 nvidia-gds

echo "==== Loading NVIDIA GDS Kernel Module ===="

modprobe nvidia-fs || true

if ! lsmod | grep -q nvidia_fs; then
  echo "WARNING: nvidia-fs not loaded. Reboot may be required."
fi

echo "==== Formatting Disk as XFS (if not formatted) ===="

if ! blkid "$CACHE_DEVICE" > /dev/null 2>&1; then
  mkfs.xfs -f "$CACHE_DEVICE"
fi

echo "==== Mounting Disk ===="

mkdir -p "$MOUNT_POINT"

if ! mount | grep -q "$MOUNT_POINT"; then
  mount "$CACHE_DEVICE" "$MOUNT_POINT"
fi

echo "$CACHE_DEVICE  $MOUNT_POINT  xfs  defaults,noatime,nodiratime  0 0" >> /etc/fstab

echo "==== Creating LMCache Directories ===="

mkdir -p /mnt/lmcache/localstorage/cache
mkdir -p /mnt/lmcache/gds/cache
mkdir -p /mnt/lmcache/nixl/cache/posix
mkdir -p /mnt/lmcache/nixl/cache/gds

echo "==== Setting Permissions ===="

chmod -R 777 /mnt/lmcache/localstorage
chmod -R 777 /mnt/lmcache/gds
chmod -R 777 /mnt/lmcache/nixl

echo "==== Verifying GDS Installation ===="

if command -v gdscheck &> /dev/null; then
  gdscheck -p
else
  echo "gdscheck not found. GDS may not be installed correctly."
fi

echo "==== Setup Complete ===="
