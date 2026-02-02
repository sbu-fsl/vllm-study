#!/bin/bash

# Kube config
mkdir -p $HOME/.kube
cp /etc/common/.kube_config /tmp/kube_config
mv /tmp/kube_config $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# SSH config
mkdir -p $HOME/.ssh
cp /etc/common/.ssh_config /tmp/ssh_config
mv /tmp/ssh_config $HOME/.ssh/config

chown -R $(id -u):$(id -g) $HOME/.ssh
chmod 600 $HOME/.ssh/config

# Custom configs
cp /etc/common/config.json /tmp/config.json
mv /tmp/config.json $HOME/config.json
chown $(id -u):$(id -g) $HOME/config.json
chmod 600 $HOME/config.json
