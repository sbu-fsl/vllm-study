#!/bin/bash

git clone https://github.com/amirhnajafiz/prometheus-digger.git
cd prometheus-digger

make

cp promdigger /tmp/promdigger
sudo mv /tmp/promdigger /usr/local/bin/promdigger

cd ..
rm -rf prometheus-digger

sudo chmod +x /usr/local/bin/promdigger
