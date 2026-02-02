#!/bin/bash

# clone in the prometheus-digger repository
git clone https://github.com/amirhnajafiz/prometheus-digger.git
cd prometheus-digger

# make sure we have go installed
if ! command -v go &> /dev/null
then
    echo "Go could not be found! Please install Go to proceed."
    exit 1
fi

# call make to build promdigger
make

# move the promdigger binary to /usr/local/bin
cp promdigger /tmp/promdigger
sudo mv /tmp/promdigger /usr/local/bin/promdigger

# cleanup
cd ..
rm -rf prometheus-digger

# make sure promdigger is executable
sudo chmod +x /usr/local/bin/promdigger
