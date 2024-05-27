#!/bin/bash

wget https://figshare.com/ndownloader/files/25635053 -O data/mnist.pkl.gz
wget https://github.com/nbereux/dataset/raw/main/1kg_xtrain.d.zip -O data/1kg_xtrain.d.zip
unzip data/1kg_xtrain.d.zip -d data/
rm data/1kg_xtrain.d.zip
wget https://github.com/nbereux/dataset/raw/main/mickey.npy.zip -O data/mickey.npy.zip
unzip data/mickey.npy.zip -d data/
rm data/mickey.npy.zip


