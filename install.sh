#!/usr/bin/bash

# pytorch 1.3.1 is needed as some of our pre-trained models are old
conda install -y pytorch=1.3.1 cudatoolkit=10.1 -c pytorch
conda install -y nltk
conda install -y scipy
conda install -y scikit-learn
conda install -y -c anaconda sqlite 
conda install -y -c conda-forge ipdb 
conda install -y tqdm

pip install pyyaml
pip install colored_traceback

pip install dataset
pip install munkres
pip install ftfy
pip install tensorboard


cd ~
mkdir data
mkdir -p results/PEER
cd data 
wget https://zenodo.org/record/4478267/files/PEER.zip
unzip PEER.zip
