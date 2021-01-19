#!/bin/bash


#PBS -l select=1:ncpus=2:ngpus=1:ssd=true:mem=25gb


module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/nn-spectral-spacings

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1



# Execute code
python create_deepfeature_dataset.py --bs 64 --outdir $WORK/data/CIFAR10-resnet50 --data $TMPDIR/data/CIFAR10

