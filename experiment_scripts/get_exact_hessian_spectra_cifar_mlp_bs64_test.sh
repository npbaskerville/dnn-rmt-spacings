#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=100gb

module load lang/gcc/9.1.0 
module load lang/cuda
module load lang/python/anaconda/3.8.8-2021.05-torch

pip install --user torchvision
cd /home/jr19127/nn-spectral-spacings/GadamX/rmt

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1



# Execute code
python get_exact_hessian_spectra.py --data_path $WORK/data/cifar10 --model MLP_CIF --dataset CIFAR10 --checkpoint=$WORK/nn_spectral_models/CIFAR10/MLP/SGD/checkpoint-00300.pt --out $WORK/nn_spectral_models/hessian_spectra/CIFAR10/MLP/bs64_test.hdf5 -test -fp16
