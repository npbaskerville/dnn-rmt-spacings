#!/bin/bash

#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=100gb


module load lang/gcc/9.1.0 
module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/nn-spectral-spacings/GadamX/rmt

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1



# Execute code
python get_exact_hessian_spectra.py --data_path $WORK/data/cifar10 --model LeNet --dataset CIFAR10 --checkpoint=/work/jr19127/nn_spectral_models/CIFAR10/LeNetsmallCIFAR10/LeNet/SGD/seed\=1_lr\=0.003_mom\=0.9_wd\=0.0005_batchsize\=128_numepochs\=300/checkpoint-00300.pt --out $WORK/nn_spectral_models/hessian_spectra/CIFAR10/LeNetsmall/bs64_test.hdf5 -test 
