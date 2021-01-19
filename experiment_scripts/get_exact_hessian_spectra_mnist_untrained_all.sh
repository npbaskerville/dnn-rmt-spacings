#!/bin/bash


#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=95gb


module load lang/gcc/9.1.0 
module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/nn-spectral-spacings/GadamX/rmt

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1



# Execute code
python get_exact_hessian_spectra.py --data_path $WORK/data/MNIST --model Logistic --dataset MNIST --checkpoint=$WORK/nn_spectral_models/MNIST_untrainedMNIST/Logistic/SGD/seed\=1_lr\=0.003_mom\=0.9_wd\=0.0005_batchsize\=128_numepochs\=0/checkpoint-00000.pt --out $WORK/nn_spectral_models/hessian_spectra/MNIST_untrained/bsall_train.hdf5 --bs -1
