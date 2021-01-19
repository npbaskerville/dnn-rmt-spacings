#!/bin/bash


#PBS -l walltime=48:00:00 -l select=1:ncpus=2:ngpus=1:ssd=true:mem=25gb


module load lang/gcc/9.1.0 
module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/nn-spectral-spacings/GadamX

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1



# Execute code
python run_sgd.py --dir $WORK/nn_spectral_models/MNIST_MLP --data_path $WORK/data/MNIST --model MLP --dataset MNIST --use_test


