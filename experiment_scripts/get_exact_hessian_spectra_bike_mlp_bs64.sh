#!/bin/bash

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=150gb


module load lang/gcc/9.1.0 
module load lang/cuda
module load lang/python/anaconda/pytorch

cd /home/jr19127/nn-spectral-spacings/GadamX/rmt

# Do some stuff

# PBS ARRAY INDEX is 1-up, but bash arrays are 0-up, so leading pad with -1

# Execute code
python get_exact_hessian_spectra.py --data_path $WORK/data/Bike --model MLP_Bike --dataset Bike --checkpoint=$WORK/nn_spectral_models/Bike_MLPBike/MLP_Bike/SGD/seed\=1_lr\=0.003_mom\=0.9_wd\=0.0005_batchsize\=128_numepochs\=300/checkpoint-00300.pt --out $WORK/nn_spectral_models/hessian_spectra/Bike/MLP/bs64.hdf5 -regression
