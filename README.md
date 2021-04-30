Code to accompany the paper [Applicability of random matrix theory in deep learning](https://arxiv.org/abs/2102.06740).

## Set-up

```pip install -r requirements.txt```

## Running the code

### Generating resnet34 CIFAR10 embedded dataset
The python script `create_deepfeature_dataset.py` will uses PyTorch to download pre-trained Resnet34 weights, remove the final (classification) layer and embed all of the train and test set of CIFAR10. The results are written as numpy arrays to disk.

There is a bash script `experiment_scripts/create_deepfeature.sh` that was used by the authors to run the above Python script on an HPC with GPU nodes using PBS. This is platform dependent.


### Training models
The training is done with GadamX, a very paired-down version of which is included here. `GadamX/run_sgd.py` when called with `Logistic` argument will train a logisitc regression on a specified dataset. The scripts `experiment_scripts/run_sgd_*.sh` show how this script was used to train the models used in the paper. Again, the script is platform specific, but the logic of calling the `run_sgd.py` script can be used on any platform.


### Extracting hessian spectra
The python script `GadamX/rmt/get_exact_hessian_spectra.py` loads in pre-trained models and computes the hessian over a dataset in specified batch sizes. The hessian eigenvalues are computed and saved in HDF5 format. The scripts `experiments_scripts/get_exact_hessian_spectra_*.sh` were used by the authors to obtain the required spectra used in the paper.

### Analysing the spectra
Once the above has been completed, one should have hessian spectra for the different models and conditions storred in HDF5 format on disk. The notebooks in `notebooks` read these spectra in and do unfolding and spacing ratio calculations to generate the figures in the paper. 

### Exponential hardness of the true loss
The eponymous directory under `notebook` contains the notebook to produce the results matching true loss curves for ImageNet to exponentials. We used the [script](https://github.com/pytorch/examples/tree/master/imagenet) in the PyTorch examples to train on ImageNet.
