This code is a (very)  paired-down version of [GadamX](link to repo here) with a few small modifications to make it work with up-to-date torch and torchvision and use tensor datasets (needed to Resnet50 embedded CIFAR10).


What follows is the original README

# Curvature

```
#SETUP
python3 setup.py develop
we reccomend using torchvision 0.2.0 which is most stable, although the code can be adapted to work to 0.4.0
```
```
nvidia-smi -l (choose 1-3 which is free, where l is the number of seconds)
export CUDA_VISIBLE_DEVICES=1
```

## Examples

```bash
#SGD VGG16 - CIFAR100
##scasle the learning rate proportionally to the batch_size, i.e [0.05,128],[0.1,256],[0.2,512]
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --model=VGG16 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=0 --seed=1 --batch_size 128

#SGD VGG16BN - CIFAR100
##scasle the learning rate proportionally to the batch_size, i.e [0.1,128],[0.2,256],[0.4,512]
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --model=VGG16BN --epochs=300 --save_freq=50 --lr_init=0.1 --wd=0.0005 --seed=1 --batch_size 128

#SGD WideResNet - CIFAR100
##scasle the learning rate proportionally to the batch_size, i.e [0.1,128],[0.2,256],[0.4,512]
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --model=WideResNet28x10 --epochs=300 --save_freq=50 --lr_init=0.1 --wd=0.0005 --seed=1 --batch_size 128

#SGD WideResNet - ImageNet32 
##scasle the learning rate proportionally to the batch_size, i.e [0.1,128],[0.2,256],[0.4,512]
python3 run_sgd.py --dir=out/ --dataset=ImageNet32 --data_path=data/ --use_test --model=VGG16BN --epochs=300 --save_freq=50 --lr_init=0.1 --wd=0.0005 --seed=1 --batch_size 128


#KFAC experiments
##reduce damping by a factor proportional to the increase in batch size i.e [16,16],[8,32],[4,64],[2,128],[1,265], etc...
python3 run_KFAC.py --dataset CIFAR100 --data_path data/ --dir out/ --wd 0 --lr_init 1 --damping 16 --batch_size 16  --model VGG16 --epochs 300 --seed=1

#Adam experiments
##scale the learning rate as the square root of the batch size increase. i.e [0.0004,128],[0.00056]...
python3 run_adam.py --dataset CIFAR100 --data_path data/ --dir out/ --decoupled_wd --wd 0 --lr_init 0.0004 --model VGG16 --batch_size

#Spectrum
python3 spectrum.py --dataset=CIFAR100 --use_test --data_path=data/ --model=VGG16 --ckpt=./ckpts/c100/vgg16/sgd/run1/checkpoint-00050.pt --iters=100 --basis --curvature_matrix=hessian
##generalised gauss newton
python3 spectrum.py --dataset=CIFAR100 --use_test --data_path=data/ --model=VGG16 --ckpt=./ckpts/c100/vgg16/sgd/run1/checkpoint-00050.pt --iters=100 --basis --curvature_matrix=gn
##difference between the hessian and the generalised gauss newton
python3 spectrum.py --dataset=CIFAR100 --use_test --data_path=data/ --model=VGG16 --ckpt=./ckpts/c100/vgg16/sgd/run1/checkpoint-00050.pt --iters=100 --basis --curvature_matrix=nonggn

#Loss statistics (including hessian variance etc..)
##similarly choose the appropriate curvature matrix
python3 loss_stats.py --dataset=CIFAR100  --data_path=data/ --model=PreResNet110 --ckpt=./ckpts/c100/PreResNet110/OFIT/runshrink/checkpoint-00000.pt --stats_batch=256 --curvature_matrix=hessian


```

## SGD 

```bash
python3 experiments/swag/run_sgd.py \
                 --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --momentum=<MOM> \
                 --wd=<WD>                 
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```MOM``` &mdash; SGD momentum (default: 0.9)
* ```WD``` &mdash; weight decay (default: 1e-4)

## Spectrum

```bash
python3 experiments/lanczos/spectrum.py \
                 --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --ckpt=<CKPT> \
                 [--swag] \
                 --iters=<ITERS> \
                 --spectrum_path=<SPECTRUM_PATH> \
                 --basis_path=<BASIS_PATH>
```

* ```CKPT``` &mdash; path to model checkpoint (use ```--swag``` to load SWAG checkpoint)
* ```ITERS``` &mdash; number of Lanczos iterations

# Curvature

