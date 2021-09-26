import argparse
import os
import sys
import time
import tabulate

import torch
from torch.utils.data import DataLoader, TensorDataset

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA
import numpy as np

#add standard imagenetmodels
import torchvision.models as modelstorch
from torchvision import transforms
resnet50 = modelstorch.resnet50(pretrained=False)
resnet18 = modelstorch.resnet18(pretrained=False)

print('numpy imported')

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default="data/",
                    required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default="data/", required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume  from (default: None)')
parser.add_argument('--reset_resume', action='store_true')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.003, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr_r', type=float, default=0.01, help='learning rate ratio. Default to 0.01')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')
parser.add_argument("--normalized_wd", action='store_true',
                    help='Whether to use normalised wd. WD = WD_norm \sqrt(\frac{b}{BT})')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')
parser.add_argument('--step_schedule', action='store_true')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--cosine_annealing', action='store_true')
parser.add_argument('--swag', action='store_true', help='activate SWA-Gaussian')
parser.add_argument("--linear_annealing", action='store_true')
parser.add_argument("--new_schedule", action='store_true')

# Xingchen Addition
parser.add_argument("--no_covariance", action='store_true',
                    help='Do not use Gaussian covariance in SWAG - essentially use SWA')

parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--schedule_swag_lr', action='store_true')
parser.add_argument('--swag_c_epochs', default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--swag_save_stats', action='store_true', help='Save the SWA statistics')
parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument("--verbose", action='store_true', help='verbose training mode')

# Xingchen Addition
parser.add_argument("--save_freq_weight_norm", type=int, default=1, metavar='N', help='save frequency of weight norm')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

# convert types
try:
    args.swag_c_epochs = int(args.swag_c_epochs)
except ValueError:
    args.swag_c_epochs = float(args.swag_c_epochs)

args.dir = args.dir  + args.dataset + '/' + args.model + '/SGD'
thing = ""
if args.no_schedule:
    thing += "_flat"
if args.step_schedule:
    thing += "_step"
if args.swag:
    args.dir += "SWA"
    thing = "_swalr="+str(args.swag_lr)+'_swastart='+str(args.swag_start)
if args.new_schedule:
    thing += "_new"
args.dir += '/seed='+str(args.seed)+'_lr=' + str(args.lr_init) + thing + '_mom=' + str(args.momentum) + '_wd=' + str(
    args.wd) + '_batchsize='+str(args.batch_size) + '_numepochs=' +str(args.epochs) + '/'


if args.no_schedule and args.step_schedule:
    raise ValueError("Both no_schedule and step_schedule are turned on. Quitting due to ambiguity.")

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
if args.model == 'resnet50':
    model_cfg = resnet50
else:
    model_cfg = getattr(models, args.model)

# loaders, num_classes = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     model_cfg.transform_train,
#     model_cfg.transform_test,
#     use_validation=not args.use_test,
# )
if args.dataset == 'ImageFolder':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )



    loaders, num_classes = data.loaders(
        # train_subset=None,
        # train_subset_seed=None,
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_train=transform_train,
        transform_test=transform_test,
        use_validation=not args.use_test,
    )
elif args.dataset == "Tensor":
    loaders = {}
    for subset in ["train", "test"]:
        dataset = os.path.join(args.data_path, subset)
        x = torch.tensor(np.load(os.path.join(dataset, "features.npy")))
        y = torch.tensor(np.load(os.path.join(dataset, "labels.npy")))
        dset = TensorDataset(x, y)
        loaders[subset] = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=args.num_workers)
    num_classes = len(torch.unique(y))
    data_dim = x.shape[1]

else:
    loaders, num_classes = data.loaders(
        # train_subset=None,
        # train_subset_seed=None,
        dataset=args.dataset,
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        use_validation=not args.use_test,
    )

print('Preparing model')
if args.dataset == 'ImageFolder' or args.model == 'resnet18':
    #model = model_cfg(pretrained=False, num_classes=num_classes)
    model = model_cfg
    model.to(args.device)
elif args.dataset == "Tensor":
    print(*model_cfg.args, dict(**model_cfg.kwargs))
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, input_dim=data_dim, **model_cfg.kwargs)
    #model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
   model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

if torch.cuda.device_count() > 1:
    print("Multi-GPU: Using " + str(torch.cuda.device_count()) + " GPUs for training.")
    import torch.nn as nn

    model = nn.DataParallel(model)
model.to(args.device)

if args.dataset == 'ImageFolder':
    if args.swag:
        if not args.no_covariance:
            print('SWA-Gaussian training')
            swag_model = SWAG(model,
                              subspace_type=args.swag_subspace, subspace_kwargs={'max_rank': args.swag_rank},
                               num_classes=num_classes)
            swag_model.to(args.device)
        else:
            print('No Covariance Estimation')
            swag_model = SWA(model, num_classes=num_classes)
            swag_model.to(args.device)
    else:
        print('SGD training')
else:
    if args.swag:
        if not args.no_covariance:
            print('SWA-Gaussian training')
            swag_model = SWAG(model_cfg.base,
                              subspace_type=args.swag_subspace, subspace_kwargs={'max_rank': args.swag_rank},
                              *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            swag_model.to(args.device)
        else:
            print('No Covariance Estimation')
            swag_model = SWA(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            swag_model.to(args.device)
    else:
        print('SGD training')


if args.no_schedule:
    print("Constant learning rate schedule")
elif args.step_schedule:
    print("Piecewise constant learning rate schedule")
elif args.cosine_annealing:
    print("Cosing annealing learning rate schedule.")
else:
    print("Linearly decaying learning rate schedule")


def schedule(epoch, total_epoch):
    t = epoch / (args.swag_start if (args.swag and not args.schedule_swag_lr) else total_epoch)
    lr_ratio = args.swag_lr / args.lr_init if (args.swag and not args.schedule_swag_lr) else args.lr_r
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


def new_schedule(epoch, total_epoch):
    t = epoch / (args.swag_start if (args.swag and not args.schedule_swag_lr) else total_epoch)
    lr_ratio = args.swag_lr / args.lr_init if (args.swag and not args.schedule_swag_lr) else args.lr_r
    if t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * t / 0.9
    else:
        factor = lr_ratio
    return args.lr_init * factor


def schedule_piecewise_const(epoch):
    """
    Use a piecewise constant learning rate
    d: The proportion of new learning rate to the initial learning rate. 0.5 means halving the learning rate
    r: frequency of reducing the learning rate. e.g. 40: reducing learning rate every 40 epochs
    """
    if args.swag:
        if epoch in args.when and epoch < args.swag_start:
            args.lr_init /= 10
        elif epoch >= args.swag_start:
            return args.swag_lr
    elif epoch in args.when:
        args.lr_init /= 10
    return args.lr_init


def schedule_cos_annealing(epoch, total_epoch):
    """
     Cosine annealing learning rate scheduler
     Adapts from https://github.com/hysts/pytorch_shake_shake/blob/master/train.py

     """

    def _cosine_annealing(epoch, total_epoch, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(epoch / total_epoch * np.pi))

    if args.swag and epoch > args.swag_start and (not args.schedule_swag_lr):
        return args.lr_init * _cosine_annealing(
            args.swag_start,
            total_epoch,
            1.,
            0.,
        )
    return args.lr_init * _cosine_annealing(
        epoch,
        total_epoch,
        1.,
        0.,
    )


def schedule_variant(epoch):
    """A variant of the linear learning rate schedule"""

    def _linear_annealing(epoch):
        t = epoch / args.epochs
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr_init * factor

    lr_tmp = _linear_annealing(epoch)
    if args.swag:
        if (epoch > args.swag_start) or lr_tmp <= args.swag_lr:
            return _linear_annealing(args.swag_start)
    return lr_tmp


if args.normalized_wd:
    weight_decay = args.wd * np.sqrt(args.batch_size / (args.epochs * len(loaders['train'])))
else:
    weight_decay = args.wd

criterion = losses.cross_entropy
if args.dataset == "Bike":
    criterion = losses.squared_error

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=weight_decay
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # test_res = utils.eval(loaders['test'], model, criterion)

if args.swag and args.swag_resume is not None:
    checkpoint = torch.load(args.swag_resume)
    swag_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_tr_loss', 'swa_tr_acc', 'swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[
                                                                                                             -2:]
    swag_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    # END ADD NEW CODE DIEGO
    if not args.no_schedule:
        e = epoch - start_epoch if args.reset_resume else epoch
        total_e = args.epochs - start_epoch if args.reset_resume else args.epochs
        if args.step_schedule:
            lr = schedule_piecewise_const(e)
        elif args.linear_annealing:
            lr = schedule_variant(e)
        elif args.cosine_annealing:
            lr = schedule_cos_annealing(e, total_e)
        elif args.new_schedule:
            lr = new_schedule(e, total_e)
        else:
            lr = schedule(e, total_e)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    if args.swag and args.swag_c_epochs < 1 and epoch >= args.swag_start:
        # If mode collection is more frequent than once per epoch
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose,
                                      swag_model=swag_model, swag_batch_c=int(len(loaders['train']) * args.swag_c_epochs))
    else:
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose)

    # update batch norm parameters before testing
    utils.bn_update(loaders['train'], model)
    test_res = utils.eval(loaders['test'], model, criterion)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}
        if args.dataset == "Bike":
            test_res = {"loss": None}


    if args.swag and (epoch + 1) > args.swag_start:
        # If the frequency of collecting swag models is less than once per epoch - otherwise the models have been
        # collected already in the train_epoch call.
        if args.swag_c_epochs >= 1 and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
            swag_model.collect_model(model)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            train_res_swag = utils.eval(loaders['train'], swag_model, criterion, optimizer)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)

        else:
            swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}
            train_res_swag = {'loss': None, 'accuracy': None}

    else:
        train_res_swag = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag and (epoch + 1) > args.swag_start:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                epoch=epoch + 1,
                state_dict=swag_model.state_dict(),
            )

    #Xingchen code addition - enable to save the L2 and Linf weight norms
    if (epoch + 1) % args.save_freq_weight_norm == 0:
        utils.save_weight_norm(
            args.dir,
            epoch + 1,
            name='weight_norm',
            model=model
        )
        if args.swag and (epoch + 1) > args.swag_start:
            utils.save_weight_norm(
                args.dir,
                epoch + 1,
                name='swa_weight_norm',
                model=swag_model
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              test_res['top5_accuracy'], time_ep, memory_usage]

    epoch = epoch - start_epoch if args.reset_resume else epoch

    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        time_ep=time_ep,
        memory_usage=memory_usage,
        train_accuracy=train_res['accuracy'],
        train_top5_accuracy=train_res['top5_accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
        test_top5_accuracy=test_res['top5_accuracy']
    )

    if args.swag:
        values = values[:-2] + [train_res_swag['loss'], train_res_swag['accuracy'],
                                swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
        np.savez(
            args.dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            time_ep=time_ep,
            memory_usage=memory_usage,
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy=train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy=test_res['top5_accuracy'],
            swag_loss=swag_res['loss'],
            swag_train_loss=train_res_swag['loss'],
            swag_train_acc=train_res_swag['accuracy'],
            swag_accuracy=swag_res['accuracy'],
            swag_top5_accuracy=swag_res['top5_accuracy']
        )

    # if args.swag:
    #     values = values[:-2] + [swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if train_res['loss'] != train_res['loss']:
       break

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        epoch=args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag and args.swag_save_stats:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            epoch=args.epochs,
            state_dict=swag_model.state_dict(),
        )
