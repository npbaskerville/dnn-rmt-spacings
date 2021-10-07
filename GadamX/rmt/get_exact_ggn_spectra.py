"""
Obtain the GGN spectrum over minibatches of some batch size.
"""
import argparse

import sys

sys.path.append('..')
import h5py
import math
import numpy as np
import torch
from curvature import data, models, losses
from tqdm import tqdm
from torch.utils.data import TensorDataset
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--model", type=str, default="Logistic")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("-test", action="store_true", help="If to eval on test set.")
parser.add_argument("--out", type=str, default="output/spectrum.hdf5")
parser.add_argument("-regression", action="store_true")
parser.add_argument("-fp16", action="store_true")


args = parser.parse_args()

dtype = torch.float16 if args.fp16 else torch.float32

network = args.model
epochsave = int(args.checkpoint.split('checkpoint-')[1].split('.')[0])

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

if args.dataset == "Tensor":
    full_datasets = {}
    for subset in ["train", "test"]:
        dataset = os.path.join(args.data_path, subset)
        x = torch.tensor(np.load(os.path.join(dataset, "features.npy")))
        y = torch.tensor(np.load(os.path.join(dataset, "labels.npy")))
        dset = TensorDataset(x, y)
        full_datasets[subset] = dset
    num_classes = len(torch.unique(y))

else:
    datasets, num_classes = data.datasets(
        args.dataset,
        args.data_path,
        transform_train=model_cfg.transform_test,
        transform_test=model_cfg.transform_test,
        use_validation=False,
        train_subset=None,
        train_subset_seed=None,
    )


    full_datasets, _ = data.datasets(
        args.dataset,
        args.data_path,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        use_validation=False,
    )


if args.test:
    dset = full_datasets['test']
else:
    dset = full_datasets["train"]
n_data = len(dset)

batch_size = args.bs if args.bs > 0 else n_data
full_loader = torch.utils.data.DataLoader(
    dset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, input_dim=np.prod(dset[0][0].shape), **model_cfg.kwargs)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
if batch_size > 128:
    device = torch.device('cpu')

model.to(device=device, dtype=dtype)
num_parametrs = sum([p.numel() for p in model.parameters()])
criterion = losses.squared_error if args.regression else losses.cross_entropy


outfile = h5py.File(args.out, "w")
ggn_evals = outfile.create_dataset("ggn_evals", (int(math.ceil(n_data / batch_size)), num_parametrs), dtype=float)


model.zero_grad()
for batch_ind, (input, target) in tqdm(enumerate(full_loader)):
    hessian = torch.zeros(num_parametrs, num_parametrs, dtype=dtype).cpu()
    model.zero_grad()
    input = input.to(device=device, dtype=dtype)
    output = model(input)
    print(output.shape)
    grads = [torch.autograd.grad(out, model.parameters(), create_graph=True) for out in output]
    ggn = sum([grad.T @ grad for grad in grads])/len(grads)
    print(ggn.shape)

    ggn_evals[batch_ind] = np.linalg.eigvalsh(ggn.detach().cpu().numpy())
    del ggn, grads