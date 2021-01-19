"""
Embed CIFAR10 images with pre-trained resnet34 and save the resulting tensors to disk.
"""
import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1024, help="Batch size to compute embeddings in")
    parser.add_argument("--outdir", type=str, required=True, help="Path to directory in which embeddings and labels will be saved.")
    parser.add_argument("--datadir", type=str, required=True, help="Path to directory containing CIFAR10.")
    args = parser.parse_args()

    resnet = models.resnet34(pretrained=True)
    modules=list(resnet.children())[:-1]
    resnet=torch.nn.Sequential(*modules)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    for train in [True, False]:
        dset = datasets.CIFAR10(args.datadir, download=True, transform=transform, train=train)

        loader = torch.utils.data.DataLoader(dset, batch_size=args.bs, num_workers=6)

        resnet.eval()
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"

        resnet = resnet.to(device)
        embeds = []
        labels = []
        for iter_ind, (img, lbl) in enumerate(loader):
            print("{} of {}".format(iter_ind, len(loader)))
            embeds.append(resnet(img.to(device)).cpu().detach().numpy())
            labels.append(lbl.cpu().detach().numpy())

        outdir = os.path.join(args.outdir, "train" if train else "test")
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "features"), np.concatenate(embeds))
        np.save(os.path.join(outdir, "labels"), np.concatenate(labels))


if __name__ == "__main__":
    main()
