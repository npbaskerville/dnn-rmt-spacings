# Use this file for versions of torchvision newer than 0.2.1
# Xingchen Wan addition 5 Nov

import numpy as np
import torch
import torchvision
import os

from curvature.imagenet32 import IMAGENET32
from curvature.bike import BikeDataset

#assert torchvision.__version__ >= "0.4.0", "Expected torchvision version 0.4.0 but got " + torchvision.__version__


def datasets(
        dataset,
        path,
        transform_train,
        transform_test,
        use_validation=True,
        val_size=0.1,
        train_subset=None,
        train_subset_seed=None):
    assert dataset in {'CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet32', 'ImageNet', "Bike"}
    print('Loading %s from %s' % (dataset, path))

    path = os.path.join(path, dataset.lower())

    if dataset == 'ImageNet32':
        ds = IMAGENET32
    elif dataset == "Bike":
        ds = BikeDataset
    else:
        ds = getattr(torchvision.datasets, dataset)

    train_set = ds(root=path, train=True, download=True, transform=transform_train)
    n_train_samples = len(train_set)
    if isinstance(val_size, float):
        assert val_size < 1, "If entered as a float number to represent the fraction " \
                             "of validation data, this number must be smaller than 1."
        val_size = int(n_train_samples * val_size)
    elif isinstance(val_size, int):
        pass
    else:
        raise TypeError("val_size needs to be either an int or a float, but got "+type(val_size))
    print(train_set.targets)
    num_classes = torch.max(torch.as_tensor(train_set.targets)).item() + 1
    if use_validation:
        print('Using %d samples for validation [deterministic split]' % (val_size))
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.data[:-val_size]
        train_set.train_data = train_set.data
        train_set.train_labels = train_set.labels

        test_set = ds(root=path, train=True, download=False, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.data[-val_size:]
        test_set.test_data = test_set.data
        test_set.test_labels = test_set.labels
        #delattr(test_set, 'data')
        #delattr(test_set, 'targets')
    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(root=path, train=False, download=False, transform=transform_test)

    if train_subset is not None:
        order = np.arange(train_set.data.shape[0])
        if train_subset_seed is not None:
            rng = np.random.RandomState(train_subset_seed)
            rng.shuffle(order)
        train_set.data = train_set.data[order[:train_subset]]
        train_set.targets = np.array(train_set.targets)[order[:train_subset]].tolist()

    #print('Using train (%d) + test (%d)' % (train_set.train_data.shape[0], test_set.test_data.shape[0]))

    return \
        {
            'train': train_set,
            'test': test_set
        }, \
        num_classes


def loaders(
        dataset,
        path,
        batch_size,
        num_workers,
        transform_train,
        transform_test,
        use_validation=True,
        val_size=0.1,
        shuffle_train=True):

    ds_dict, num_classes = datasets(
        dataset, path, transform_train, transform_test, use_validation=use_validation, val_size=val_size)

    return \
        {
            'train': torch.utils.data.DataLoader(
                ds_dict['train'],
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                ds_dict['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes


class CIFAR10AUG(torch.utils.data.Dataset):
    base_class = torchvision.datasets.CIFAR10

    def __init__(self, root, train=True, transform=None, download=False, shuffle_seed=1):
        self.base = self.base_class(root, train=train, transform=None, target_transform=None, download=download)
        self.transform = transform

        self.pad = 4
        self.size = len(self.base) * (2 * self.pad + 1) * (2 * self.pad + 1) * 2
        rng = np.random.RandomState(shuffle_seed)
        self.order = rng.permutation(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = self.order[index]

        base_index = index // ((2 * self.pad + 1) * (2 * self.pad + 1) * 2)
        img, target = self.base[base_index]

        transform_index = index % ((2 * self.pad + 1) * (2 * self.pad + 1) * 2)
        flip_index = transform_index // ((2 * self.pad + 1) * (2 * self.pad + 1))
        crop_index = transform_index % ((2 * self.pad + 1) * (2 * self.pad + 1))
        crop_x = crop_index // (2 * self.pad + 1)
        crop_y = crop_index % (2 * self.pad + 1)

        if flip_index:
            img = torchvision.transforms.functional.hflip(img)
        img = torchvision.transforms.functional.pad(img, self.pad)
        img = torchvision.transforms.functional.crop(img, crop_x, crop_y, 32, 32)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR100AUG(CIFAR10AUG):
    base_class = torchvision.datasets.CIFAR100
