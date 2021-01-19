# rom __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
#
if sys.version_info[0] == 2:
     import cPickle as pickle
else:
     import pickle
#
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity# download_and_extract_archive


class IMAGENET32(VisionDataset):
    """`IMAGENET32 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.


        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'Imagenet32_train_npz'
    url = "http://www.image-net.org/image/downsample/Imagenet32_train_npz.zip"
    filename = "Imagenet32_train_npz.zip"
    tgz_md5 = 'b0d308fb0016e41348a90f0ae772ee38'
    train_list = [
        ['train_data_batch_1.npz', '464fde20de6eb44c28cc1a8c11544bb1'],
        ['train_data_batch_2.npz', 'bdb56e71882c3fd91619d789d5dd7c79'],
        ['train_data_batch_3.npz', '83ff36d76ea26867491a281ea6e1d03b'],
        ['train_data_batch_4.npz', '98ff184fe109d5c2a0f6da63843880c7'],
        ['train_data_batch_5.npz', '462b8803e13c3e6de9498da7aaaae57c8'],
        ['train_data_batch_6.npz', 'e0b06665f890b029f1d8d0a0db26e119'],
        ['train_data_batch_7.npz', '9731f469aac1622477813c132c5a847a'],
        ['train_data_batch_8.npz', '60aed934b9d26b7ee83a1a83bdcfbe0f'],
        ['train_data_batch_9.npz', 'b96328e6affd718660c2561a6fe8c14c'],
        ['train_data_batch_10.npz', '1dc618d544c554220dd118f72975470c'],
    ]

    test_list = [
        ['val_data', 'a8c04a389f2649841fb7a01720da9dd9'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(IMAGENET32, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = np.load(f)
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def _load_meta(self):
        pass
        #path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        #if not check_integrity(path, self.meta['md5']):
        #    raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                      ' You can use download=True to download it')
        #with open(path, 'rb') as infile:
        #    if sys.version_info[0] == 2:
        #        data = pickle.load(infile)
        #    else:
        #        data = pickle.load(infile, encoding='latin1')
        #    self.classes = data[self.meta['key']]
        #self.classes = int(np.max(self.targets)) + 1
        #self.class_to_idx = {i: i for i in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

   # def download(self):
    #    if self._check_integrity():
    #        print('Files already downloaded and verified')
    #        return
    #    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



