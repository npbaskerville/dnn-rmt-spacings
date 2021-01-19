"""
    leakyVGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/leakyVGG.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['leakyVGG6','leakyVGG16basic','leakyVGG11', 'leakyVGG11BN','leakyVGG16', 'leakyVGG16BN', 'leakyVGG19', 'leakyVGG19BN',]


def make_layers(cfg, batch_norm=True):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    #6: [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    6: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class leakyVGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(leakyVGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = leakyVGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
    ])


class Basic:
    base = leakyVGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class leakyVGG16basic(Basic):
    pass

class leakyVGG6(Base):
    kwargs = {'depth': 6}
    pass

class leakyVGG11(Base):
    kwargs = {'depth': 11}
    pass

class leakyVGG16(Base):
    pass


class leakyVGG16BN(Base):
    kwargs = {'batch_norm': True}


class leakyVGG19(Base):
    kwargs = {'depth': 19}


class leakyVGG19BN(Base):
    kwargs = {'depth': 19, 'batch_norm': True}

class leakyVGG11(Base):
    pass


class leakyVGG11BN(Base):
    kwargs = {'batch_norm': True}


# The leakyVGG-16 model for Backpack - added by Xingchen Wan 30 Nov 2019


def make_layers_backpack(cfg, batch_norm=True):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            in_channels = v
    return layers


def get_backpacked_leakyVGG(model: leakyVGG, depth=6, batch_norm=False, num_classes=10, cuda=True):

    import backpack, numpy as np

    features_layer_list = make_layers_backpack(cfg[depth], batch_norm)
    flatten_layer = [backpack.core.layers.Flatten()]
    classifier_list = [
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, num_classes)
    ]
    backpacked_model_layers = features_layer_list + flatten_layer + classifier_list
    # Initialise the Backpack-ready model
    backpacked_model = nn.Sequential(*backpacked_model_layers)

    def _copy_block_content(model1, model2, offset=0):
        """Copy the weight and bias model1 -> model2, layer wise. Only model with identical names are reported"""
        m2_state_dict = model2.state_dict()
        for k, v in model1.state_dict().items():
            n_layer = int(k.split(".")[0]) + offset
            model2_key = str(n_layer)+"."+k.split(".")[1]
            assert model2_key in m2_state_dict.keys(), model2_key + "is not in m2_state_key!. m2_state_key is " + str(m2_state_dict.keys())
            m2_state_dict[model2_key].copy_(v)
        return model2, offset

    backpacked_model, offset = _copy_block_content(model.features, backpacked_model)
    #offset = np.max(np.array([int(n.split(".")[0]) for n in model.features.state_dict().keys()])) + 1
    if depth == 6:
        offset = 0 # Apologies for the magic number, but this is just an expediency for now. Xingchen
    else:
        raise NotImplementedError
    backpacked_model, _ = _copy_block_content(model.classifier, backpacked_model, offset)
    backpacked_model.to('cuda' if cuda else 'cpu')
    return backpacked_model
