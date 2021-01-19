# Logistic regression toy example
# Added by Xingchen Wan on 26 Dec 2019

import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['Logistic', 'LogisticCIFAR']

#for MNIST change 28x28
#for CIFAR100 32x32x3
class _LogisticRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super(_LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(self.input_dim, num_classes, bias=True)

    def forward(self, x):
        return self.layer(x.view(-1, self.input_dim))


class Logistic:
    base = _LogisticRegression
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(( 0.5), ( 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(( 0.5), (0.5))])

class _LogisticRegressionCIF(nn.Module):
    def __init__(self, num_classes=100, input_dim=32*32):
        super(_LogisticRegressionCIF, self).__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(self.input_dim, num_classes, bias=True)

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = x.view(-1, self.input_dim)
        return self.layer(x)


class LogisticCIFAR:
    base = _LogisticRegressionCIF
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.4376821 , 0.4437697 , 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.45242316, 0.45249584, 0.46897713), (0.21943445, 0.22656967, 0.22850613))
    ])
