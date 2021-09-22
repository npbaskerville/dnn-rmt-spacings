import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

__all__ = ['MLP_CIF', 'MLP', 'MLP_med', 'MLP_big', 'MLP_sdp', 'MLP_deep']

class NN_CIF(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(32*32*3, 10, bias=True)
        self.lin2 = nn.Linear(10, 300, bias=True)
        self.lin3 = nn.Linear(300, 100, bias=True)

    def forward(self, xb):
        x = xb.view(-1,32*32*3)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class Mnist_NN(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(784, 10, bias=True)
        self.lin2 = nn.Linear(10, 100, bias=True)
        self.lin3 = nn.Linear(100, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class Mnist_MLP_deep(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(784, 10, bias=True)
        self.lin2 = nn.Linear(10, 100, bias=True)
        self.lin3 = nn.Linear(100, 100, bias=True)
        self.lin4 = nn.Linear(100, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)

class Mnist_NN_sdp(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(784, 5, bias=True)
        self.lin2 = nn.Linear(5, 200, bias=True)
        self.lin3 = nn.Linear(200, 50, bias=True)
        self.lin4 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)

class Mnist_med(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(784, 15, bias=True)
        self.lin2 = nn.Linear(15, 150, bias=True)
        self.lin3 = nn.Linear(150, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class Mnist_NN_big(nn.Module):
    def __init__(self, num_classes=10, input_dim=28*28):
        super().__init__()
        self.lin1 = nn.Linear(784, 512, bias=True)
        self.lin2 = nn.Linear(512, 256, bias=True)
        self.lin3 = nn.Linear(256, 10, bias=True)

    def forward(self, xb):
        x = xb.view(-1,784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

#for MNIST change 28x28
#for CIFAR100 32x32x3
# class _LogisticRegression(nn.Module):
#     def __init__(self, num_classes=10, input_dim=28*28):
#         super(_LogisticRegression, self).__init__()
#         self.input_dim = input_dim
#         self.layer = nn.Linear(self.input_dim, num_classes, bias=True)
#
#     def forward(self, x):
#         return self.layer(x.view(-1, self.input_dim))


class MLP:
    base = Mnist_NN
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, ), (0.5, ))])

class MLP_deep:
    base = Mnist_MLP_deep
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, ), (0.5, ))])
class MLP_sdp:
    base = Mnist_NN_sdp
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class MLP_med:
    base = Mnist_med
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class MLP_big:
    base = Mnist_NN_big
    args = list()
    kwargs = dict()
    # Default transform
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MLP_CIF:
    base = NN_CIF
    args = list()
    kwargs = dict()
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
