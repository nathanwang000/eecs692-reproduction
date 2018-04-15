import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DoubleMNIST(Dataset):

    def __init__(self, train=True, transform=None):
        np.random.seed(42)
        self.mnist = datasets.MNIST(root='.', train=train, download=True)
        self.transform = transform
        # shuffle the order
        self.num_orders = 2
        self.order = [np.random.permutation(len(self.mnist)) \
                       for _ in range(self.num_orders)]
        self.order = np.hstack(self.order)

    def __len__(self):
        return len(self.order) - 1

    def __getitem__(self, idx):
        
        im1, y1 = self.mnist[self.order[idx]]
        im2, y2 = self.mnist[self.order[idx+1]]

        if self.transform:
            im1 = self.transform(im1)['image']
            im2 = self.transform(im2)['image']
        y = y1 + y2
        return ((im1, im2), y)
