import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from lib.utility import convert_image_np, to_cuda, to_var

class Trainer(object):

    def __init__(self, model, optimizer=None, use_cuda=True, epoch=1, twoImage=False):
        self.model = model
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters())

        self.use_cuda = use_cuda
        self.epoch = epoch
        self.twoImage = twoImage

    def train_step2(self, data, target):
        x1, x2 = data
        y = target
        
        yhat1 = self.model(x1)
        l1 = F.nll_loss(yhat1, y)

        yhat2 = self.model(x2)
        l2 = F.nll_loss(yhat2, y)

        x1_ = self.model.stn(x1)
        x2_ = self.model.stn(x2)
        l_interact = ((x2_ - x1_)**2).mean() 
        
        loss = l1 + l2 + l_interact
        print(l1, l2, l_interact)
        loss.backward()

        return loss.item()
        
    def train_step(self, data, target):
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return loss.item()

    def train(self, train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            if type(data) is dict:
                data = data['image']
            if self.use_cuda:
                data, target = to_cuda(data), to_cuda(target)

            data, target = to_var(data), to_var(target)                
            self.optimizer.zero_grad()
            if self.twoImage:
                loss = self.train_step2(data, target)
            else:
                loss = self.train_step(data, target)                
            self.optimizer.step()
            
            if batch_idx % 500 == 0:
                if type(data) is list:
                    len_data = len(data[0])
                else:
                    len_data = len(data)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len_data, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
        self.epoch += 1

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if type(data) is dict:            
                data = data['image']
            if self.use_cuda:
                data, target = to_cuda(data), to_cuda(target)
            data, target = to_var(data, volatile=True), to_var(target)
            output = self.model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
        return 100. * correct / len(test_loader.dataset)
