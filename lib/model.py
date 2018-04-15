import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import tqdm

class STN(nn.Module):
    def __init__(self, padding_mode='border',
                 init_mode='id'):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.padding_mode = padding_mode
        self.init_mode = init_mode

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7), # 1x28x28
            nn.MaxPool2d(2, stride=2), # 8x22x22
            nn.ReLU(True), # 8x11x11
            nn.Conv2d(8, 10, kernel_size=5), # 10x7x7
            nn.MaxPool2d(2, stride=2), # 10x3x3
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        if self.init_mode == 'id':
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.fill_(0)
            self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size()) 
        x = F.grid_sample(x, grid, padding_mode=self.padding_mode)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def raw_score(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1x28x28->10x24x24
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5) # 10x12x12->25x8x8
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(16*25, 50) # 25x4x4
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 25*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class FCN(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Dropout(), 
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

############ second replication experiment ####################
class STN2(nn.Module):
    def __init__(self, padding_mode='border',
                 init_mode="id"):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320*2, 50)
        self.fc2 = nn.Linear(50, 19)
        self.padding_mode = padding_mode
        self.init_mode = init_mode

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7), # 1x28x28
            nn.MaxPool2d(2, stride=2), # 8x22x22
            nn.ReLU(True), # 8x11x11
            nn.Conv2d(8, 10, kernel_size=5), # 10x7x7
            nn.MaxPool2d(2, stride=2), # 10x3x3
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        if init_mode == 'id':
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.fill_(0)
            self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size()) 
        x = F.grid_sample(x, grid, padding_mode=self.padding_mode)

        return x

    def forward(self, x):
        # x has two input, transform independently
        x1, x2 = x
        
        # transform the input
        x1 = self.stn(x1)
        x2 = self.stn(x2)        

        # Perform the usual forward pass
        x1 = F.relu(F.max_pool2d(self.conv1(x1), 2))
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x2)), 2))

        x1 = x1.view(-1, 320)
        x2 = x2.view(-1, 320)
        # concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN2(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, kernel_size=5) # 2x28x28->10x24x24
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5) # 10x12x12->25x8x8
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(16*25, 50) # 25x4x4
        self.fc2 = nn.Linear(50, 19)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 25*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
########################## third experiments ######################
class STN3(nn.Module):
    def __init__(self, padding_mode='border',
                 init_mode='lr'):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320 * 2, 50)
        self.fc2 = nn.Linear(50, 19)
        self.padding_mode = padding_mode
        self.init_mode = init_mode

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7), # 1x28x56
            nn.MaxPool2d(2, stride=2), # 8x22x50
            nn.ReLU(True), # 8x11x25
            nn.Conv2d(8, 10, kernel_size=5), # 10x7x21
            nn.MaxPool2d(2, stride=2), # 10x3x10
            nn.ReLU(True)
        )

        # Regressor for the two 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 10, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2 * 2)
        )

        if self.init_mode == 'id':
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.fill_(0)
            self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0,
                                                          1, 0, 0, 0, 1, 0])
        elif self. init_mode == 'lr':
            self.fc_loc[2].weight.data.fill_(0)
            self.fc_loc[2].bias.data = torch.FloatTensor([0.5, 0, -0.5, 0, 1, 0,
                                                          0.5, 0,  0.5, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 10)
        theta = self.fc_loc(xs)

        theta1 = theta[:,:6].contiguous().view(-1,2,3)
        theta2 = theta[:,6:].contiguous().view(-1,2,3)

        n,c,h,w = x.size()

        grid1 = F.affine_grid(theta1, torch.Size([n,c,h,math.floor(w/2)]))
        x1 = F.grid_sample(x, grid1, padding_mode=self.padding_mode)

        grid2 = F.affine_grid(theta2, torch.Size([n,c,h,math.floor(w/2)]))
        x2 = F.grid_sample(x, grid2, padding_mode=self.padding_mode)
        
        return [x1, x2]

    def forward(self, x):
        # x has two input, transform independently
        x1, x2 = x
        x = torch.cat([x1,x2], dim=3)
        
        # transform the input
        x1, x2 = self.stn(x)

        # Perform the usual forward pass
        x1 = F.relu(F.max_pool2d(self.conv1(x1), 2))
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x2)), 2))

        x1 = x1.view(-1, 320)
        x2 = x2.view(-1, 320)
        # concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNN3(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1x28x56->10x24x52
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5) # 10x12x26->25x8x22
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(25*4*11, 50) # 25x4x11
        self.fc2 = nn.Linear(50, 19)

    def forward(self, x):
        x = torch.cat(x, dim=3) # longer width
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 25*4*11)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
