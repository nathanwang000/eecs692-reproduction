import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import tqdm
from lib.model import STN, CNN, FCN, STN2, STN3, CNN2, CNN3
from lib.train import Trainer
from lib.utility import convert_image_np, wrap, RandomRTS, to_cuda
from lib.dataset import DoubleMNIST
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="STN")
    parser.add_argument('-m', choices=['stn', 'cnn', 'fcn'],
                        help='model [stn|cnn|fcn]', default='stn')
    parser.add_argument('-r', help="degree of rotation", type=float, default=0)
    parser.add_argument('-t', help="translation", type=float, default=0)
    parser.add_argument('-s', help="scaling diff from 1", type=float, default=0)
    parser.add_argument('-l', help="save location", default="models/")
    parser.add_argument('-i', help="STN initialization scheme",
                        choices=['id', 'lr', 'rand'], default="id")    
    parser.add_argument('-p', help="padding mode",
                        choices=['zeros', 'border'], default="zeros")
    parser.add_argument('-e', help=''' 
    mnist: default mnist experiment
    double: 2 digits addition using 2 channels
    one: 2 digits addition using 1 image
    ''',
                        choices=['mnist', 'double', 'one'], default="mnist")    
    return parser
    
if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    # get dataset
    if args.e == 'mnist':
        train_data = datasets.MNIST(root='.', train=True, download=True,
                                    transform=transforms.Compose([
                                        RandomRTS(degree=(-args.r, args.r),
                                                  translate=(-args.t, args.t),
                                                  scale=(1-args.s,1+args.s)),
                                        wrap(transforms.ToTensor()),
                                        wrap(transforms.Normalize((0.1307,), (0.3081,)))
                                    ]))
        test_data = datasets.MNIST(root='.', train=False,
                                   transform=transforms.Compose([
                                       RandomRTS(degree=(0,0),
                                                 translate=(0,0),
                                                 scale=(1,1)),
                                       wrap(transforms.ToTensor()),
                                       wrap(transforms.Normalize((0.1307,), (0.3081,)))
                                   ]))
    elif args.e == 'double' or args.e == 'one':
        train_data = DoubleMNIST(train=True, transform=transforms.Compose([
            RandomRTS(degree=(-args.r, args.r),
                      translate=(-args.t, args.t),
                      scale=(1-args.s,1+args.s)),
            wrap(transforms.ToTensor()),
            wrap(transforms.Normalize((0.1307,), (0.3081,)))
        ]))
        test_data = DoubleMNIST(train=False, transform=transforms.Compose([
            RandomRTS(degree=(0,0),
                      translate=(0,0),
                      scale=(1,1)),
            wrap(transforms.ToTensor()),
            wrap(transforms.Normalize((0.1307,), (0.3081,)))
        ]))
        
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                              shuffle=True, num_workers=4)

    # get model
    if args.m == 'stn':
        if args.e == 'mnist':
            model = STN(padding_mode=args.p, init_mode=args.i)
        elif args.e == 'double':
            model = STN2(padding_mode=args.p, init_mode=args.i)
        elif args.e == 'one':
            model = STN3(padding_mode=args.p, init_mode=args.i)
    elif args.m == 'cnn':
        if args.e == 'mnist':
            model = CNN()
        elif args.e == 'double':
            model = CNN2()
        elif args.e == 'one':
            model = CNN3()
    elif args.m == 'fcn':
        model = FCN()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = to_cuda(model)

    # training
    optimizer = optim.Adam(model.parameters())
    t = Trainer(model, optimizer, use_cuda=use_cuda)
    
    for epoch in range(1, 10+1):
        t.train(train_loader)
        t.test(test_loader)

    # save the model
    loc = os.path.join(args.l, args.e)
    os.system('mkdir -p %s' % loc)
    loc = os.path.join(loc,
                       "%s_R%d_T%d_S%d.cpt" % (args.m, args.r, args.t*100, args.s*100))
    torch.save(model, loc)
    
    
