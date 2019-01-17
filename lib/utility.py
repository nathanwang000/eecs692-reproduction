import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import math
import tqdm
from PIL import Image

def to_cuda(x):
    if type(x) is list:
        x = [x_.cuda() for x_ in x]
    else:
        x = x.cuda()
    return x

def to_var(x, *args, **kwargs):
    if type(x) is list:
        x = [Variable(x_, *args, **kwargs) for x_ in x]
    else:
        x = Variable(x, *args, **kwargs)
    return x

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(model, visual_loader, use_cuda=True, nrow=4):
    import matplotlib.pyplot as plt    
    # Get a batch of training data
    data, _ = next(iter(visual_loader))
    if type(data) is dict:
        data = data['image']
    data = to_var(data, volatile=True)

    if use_cuda:
        data = to_cuda(data)
    
    input_tensor = data.cpu().data
    transformed_input_tensor = model.stn(data).cpu().data

    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor, nrow=nrow))

    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor, nrow=nrow))

    # Plot the results side-by-side
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(in_grid)
    axarr[0].axis('off')
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(out_grid)
    axarr[1].axis('off')    
    axarr[1].set_title('Transformed Images')

    
def RTS(image, angle, center=None, translate=None, scale=None):
    ''' translate is in terms of fractional difference'''
    w, h = image.size
    if center is None:
        center = w/2, h/2 #return image.rotate(angle)
    
    if translate is not None:
        translate[0] *= w
        translate[1] *= h
    
    angle = -angle/180.0*math.pi
    nx,ny = x,y = center
    sx=sy=1.0
    if translate:
        (nx,ny) = (center[0] + translate[0], center[1] + translate[1])
    if scale:
        (sx,sy) = scale
        
    cosine = math.cos(angle)
    sine = math.sin(angle)
    
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    params = np.array([a,b,c,d,e,f])
    return image.transform(image.size, Image.AFFINE, params, resample=Image.BILINEAR), params

def randRange(M, m):
    r = M - m
    return np.random.rand() * r + m

class RandomRTS(object):
    """Random rotate samples in range(-degree, degree).
    
    Args:
    degree
    """
    def __init__(self, degree=(-30, 30), translate=(0, 0), scale=(1,1)):
        self.degree = degree
        self.center = None # do not change rotation center
        self.translate = translate
        self.scale = scale

    def __call__(self, sample):
        degree = randRange(self.degree[1], self.degree[0])
        scale = randRange(self.scale[1], self.scale[0])
        tx = randRange(self.translate[1], self.translate[0])
        ty = randRange(self.translate[1], self.translate[0])
 
        im, params = RTS(sample, degree, self.center, translate=[tx, ty],
                         scale=(scale, scale))
        return {'image': im, 'theta': {'degree': degree,
                                       'scale': scale,
                                       'translate': [tx, ty]}}
    
def wrap(f):
    def f_(sample):
        if type(sample) is not dict:
            return f(sample)
        image, theta = sample['image'], sample['theta']
        return {'image': f(image), 'theta': theta}
    return f_
