# Eevaluation for sequence transformert network
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def plot_fill(x, lines, color='b', label='default'):
    for l in lines:
        plt.plot(x, l, color=color, alpha=0.2)
    plt.plot(x, np.array(lines).mean(0), color=color, label=label)
    
def getTheta(model, x):
    xs = model.localization(x)
    xs = xs.view(-1, model.num_flat_features(xs))
    theta = model.fc_loc(xs).cpu().data.numpy()
    return theta
    
def visualize_stn_ts(visual_loader, model, feature=0, top=3, use_cuda=True,
                     ylabel='magnitude', fontsize=15):
    # Get a batch of training data
    data, _ = next(iter(visual_loader))
    data = Variable(data, volatile=True)

    if use_cuda:
        data = data.cuda()
    
    input_tensor = data.cpu().data.squeeze(1).numpy()
    transformed_input_tensor = model.stn(data).cpu().data.squeeze(1).numpy()
    print("different?", np.abs(input_tensor - transformed_input_tensor).sum())
    
    n, t, d = input_tensor.shape

    nrows = min(n,top)
    fig, axes = plt.subplots(nrows=nrows, ncols=2)
    fig.tight_layout()
    print(getTheta(model, data)[:nrows,:])    
    for i in range(nrows):
        a = transformed_input_tensor[i,:,feature]
        b = input_tensor[i,:,feature]
        m, M = min(min(a), min(b)), max(max(a), max(b))

        plt.subplot("%d2%d" % (nrows, i*2+1))
        plt.plot(input_tensor[i,:,feature])
        # ax = plt.gca()
        # ylim = ax.get_ylim()
        ylim = [m-0.01, M+0.01]
        plt.ylim(ylim)
        plt.title('original', fontsize=fontsize)
        plt.xlabel('time', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.grid(linestyle='--')        

        plt.subplot("%d2%d" % (nrows, i*2+2))        
        plt.plot(transformed_input_tensor[i,:,feature])
        plt.ylim(ylim)
        plt.title('transformed', fontsize=fontsize)
        plt.xlabel('time', fontsize=fontsize)
        plt.grid(linestyle='--')

def theta_cluster(loader, model, use_cuda=True):
    theta = []
    for x, _ in loader:
        x = Variable(x, volatile=True)
        if use_cuda:
            x = x.cuda()

        t = getTheta(model,x)
        theta.append(t)
        
    return np.vstack(theta)
                                                                
FEATURE_NAMES = ['Capillary refill rate->0.0', 'Capillary refill rate->1.0', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glascow coma scale eye opening->To Pain', 'Glascow coma scale eye opening->3 To speech', 'Glascow coma scale eye opening->1 No Response', 'Glascow coma scale eye opening->4 Spontaneously', 'Glascow coma scale eye opening->None', 'Glascow coma scale eye opening->To Speech', 'Glascow coma scale eye opening->Spontaneously', 'Glascow coma scale eye opening->2 To pain', 'Glascow coma scale motor response->1 No Response', 'Glascow coma scale motor response->3 Abnorm flexion', 'Glascow coma scale motor response->Abnormal extension', 'Glascow coma scale motor response->No response', 'Glascow coma scale motor response->4 Flex-withdraws', 'Glascow coma scale motor response->Localizes Pain', 'Glascow coma scale motor response->Flex-withdraws', 'Glascow coma scale motor response->Obeys Commands', 'Glascow coma scale motor response->Abnormal Flexion', 'Glascow coma scale motor response->6 Obeys Commands', 'Glascow coma scale motor response->5 Localizes Pain', 'Glascow coma scale motor response->2 Abnorm extensn', 'Glascow coma scale total->11', 'Glascow coma scale total->10', 'Glascow coma scale total->13', 'Glascow coma scale total->12', 'Glascow coma scale total->15', 'Glascow coma scale total->14', 'Glascow coma scale total->3', 'Glascow coma scale total->5', 'Glascow coma scale total->4', 'Glascow coma scale total->7', 'Glascow coma scale total->6', 'Glascow coma scale total->9', 'Glascow coma scale total->8', 'Glascow coma scale verbal response->1 No Response', 'Glascow coma scale verbal response->No Response', 'Glascow coma scale verbal response->Confused', 'Glascow coma scale verbal response->Inappropriate Words', 'Glascow coma scale verbal response->Oriented', 'Glascow coma scale verbal response->No Response-ETT', 'Glascow coma scale verbal response->5 Oriented', 'Glascow coma scale verbal response->Incomprehensible sounds', 'Glascow coma scale verbal response->1.0 ET/Trach', 'Glascow coma scale verbal response->4 Confused', 'Glascow coma scale verbal response->2 Incomp sounds', 'Glascow coma scale verbal response->3 Inapprop words', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH', 'mask->Capillary refill rate', 'mask->Diastolic blood pressure', 'mask->Fraction inspired oxygen', 'mask->Glascow coma scale eye opening', 'mask->Glascow coma scale motor response', 'mask->Glascow coma scale total', 'mask->Glascow coma scale verbal response', 'mask->Glucose', 'mask->Heart Rate', 'mask->Height', 'mask->Mean blood pressure', 'mask->Oxygen saturation', 'mask->Respiratory rate', 'mask->Systolic blood pressure', 'mask->Temperature', 'mask->Weight', 'mask->pH']
