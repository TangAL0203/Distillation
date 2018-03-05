#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision
import math
import os
import math


# 定义有temperature的softmax函数
class T_Softmax(nn.Module):
    r'''
    Softmax is defined as :
    math:`f_i(x) = \frac{\exp(x_i/T)}{\sum_j \exp(x_j/T)}`
    Shape:
        - Input: any shape
        - Output: same as input
    Use
    '''
    def __init__(self, dim=None, T=1):
        super(T_Softmax, self).__init__()
        if dim==None:
            self.dim = 0 # 默认一行为一个输出
        else:
            self.dim = dim
        self.T = T # temperature
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        input.div_(self.T)
        input_tensor = input.data
        input_tensor = torch.pow(math.e, input_tensor)
        
        if self.dim == 0:
            sum_ex = input_tensor.sum(dim=1).repeat(input_tensor.shape[1],1).t()
        else:
            sum_ex = input_tensor.sum(dim=0).repeat(1,input_tensor.shape[0]).t()

        input_tensor = torch.div(input_tensor,sum_ex)
        input.data = input_tensor
    
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'




















