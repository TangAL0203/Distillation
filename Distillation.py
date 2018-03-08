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

    def forward(self, input):
        temp = torch.div(input, self.T)
        temp_exp = torch.exp(temp)
        
        if self.dim == 0:
            sum_exp = temp_exp.sum(dim=1).repeat(temp_exp.shape[1],1).t()
            out = torch.div(temp_exp, sum_exp)
        else:
            sum_exp = temp_exp.sum(dim=0).repeat(1,temp_exp.shape[0]).t()
            out = torch.div(temp_exp, sum_exp)

        return out

# define distillation loss function
def distillation_loss(nn.Module):
    r'''
    Distillation loss as flows:
    loss = w1*loss1 + w2*loss2
    loss1 = T_Softmax.
    '''





















