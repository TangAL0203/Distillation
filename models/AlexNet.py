#-*-coding:utf-8-*-
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision

class LRN(nn.Module):
    def __init__(self, local_size=1, bias = 1.0, alpha=1.0, beta=5, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS # local_size为奇数，为偶数不确定怎么处理？
        # True: 通道间做归一化(local_size表示求和通道个数) False：通道内做归一化(local_size表示求和区间边长)
        if self.ACROSS_CHANNELS: # LRN处理方式一：在通道间进行归一化
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), # kernel_size 为 一个元组
                    stride=1, # stride 可以为一个整数型(通道，H，W三者都一样)，或者是一个元组
                    padding=(int((local_size-1.0)/2), 0, 0)) # 在通道上进行填充,在0.2.0_3版本上报错
            # 经过average处理之后，不同通道的单元进行了求和，然后与kernel_size相除，后续需要乘以local_size
        else: # LRN处理方式二：在通道内做归一化。在2D空间上进行单侧抑制处理
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,              
                    padding=int((local_size-1.0)/2))
        self.local_size = local_size
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1) #　对不同通道的单元求平方，需要在dim=1处插入一个维度，变成ＮＣＤＨＷ(D是表示输入或输出通道)，关于nn.AvgPool3d请查看API手册
            div = self.average(div).mul(self.local_size).squeeze(1) # 在通道间求和，平均，再乘以local_size变成求和的结果，在dim=1去掉
            div = div.mul(self.alpha).add(self.bias).pow(self.beta) # 按照LRN公式进行处理
        else:
            div = x.pow(2)
            div = self.average(div).mul(self.local_size*self.local_size) # 待确认？
            div = div.mul(self.alpha).add(self.bias).pow(self.beta)
        x = x.div(div)
        return x

# 共有１６个参数(加上bias参数)
class AlexNet(nn.Module):
    def __init__(self, num_classes = 2): # 默认为两类，猫和狗
#         super().__init__() # python3
        super(AlexNet, self).__init__()
        # 开始构建AlexNet网络模型，5层卷积，3层全连接层
        # 5层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 3层全连接层
        # 前向计算的时候，最开始输入需要进行view操作，将3D的tensor变为1D
        self.fc6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = x.view(-1, 6*6*256)
        x = self.fc8(self.fc7(self.fc6(x)))
        return x




