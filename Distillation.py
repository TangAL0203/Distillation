#-*-coding:utf-8-*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision.transforms as transforms
import math
import os
import math
import argparse
from mobilenet import *

parser = argparse.ArgumentParser(description='Pytorch Distillation Experiment')

parser.add_argument('--student_arch', metavar='ARCH', default='mobilenet', help='student model architecture')
parser.add_argument('--teacher_arch', metavar='ARCH', default='resnet50', help='teacher model architecture')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--data_name', metavar='DATA_NAME', type=str, default='CIFAR10', help='dataset name')
parser.add_argument('--classes_num', type=int, default=2, help='classes num of dataset')
parser.add_argument('--data_path', metavar='DATA_PATH', type=str, default=['./data/train', './data/test'],
                    help='path to train and test dataset', nargs=2)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to student_arch and teacher_arch latest checkpoint (default: none)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--pretrained', default=True, help='if use pre-trained model to init')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

args = parser.parse_args()

import logging
#======================generate logging imformation===============
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# you should assign log_name first such as mobilenet_resnet50_CIFAR10.log
log_name = args.student_arch+'_'+args.teacher_arch+'_'+args.data_name+'.log'
TrainInfoPath = os.path.join(log_path, log_name)
# formater
# formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
formatter = logging.Formatter('%(levelname)s %(message)s')
# cmd Handler 
cmdHandler = logging.StreamHandler() 
# cmdHandler.setFormatter(formatter)
# File Handler including info
infoFileHandler = logging.FileHandler(TrainInfoPath, mode='w')
infoFileHandler.setFormatter(formatter)
# info Logger
infoLogger = logging.getLogger('info') 
infoLogger.setLevel(logging.DEBUG) 
infoLogger.addHandler(cmdHandler)
infoLogger.addHandler(infoFileHandler)

best_prec1 = 0 # save best accuracy value
num_batches = 0 # Record the number of batches

infoLogger.info("student_arch is: {}".format(args.student_arch))
infoLogger.info("teacher_arch is: {}".format(args.teacher_arch))
infoLogger.info("data         is: {}".format(args.data_name))
infoLogger.info("init lr      is: {}".format(args.lr))
infoLogger.info("batch size   is: {}".format(args.batch_size))
infoLogger.info("epochs       is: {}".format(args.epochs))
infoLogger.info("momentum     is: {}".format(args.momentum))
infoLogger.info("weight_decay is: {}".format(args.weight_decay))

class MobileNet(nn.Module):
    def __init__(self, classes_num):
        super(MobileNet, self).__init__()
        self.classes_num = classes_num

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # add zero padding=1 to maintain size unchanged
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.classes_num)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

# load params from weights file
def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

def adjust_lr_EachEpoch(optimizer):
    """the learning rate to the initial LR decayed by 10 every 30 epochs
        and multiplied by 0.926 at every epoch.
    """
    global args
    lr = args.lr * 0.926
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_30Epoch(optimizer, epoch):
    global args
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define test func
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for i, (batch, label) in enumerate(test_loader):
        output = model(Variable(batch.cuda()))
        pred_label = output.data.max(1)[1]
        correct += pred_label.cpu().eq(label).sum()
        total +=batch.size(0)
    infoLogger.info("Accuracy :"+str(round( float(correct) / total , 3 )))
    model.train()
    return round( float(correct) / total , 3 )

# define train func
def test_batch(model, batch, label):
    model.eval()
    output = model(Variable(batch.cuda()))
    pred_label = output.data.max(1)[1]
    return float(pred_label.cpu().eq(label).sum()) / batch.size(0)
    model.train()

def train_batch(student_model, teacher_model, optimizer, criterion, batch, label): 
    label = label.cuda(async=True)
    label_var = Variable(label)
    output = student_model(Variable(batch.cuda()))
    teacher_output = teacher_model(Variable(batch.cuda()))
    teacher_output = teacher_output.detach()
    # compute gradient and do SGD step
    optimizer.zero_grad() # 
    criterion(output, label_var, teacher_output, T=20.0, alpha=0.7).backward()
    optimizer.step()

    return criterion(output, label_var, teacher_output, T=20.0, alpha=0.7).data

def train_epoch(student_model, teacher_model, train_loader, criterion, optimizer):
    student_model.train()
    teacher_model.eval()
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(student_model, teacher_model, optimizer, criterion, batch, label)
        if num_batches%args.print_freq == 0:
            temp_str = '%23s%-9s%-13s'%(('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
            infoLogger.info(temp_str)
        num_batches +=1

# save the checkpoint which has the best accuracy
def save_best_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    global best_prec1
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.student_arch+'_'+args.teacher_arch+'_'+args.data_name+'_'+str(best_prec1)+'.pth') # Copy the contents of the file to another file

# define the loss func of distillation
def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def main():
    global args, best_prec1
    # create student_model and teacher_model teacher_arch
    if args.student_arch == 'mobilenet':
        student_model = MobileNet(args.classes_num)
        if args.pretrained and len(args.resume)==2:
            pretrained_path = args.resume[0]
            load_checkpoint(student_model, pretrained_path)
    elif args.student_arch == 'resnet50' and len(args.resume)==2:
        student_model = models.resnet50(pretrained=False)
        load_checkpoint(student_model, args.resume[0])
    elif args.student_arch == 'resnet34' and len(args.resume)==2:
        student_model = models.resnet34(pretrained=False)
        load_checkpoint(student_model, args.resume[0])

    if args.teacher_arch == 'mobilenet':
        teacher_model = MobileNet(args.classes_num)
        if len(args.resume)==2:
            load_checkpoint(teacher_model, args.resume[1])
    elif args.teacher_arch == 'resnet50' and len(args.resume)==2:
        teacher_model = models.resnet50(pretrained=False)
        load_checkpoint(teacher_model, args.resume[1])
    elif args.teacher_arch == 'resnet34' and len(args.resume)==2:
        teacher_model = models.resnet34(pretrained=False)
        load_checkpoint(teacher_model, args.resume[1])


    if torch.cuda.is_available():
        student_model.cuda()
        teacher_model.cuda()

    # create data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


    if args.data_name in torchvision.datasets.__all__:   # cifar10 dataset
        
        trainset = torchvision.datasets.CIFAR10('./data/',train=True,download=False,transform=train_transforms)
        testset  = torchvision.datasets.CIFAR10('./data/',train=False,download=False,transform=test_transforms)

        train_loader = torch.utils.data.DataLoader(trainset, 
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
            )

        test_loader = torch.utils.data.DataLoader(testset, 
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True
            )
    else:  # catdog dataset
        trainset = torchvision.datasets.ImageFolder(args.data_path[0], train_transforms)
        testset = torchvision.datasets.ImageFolder(args.data_path[1], test_transforms)

        train_loader = torch.utils.data.DataLoader(trainset, 
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
            )

        test_loader = torch.utils.data.DataLoader(testset, 
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True
            )

    # define loss function (criterion) and optimizer
    criterion = distillation.cuda()

    optimizer = optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    # get previous accuracy
    # if args.resume and args.pretrained:
    #     checkpoint = torch.load(args.resume)
    #     best_prec1 = checkpoint['best_prec1']

    # start training
    infoLogger.info("Start training.")
    for epoch in range(args.epochs):
        infoLogger.info("Epoch: "+str(epoch))
        train_epoch(student_model, teacher_model, train_loader, criterion, optimizer)
        current_acc = test(student_model, test_loader)
        best_prec1 = max(current_acc, best_prec1)
        save_best_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.student_arch,
                    'data_name': args.data_name,
                    'state_dict': student_model.state_dict(),
                    'best_prec1': best_prec1,
                }, 
                is_best=current_acc > best_prec1, 
                filename=args.student_arch+'_'+args.teacher_arch+'_'+args.data_name+'_'+str(current_acc)+'.pth'
            )


# usage
# python Distillation.py --student_arch mobilenet --teacher_arch resnet50 --data_name catdog --classes_num 2 --epochs 30 --pretrained True --resume *.pth *.pth --print_freq 50






# 定义有temperature的softmax函数
# class T_Softmax(nn.Module):
#     r'''
#     Softmax is defined as :
#     math:`f_i(x) = \frac{\exp(x_i/T)}{\sum_j \exp(x_j/T)}`
#     Shape:
#         - Input: any shape
#         - Output: same as input
#     Use
#     '''
#     def __init__(self, dim=None, T=1):
#         super(T_Softmax, self).__init__()
#         if dim==None:
#             self.dim = 0 # 默认一行为一个输出
#         else:
#             self.dim = dim
#         self.T = T # temperature

#     def forward(self, input):
#         temp = torch.div(input, self.T)
#         temp_exp = torch.exp(temp)
        
#         if self.dim == 0:
#             sum_exp = temp_exp.sum(dim=1).repeat(temp_exp.shape[1],1).t()
#             out = torch.div(temp_exp, sum_exp)
#         else:
#             sum_exp = temp_exp.sum(dim=0).repeat(1,temp_exp.shape[0]).t()
#             out = torch.div(temp_exp, sum_exp)

#         return out













