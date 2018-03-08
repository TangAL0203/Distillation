#-*-coding:utf-8-*-
import shutil
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data_name', metavar='DATA_NAME', type=str, default='CIFAR10', help='dataset name')
parser.add_argument('--classes_num', type=int, default=2, help='classes num of dataset')
parser.add_argument('--data_path', metavar='DATA_PATH', type=str, default=['./data/train', './data/test'],
                    help='path to dataset', nargs=2)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--pretrained', default=True, help='if use pre-trained model to init')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()

import logging
#======================generate logging imformation===============
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# you should assign log_name first
log_name = args.arch+'_'+args.data_name+'.log'
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


infoLogger.info("arch is: {}".format(args.arch))
infoLogger.info("data is: {}".format(args.data_name))
infoLogger.info("lr   is: {}".format(args.lr))



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

def adjust_learning_rate(optimizer, epoch):
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

def train_batch(model, optimizer, criterion, batch, label): 
    label = label.cuda(async=True)
    label_var = Variable(label)
    output = model(Variable(batch.cuda()))
    # compute gradient and do SGD step
    optimizer.zero_grad() # 
    criterion(output, label_var).backward()
    optimizer.step()

    return criterion(output, label_var).data

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, criterion, batch, label)
        if num_batches%50 == 0:
            temp_str = '%23s%-9s%-13s'%(('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
            infoLogger.info(temp_str)
        num_batches +=1

# save the checkpoint which has the best accuracy
def save_best_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    global best_prec1
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.arch+'_'+args.data_name+'_'+str(best_prec1)+'.pth') # Copy the contents of the file to another file
        os.remove(filename)

local_weights = False

def main():
    global args, best_prec1
    # create model
    if args.arch == 'mobilenet':
        model = MobileNet(args.classes_num)
        if args.pretrained and not args.resume :
            pretrained_path = './models/mobilenet_sgd_68.848.pth.tar'
            load_checkpoint(model, pretrained_path)
        elif args.pretrained and  args.resume :
            load_checkpoint(model, args.resume)

    elif args.arch == 'resnet50':
        if not local_weights and args.pretrained:
            model = models.resnet50(pretrained=True)
        elif local_weights and args.pretrained:
            model = models.resnet50(pretrained=False)
            load_checkpoint(model, pretrained_path)
    elif args.arch == 'resnet34':
        if not local_weights and args.pretrained:
            model = models.resnet34(pretrained=True)
        elif local_weights and args.pretrained:
            model = models.resnet34(pretrained=False)
            load_checkpoint(model, pretrained_path)

    if torch.cuda.is_available():
        model.cuda()

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
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    # get previous accuracy
    if args.resume:
        checkpoint = torch.load(args.resume)
        best_prec1 = checkpoint['best_prec1']

    # start training
    infoLogger.info("Start training.")
    model.train()
    for epoch in range(args.epochs):
        infoLogger.info("Epoch: "+str(epoch))
        train_epoch(model, train_loader, criterion, optimizer)
        current_acc = test(model, test_loader)
        best_prec1 = max(current_acc, best_prec1)
        save_best_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'data_name': args.data_name,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, 
                is_best=current_acc > best_prec1, 
                filename=args.arch+'_'+args.data_name+'_'+str(current_acc)+'.pth'
            )

if __name__ == "__main__":
    main()



# load model pretrainde from imagenet dataset and fine-tune
# python mobilenet.py --data_name catdog --classes_num 2 --data_path ./data/train ./data/test --arch mobilenet --epochs 10 --pretrained True

# load model which has been trainde on catdog or cifar10 adn keep on training 
# python mobilenet.py --data_name catdog --classes_num 2 --data_path ./data/train ./data/test --arch mobilenet --epochs 10 --pretrained True --resume ./mobilenet_catdog_0.906.pth --lr 0.0001

# train the resnet50 
# python mobilenet.py --data_name catdog --classes_num 2 --data_path ./data/train ./data/test --arch resnet50 --epochs 10 --pretrained True 