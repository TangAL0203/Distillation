#-*-coding:utf-8-*-
import models.AlexNet as AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision
import math
import os
import getpass

if getpass.getuser() == 'tsq':
    train_batch_size = 8
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_batch_size = 32

use_gpu = torch.cuda.is_available()
num_batches = 0

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        total += label.size(0)

    print("Accuracy :", float(correct) / total)
    model.train()

def train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input)
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, Variable(label)).backward() 
    optimizer.step()
    return criterion(output, Variable(label)).data

def train_epoch(model, train_loader, optimizer=None):
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%50 == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1

# 训练一个epoch,测试一次
def train_test(model, train_loader, test_loader, optimizer=None, epoches=10):
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum=0.9)

    for i in range(epoches):
        model.train()
        print("Epoch: ", i)
        train_epoch(model, train_loader, optimizer)
        test(model, test_loader)
    print("Finished training.")

# 初始化模型参数
#　从0开始训练一个二分类器
# 对conv层和全连接层参数初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=1)
        m.bias.data.zero_()

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedAlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # model = AlexNet.AlexNet(2) # 从头定义一个网络
    model = ModifiedAlexNet(2) # 
    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")
    root = "./data/train"
    train_loader = dataset.train_loader(root, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    test_loader = dataset.test_loader(root, batch_size=1, num_workers=4, pin_memory=True)

    # model.apply(weight_init) # 初始化参数
    train_test(model, train_loader, test_loader, optimizer=None, epoches=5)
    # save model
    save_type = 1
    save_path = '/home/smiles/Distillation/models/AlexNet1.pth'
    # 方式一： 只保存参数
    # 方式二： 保存整个模型(模型结构和模型参数)
    if save_type == 1:
        torch.save(model.state_dict(), save_path)
    elif save_type == 2:
        save_path = '/home/smiles/Distillation/models/AlexNet1.pth'
        torch.save(model, save_path)



#　论文提到，乘以Ｔ^2，损失函数的梯度就与T无关。(公式4有提到)
#　　teacher_output部分设置为.detach()，这样就不会计算梯度值。相当于这里的KL散度就等于计算P和Q的交叉熵了。(两者相差一个H(P))


























