import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),#用来做数据增强的,为了防止训练出现过拟合,通常在小型数据集上,通过随机翻转图片
     transforms.RandomGrayscale(),#随机调整图片的亮度
     transforms.ToTensor(),#数据集加载时，默认的图片格式是numpy,所以通过transforms转换成Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,#batch_size单批次图片的数量
                                          shuffle=True, num_workers=0)#shuffle = True 表明提取数据时，随机打乱顺序

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)#num_workers指定了工作线程的数量

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')#分类的各个类别

class Net(nn.Module):#定义一个神经网络
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
        nn.Conv2d(3, 64, 3,padding = 1),nn.BatchNorm2d(64),nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3,padding = 1),nn.BatchNorm2d(128),nn.PReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3,padding = 1),nn.BatchNorm2d(256),nn.PReLU(),
        nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 4*4*256)
        x = self.fc1(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()#交叉熵作损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#优化器

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()#将优化器的所有参数梯度缓存器置零
        # forward + backward + optimize
        outputs = net(inputs)#获得输出
        loss = criterion(outputs, labels)#求损失函数
        loss.backward()#计算梯度
        optimizer.step()#更新神经网络的参数

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
