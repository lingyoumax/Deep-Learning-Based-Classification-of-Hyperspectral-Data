import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [#transforms.RandomHorizontalFlip(),#用来做数据增强的，为了防止训练出现过拟合，通常在小型数据集上，通过随机翻转图片
     #transforms.RandomGrayscale(),#随机调整图片的亮度
     transforms.ToTensor(),#数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#对输入图片进行标准化

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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()#交叉熵作损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#优化器

for epoch in range(2):  # loop over the dataset multiple times

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

print('Finished Training,Please let me show you more detail!')
print("conv1")
print("net.conv1.weight")
print(net.conv1.weight)
print(net.conv1.weight.size())
print("net.conv1.bias")
print(net.conv1.bias)
print(net.conv1.bias.size())
print("conv2")
print("net.conv2.weight")
print(net.conv2.weight)
print(net.conv2.weight.size())
print("net.conv2.bias")
print(net.conv2.bias)
print(net.conv2.bias.size())
print("fc1")
print("net.fc1.weight")
print(net.fc1.weight)
print(net.fc1.weight.size())
print("net.fc1.bias")
print(net.fc1.bias)
print(net.fc1.bias.size())
print("fc2")
print("net.fc2.weight")
print(net.fc2.weight)
print(net.fc2.weight.size())
print("net.fc2.bias")
print(net.fc2.bias)
print(net.fc2.bias.size())
print("fc3")
print("net.fc3.weight")
print(net.fc3.weight)
print(net.fc3.weight.size())
print("net.fc3.bias")
print(net.fc3.bias)
print(net.fc3.bias.size())
