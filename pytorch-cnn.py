import torch
import torchvision
import torchvision.transforms as transforms


### 1. 加载数据

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0) # windows系统设置num_workers=0

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0) # windows系统设置num_workers=0

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




import matplotlib.pyplot as plt
import numpy as np

# 显示图像函数


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取一些训练图像
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


### 2. 定义卷积神经网络

import torch.nn as nn

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积
        x = self.pool(F.relu(self.conv2(x)))
        # 全连接
        x = x.view(-1, 16 * 5 * 5)

        # hiddern layer 1
        x = F.relu(self.fc1(x))
        # hideen layer2
        x = F.relu(self.fc2(x))
        # output layer
        x = self.fc3(x)
        return x

#instanize the model
net = Net()



### 3. 定义损失函数和优化器

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### 4. 训练模型


for epoch in range(2):  # 对数据集进行多次循环

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入; 数据格式是 [inputs, labels] 列表
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # 前向计算输出
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 优化，更新权重
        optimizer.step()

        # 输出统计数据
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



### 测试模型
dataiter = iter(testloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

predicted = torch.max(outputs, 1) # 获取最大值所在的类

predicted
predicted.indices

print('Predicted: ', ' '.join('%5s' % classes[predicted.indices[j]]
for j in range(4)))