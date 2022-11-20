import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PATH = './cifar_net.pth'

# 测试网络置为0，继续训练网络置为1，重新训练网络置为2，同时测试训练数据与测试数据置为3
trainMode = 0
# 重复训练次数
REPEAT = 20


# 图像显示函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Vgg16_Net(nn.Module):
    def __init__(self):
        super(Vgg16_Net, self).__init__()
        # 2个卷积层和1个最大池化层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (32-2)/2+1 = 16    16*16*64

        )
        # 2个卷积层和1个最大池化层
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1 = 8    8*8*128
        )
        # 3个卷积层和1个最大池化层
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # (8-2)/2+1 = 4    4*4*256
        )
        # 3个卷积层和1个最大池化层
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1 = 2    2*2*512
        )
        # 3个卷积层和1个最大池化层
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1 = 1    1*1*512
        )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


# 自定义神经网络，包含两个卷积层，一个最大值池化层，三个全连接层

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 16, 5)
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


# 将模型导入GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 散粒噪声、随机翻转变换，实现数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# 标准变换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 将标准数据和增强数据合并
trainset_advance = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset_normol = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = trainset_advance.__add__(trainset_normol)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 测试数据
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 数据标签定义
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 初始化神经网络
net = Vgg16_Net()
# 记录学习数据和loss值用于作图
losslist = []
timelist = []

# 从本地导入上一次训练的神经网络
if trainMode == 0 or trainMode == 1 or trainMode == 3:
    net.load_state_dict(torch.load(PATH))

# 神经网络导入GPU
if trainMode == 1 or trainMode == 2:
    net.to(device)

# 采用交叉熵loss函数，标准SGD动量优化法，学习率0.001，动量0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练主程序
if trainMode == 1 or trainMode == 2:
    for epoch in range(REPEAT):
        running_loss = 0.0
        if __name__ == '__main__':
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                # 将数据导入GPU训练
                inputs, labels = inputs.to(device), labels.to(device)
                # 优化朝零梯度进行
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                # 反向传播求梯度，进行SGD优化
                loss.backward()
                optimizer.step()
                # 记录loss，每隔100次输出数据
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    losslist.append(running_loss / 100)
                    timelist.append(epoch * 25000 + (i + 1))
                    running_loss = 0.0
    print('Finished Training')
    # 保存模型
    torch.save(net.state_dict(), PATH)

# loss图绘制
if trainMode == 1 or trainMode == 2:
    if __name__ == '__main__':
        plt.plot(timelist, losslist)
        plt.xlabel("Times of Learning")
        plt.ylabel("Loss")
        plt.title("Loss changes as Learning")
        plt.show()

# 神经网络预测测试集，统计预测精度
if trainMode == 0 or trainMode == 3:
    net.to(device)
    if __name__ == '__main__':
        correct = 0
        total = 10000
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 将预测的最高分与标签比较，记录预测正确次数
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        print('神经网络预测10000个测试图像的精度是: {0:.2%} '.format(correct / total))

    if trainMode == 3 and __name__ == '__main__':
        correct = 0
        total = 100000
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 将预测的最高分与标签比较，记录预测正确次数
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        print('神经网络预测100000个训练图像的精度是: {0:.2%} '.format(correct / total))
