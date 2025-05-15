from torch import nn
import torch.nn.functional as F


# 定义残差块
class Residual50(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)  # 高宽减半看strides是否为2
        self.conv3 = nn.Conv2d(num_channels, num_channels*4,
                               kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels*4,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels*4)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.conv4:
            x = self.conv4(x)
        y += x
        return F.relu(y)


# 定义块
def block(input_channels, num_channels, num_residuals,
          first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            if first_block:
                blk.append(Residual50(input_channels, num_channels,
                                      use_1x1conv=True))  # 第一个模块的高宽无需减半
            else:
                blk.append(Residual50(input_channels, num_channels,
                                      use_1x1conv=True, strides=2))
        else:
            blk.append(Residual50(num_channels*4, num_channels))
    return blk


def get_resnet50():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*block(64, 64, 3, first_block=True))
    b3 = nn.Sequential(*block(256, 128, 4))
    b4 = nn.Sequential(*block(512, 256, 6))
    b5 = nn.Sequential(*block(1024, 512, 3))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(2048, 7))
    return net
