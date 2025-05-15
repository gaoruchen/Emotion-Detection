import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

import resnet50


def try_gpu(i=0):  # 测试GPU
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


batch_size = 32

# 定义数据转换
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    # torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomResizedCrop(
    #     (64, 64), scale=(0.8, 1), ratio=(0.5, 2)),
    torchvision.transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

train_ds = torchvision.datasets.ImageFolder(root='data/train', transform=transform_train)
test_ds = torchvision.datasets.ImageFolder(root='data/test', transform=transform_test)

train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

# 显示一部分数据
# images, labels = next(iter(train_iter))
# print(images.shape)
# print(images.type())
# print(labels.type())


# 实例化resnet50
net = resnet50.get_resnet50()
device = try_gpu()
net.to(device)
print('training on', device)
# 定义误差
criterion = nn.CrossEntropyLoss()
# 学习率优化器
# optimizer1 = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# optimizer2 = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.9)


def train():
    min_accuracy = 0
    for epoch in range(epoch_size):
        # if epoch < 5:
        #     optimizer = optimizer1
        # else:
        #     optimizer = optimizer2
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                # print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(
                #     epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.item()
                # ))
                y_loss.append(loss.item())
                x_epoch.append(epoch + 1.0 * batch_idx / len(train_loader))
        scheduler.step()
        print("第{}次epoch：".format(epoch + 1), end="")
        min_accuracy = test(device, min_accuracy)
    print("最大精确值为: {:.0f}%".format(100. * min_accuracy / len(test_loader.dataset)))


# 测试
def test(device, min_accuracy):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        test_loss += criterion(output, target).item()
        pred = torch.max(output.data, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if correct > min_accuracy:
        min_accuracy = correct
        # 保存最佳模型
        # torch.save(net.state_dict(), 'resnet.params')
        torch.save(net.state_dict(), 'resnet50.params')
    return min_accuracy


y_loss = []
x_epoch = []
# 训练和测试
epoch_size = 40
train()
# torch.save(net.state_dict(), 'resnet.params')
# 画出loss下降图
plt.xlabel("x_epoch")
plt.ylabel("loss")
plt.plot(x_epoch, y_loss)
plt.show()

# 验证模型是否保存

# clone = resnet.get_net()
# clone.load_state_dict(torch.load('resnet.params'))
# net.to('cpu')
#
# folder_path = 'data/train/Happiness'
# files = os.listdir(folder_path)
# image = Image.open(os.path.join(folder_path, files[0]))
# x = transform_test(image)
# x = x.unsqueeze(0)  # 添加 batch 维度

# clone.eval()
# net.eval()
# with torch.no_grad():
#     y_clone = clone(x)
#     y = net(x)
#
# pre_clone = torch.softmax(y_clone, dim=1)
# pre = torch.softmax(y, dim=1)
# print(y_clone)
# print(pre_clone)
# print('')
# print(y)
# print(pre)
# print(y == y_clone)
