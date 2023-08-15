import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


batch_size = 10
download_choice = True
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(size=32)
])
transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=download_choice, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=download_choice, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class DwPwPass(nn.Module):
    def __init__(self, in_channels, pw_out_features, kernel_size, padding=0):
        super(DwPwPass, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=pw_out_features,
            kernel_size=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, residual=None):
        x = self.depthwise_conv(x)
        x = nn.ReLU()(x)
        x = self.pointwise_conv(x)
        if residual is not None:
            x = nn.ReLU()(x + residual)
        else:
            x = nn.ReLU()(x)
        return x


class SimpleMobileNet(nn.Module):
    def __init__(self):
        super(SimpleMobileNet, self).__init__()
        self.dwpwpass1 = DwPwPass(in_channels=3, pw_out_features=32, kernel_size=3, padding=1)
        self.dwpwpass2 = DwPwPass(in_channels=32, pw_out_features=64, kernel_size=3, padding=1)
        self.dwpwpass3 = DwPwPass(in_channels=64, pw_out_features=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(8 * 8 * 128, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.dwpwpass1(x)
        x = self.maxpool(x)
        x = self.dwpwpass2(x)
        x = self.maxpool(x)
        x = self.dwpwpass3(x)
        x = self.classifier(x)
        return x



model = SimpleMobileNet()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=5e-3)
loss_function = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    running_loss = 0
    running_accuracy = 0
    running_counter = 0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        x = data[0]
        y = data[1]
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            running_accuracy += (y == y_pred.argmax(dim=1)).to(torch.float32).sum()
            running_counter += batch_size
            running_loss += loss.item()
        if (i+1) % 2000 == 0:
            # every 100 batches
            running_loss /= 2000
            running_accuracy /= running_counter
            print(f'Epoch: {epoch+1}/{epochs} - Batch {i+1}/5000 - Running loss - {running_loss:.3f} - Training Accuracy - {running_accuracy}')
