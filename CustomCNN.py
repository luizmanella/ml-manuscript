import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime

BATCH_SIZE = 64

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(5, 5), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=5 * 5 * 64, out_features=120),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(in_features=120, out_features=84),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(in_features=84, out_features=10)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=32, out_features=10)
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)
train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set)*0.8), int(len(train_set)*0.2)])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validate_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


# -------------------------------------------
# Instantiate variables related to training
# -------------------------------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = TestNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
training_loss = []
validation_acc = []
training_acc = []
mixed_precision = True

start_training_timestamp = datetime.now()
for epoch in range(num_epochs):
    running_loss = 0
    model.train()
    running_acc = 0
    running_sample_counter = 0
    for i, data in enumerate(train_loader):
        X = data[0].to(device)
        y = data[1].to(device)
        optimizer.zero_grad()
        if mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred = model(X)
                loss = criterion(y_pred, y)
        else:
            y_pred = model(X)
            loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_sample_counter += X.shape[0]
        running_acc += (y_pred.argmax(1) == y).to(torch.float).sum().item()

    with torch.no_grad():
        correct, total = 0, 0
        model.eval()
        for data in train_loader:
            X = data[0].to(device)
            y = data[1].to(device)
            y_pred = model(X)

            total += X.shape[0]
            correct += (y_pred.argmax(1) == y).to(torch.float).sum().item()
    training_loss.append(running_loss)
    validation_acc.append(correct/total)
    training_acc.append(running_acc / running_sample_counter)
    print(f'Epoch {epoch+1} - loss: {running_loss} | Validation Acc: {correct/total}')
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f'model_storage/testnet_epoch_{epoch+1}.pt')

print('-'*100)
print(f'Total training time: {datetime.now() - start_training_timestamp}')
with torch.no_grad():
    correct, total = 0, 0
    model.eval()
    for data in train_loader:
        X = data[0].to(device)
        y = data[1].to(device)
        y_pred = model(X)

        total += X.shape[0]
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
print(f'Training Accuracy: {correct / total}')

with torch.no_grad():
    correct, total = 0, 0
    model.eval()
    for data in validate_loader:
        X = data[0].to(device)
        y = data[1].to(device)
        y_pred = model(X)

        total += X.shape[0]
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
print(f'Testing Accuracy: {correct/total}')

plt.plot(training_loss)
plt.show()

