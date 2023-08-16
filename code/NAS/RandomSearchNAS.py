import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform to tensor and normalize to 0 mean
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_set, validation_set = torch.utils.data.random_split(train_set, [int(len(train_set)*0.8), int(len(train_set)*0.2)])

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
total_training_batches = int(len(train_set)/32)

class MLP(nn.Module):
    def __init__(self, number_of_layers, input_shape, hidden_shape, output_shape, activation_function):
        super(MLP, self).__init__()
        layers = []
        for l in range(number_of_layers):
            if l == 0:
                layers.append(nn.Linear(input_shape, hidden_shape))
                layers.append(activation_function())
            elif l == number_of_layers-1:
                layers.append(nn.Linear(hidden_shape, output_shape))
            else:
                layers.append(nn.Linear(hidden_shape, hidden_shape))
                layers.append(activation_function())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.classifier(x)

architectures = []
num_of_architectures = 10
for i in range(num_of_architectures):
    num_layers=torch.randint(2,10, [1]).item()
    hidden = torch.randint(20, 200, [1]).item()
    architectures.append((
        MLP(
            number_of_layers=num_layers,
            input_shape=28 * 28,
            hidden_shape=hidden,
            output_shape=10,
            activation_function=nn.ReLU
        ), num_layers, hidden)

    )

for model_tuple in architectures:
    model = model_tuple[0].to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    epochs = 1
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0
        running_acc = 0
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                running_acc += (y == y_pred.argmax(dim=1)).sum().item()
            if (i+1) % 500 == 0:
                print(f'Epoch [{epoch}/{epochs}] - Batch [{i+1}/{total_training_batches}] - Training Loss {running_loss/500:.3f} - Training Accuracy {running_acc/((i+1)*32)*100:.2f}%')
    with torch.no_grad():
        model.eval()
        validation_loss, validation_acc, validation_batch_counter = 0, 0, 0
        for i, batch in enumerate(validate_loader, 0):
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            validation_loss += loss.item()
            validation_acc += (y == y_pred.argmax(dim=1)).sum().item()
            validation_batch_counter += batch_size
        validation_loss /= validation_batch_counter
        validation_acc /= validation_batch_counter
    print(f'Validation Loss {validation_loss:.3f} - Validation Accuracy {validation_acc*100:.2f}%')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, f'model_storage/layers_{model_tuple[1]}_hidden_{model_tuple[2]}_validation_acc_{validation_acc*100:.2f}.pt')
