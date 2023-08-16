import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    def __init__(
            self,
            id,
            trainset,
            testset,
            batch_size,
            model,
            loss_function,
            optimizer,
            lr=1e-3) -> None:
        self.id = id
        self.batch_size = batch_size
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=len(testset))
        self.model = model.to(DEVICE)
        self.loss_function = loss_function()
        self.optimizer = optimizer(model.parameters(), lr=lr)

    def train_local_model(self, epoch):
        self.model.train()
        average_epoch_loss = 0
        average_epoch_acc = 0
        for i, batch in enumerate(self.trainloader, 0):
            self.optimizer.zero_grad()
            x = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()

            # Store loss and acc
            with torch.no_grad():
                average_epoch_loss += loss.item()
                average_epoch_acc += (y == y_pred.argmax(dim=1)).sum().item()
        average_epoch_loss /= len(self.trainloader)
        average_epoch_acc /= (len(self.trainloader) * self.batch_size)
        print(
            f'Agent {self.id} - Batch [{epoch[0]}/{epoch[1]}] - Training Loss {average_epoch_loss:.3f} - Training Accuracy {average_epoch_acc:.3f}')
        return (average_epoch_loss, average_epoch_acc)

    @torch.no_grad()
    def test_global_model(self):
        self.model.eval()
        # for i, batch in enumerate(self.testloader, 0):
        for x, y in self.testloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            total_loss = loss.item()
            accuracy = (y == y_pred.argmax(dim=1)).sum().item() / x.shape[0]
        print(f'Agent {self.id} - Testing Loss {total_loss} - Testing Accuracy {accuracy}')
        return total_loss, accuracy


class MLP(nn.Module):
    def __init__(self, initial_weights):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 10)

        self.fc1.weight.data = initial_weights[0] / torch.max(initial_weights[0])
        self.fc2.weight.data = initial_weights[1] / torch.max(initial_weights[1])
        self.fc3.weight.data = initial_weights[2] / torch.max(initial_weights[2])
        self.fc4.weight.data = initial_weights[3] / torch.max(initial_weights[3])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Load and preprocess MNIST dataset
batch_size = 10
number_of_agents = 4
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
agents_training_data = torch.utils.data.random_split(trainset, [int(len(trainset) / number_of_agents) for _ in
                                                                range(number_of_agents)])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
agents_testing_data = torch.utils.data.random_split(testset, [int(len(testset) / number_of_agents) for _ in
                                                              range(number_of_agents)])

epochs = 1
initial_weights = [
    torch.randn((200, 28 * 28)),
    torch.randn((200, 200)),
    torch.randn((200, 200)),
    torch.randn((10, 200)),
]
ids = range(0, number_of_agents)
list_of_agents = []
for id in ids:
    _agent_id = id + 1
    _agent_trainset = agents_training_data[id]
    _agent_testset = agents_testing_data[id]
    _agent_model = MLP(initial_weights=initial_weights)
    list_of_agents.append(Agent(
        id=_agent_id,
        trainset=_agent_trainset,
        testset=_agent_testset,
        batch_size=batch_size,
        model=_agent_model,
        loss_function=nn.CrossEntropyLoss,
        optimizer=optim.Adam
    ))

for epoch in range(1, epochs+1):
    avg_loss, avg_acc = 0, 0
    input_weights = torch.zeros(200, 28 * 28, device=DEVICE)
    h1_weights = torch.zeros(200, 200, device=DEVICE)
    h2_weights = torch.zeros(200, 200, device=DEVICE)
    output_weights = torch.zeros(10, 200, device=DEVICE)
    for agent in list_of_agents:
        a_loss, a_acc = agent.train_local_model((epoch, epochs))
        avg_loss += a_loss
        avg_acc += a_acc
        input_weights += agent.model.fc1.weight.data
        h1_weights += agent.model.fc2.weight.data
        h2_weights += agent.model.fc3.weight.data
        output_weights += agent.model.fc4.weight.data
    avg_loss /= number_of_agents
    avg_acc /= number_of_agents
    print(f'TRAINING: Average loss: {avg_loss} - Average accuracy: {avg_acc}')
    input_weights /= number_of_agents
    h1_weights /= number_of_agents
    h2_weights /= number_of_agents
    output_weights /= number_of_agents
    for agent in list_of_agents:
        agent.model.fc1.weight.data = input_weights
        agent.model.fc2.weight.data = h1_weights
        agent.model.fc3.weight.data = h2_weights
        agent.model.fc4.weight.data = output_weights

avg_loss, avg_acc = 0, 0
for agent in list_of_agents:
    a_loss, a_acc = agent.test_global_model()
    avg_loss += a_loss
    avg_acc += a_acc
    break
avg_loss /= number_of_agents
avg_acc /= number_of_agents
print(f'TESTING: Average loss: {avg_loss} - Average accuracy: {avg_acc}')