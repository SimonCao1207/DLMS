import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import warnings
from const import *
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        running_loss = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            with open(DEFAULT_LOG_DIR_OUT, "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return running_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    dct = {
        "test_loss" :  test_loss / len(test_loader.dataset),
        "acc" : 100. * correct / len(test_loader.dataset)
    }
    json_object = json.dumps(dct, indent=4)

    with open(DEFAULT_RESULT_DIR_OUT, "w") as outfile:
        outfile.write(json_object)

def plot_loss(losses, task_id):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig(f"{BASE_LOG_DIR}/{task_id}.png")

def save_loss(losses, task_id):
    with open(f"{BASE_LOG_DIR}/{task_id}.txt", "w") as txt_file:
        for loss in losses:
            txt_file.write(f"{loss}\n")

def run(args):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    subset_indices = torch.randperm(len(train_dataset))[:1000]
    sub_train_dataset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(sub_train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(args.dropout_rate)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    open(DEFAULT_LOG_DIR_OUT, 'w').close() # clear content
    train_losses = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
    # plot_loss(train_losses, args.task_id)
    save_loss(train_losses, args.task_id)
    test(model, device, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    args = parser.parse_args()

    run(args)
