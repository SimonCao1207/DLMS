import torch                                                                                                                                                                
import argparse
import torch.nn as nn                                                                                                                                                       
import torch.optim as optim                                                                                                                                                 
from torchvision import datasets, transforms                                                                                                                                
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
from const import *
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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

def train_model(lr):                                                                                                                  
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)                                                                               
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)                                                                                                       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()                                                                                                                                                      
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    num_epochs = 1
    open(DEFAULT_LOG_DIR_OUT, 'w').close() # clear content
    for epoch in range(1, num_epochs + 1):                                                                                                                                                                   
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                with open(DEFAULT_LOG_DIR_OUT, "a") as f:
                    f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            # if batch_idx > 10:  
            #     break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate (default: 1.0)")
    args = parser.parse_args()
    train_model(args.lr)
