import torch                                                                                                                                                                
import argparse
import torch.nn as nn                                                                                                                                                       
import torch.optim as optim                                                                                                                                                 
from torchvision import datasets, transforms                                                                                                                                
import torch.nn.functional as F
from torch.utils.data import DataLoader


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

# Load MNIST dataset                                                                                                                                                        
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])                                                                               
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)                                                                               
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)                                                                                                       
                                                                                                                                                                            
# Training function                                                                                                                                                         
def train_model(learning_rate, epochs=1):                                                                                                                  
    model = Net()                                                                                                                                                      
    criterion = nn.CrossEntropyLoss()                                                                                                                                       
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)                                                                                                             
                                                                                                                                                                            
    for epoch in range(epochs):                                                                                                                                             
        running_loss = 0.0                                                                                                                                                  
        for i, data in enumerate(train_loader, 0):                                                                                                                          
            inputs, labels = data                                                                                                                                           
            optimizer.zero_grad()                                                                                                                                           
            outputs = model(inputs)                                                                                                                                         
            loss = criterion(outputs, labels)                                                                                                                               
            loss.backward()                                                                                                                                                 
            optimizer.step()                                                                                                                                                
            running_loss += loss.item()                                                                                                                                     
            if i % 100 == 99:                                                                                                                                               
                print(epoch + 1, i + 1, running_loss / 100)                                                                                                     
                running_loss = 0.0                                                                                                                                          
    return "Training completed" 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate (default: 1.0)")
    args = parser.parse_args()
    train_model(args.lr)