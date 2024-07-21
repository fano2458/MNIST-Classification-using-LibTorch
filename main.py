import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Define constants
data_root = "./data"
train_batch_size = 64
test_batch_size = 1000
num_epochs = 15
log_interval = 10

# Define the CNN model
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
    self.fc1 = nn.Linear(120, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.gelu(F.max_pool2d(self.conv1(x), 2))
    x = F.gelu(F.max_pool2d(self.conv2(x), 2))
    x = F.gelu(self.conv3(x))
    x = x.view(-1, 120)
    x = F.gelu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

# Define training function
def train(epoch, model, device, data_loader, optimizer, dataset_size):
  model.train()
  total_loss = 0
  for batch_idx, (data, target) in enumerate(data_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    
  print(f'\rTrain Epoch: {epoch} [{dataset_size}] Loss: {total_loss / dataset_size:.4f}')

# Define testing function
def test(epoch, model, device, data_loader, dataset_size):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction="sum").item()
      pred = output.argmax(1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= dataset_size
  print(f'Test set: Average loss: {test_loss:.4f} | Accuracy: {correct / dataset_size:.3f}')

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and move it to the device
model = Net().to(device)

# Define data transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load training and testing datasets
train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

avg_time = 0.0
# Train and test the model for specified epochs
for epoch in range(1, num_epochs + 1):
  train(epoch, model, device, train_loader, optimizer, len(train_dataset))
  begin_time = time.time()
  test(epoch, model, device, test_loader, len(test_dataset))
  epoch_time = time.time() - begin_time
  print(f"Epoch {epoch} finished. Time taken: {epoch_time} seconds")
  avg_time += epoch_time

print("Average epoch time is", avg_time/num_epochs)