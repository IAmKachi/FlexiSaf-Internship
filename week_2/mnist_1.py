import torch
from torch import nn
import torch.nn.functional as F

from torch.utils import data
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(r'week_2\input', download=True, train=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True)

# Build a feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1),
)

# define the loss
criterion = nn.NLLLoss()

# get our data
dataiter = iter(trainloader)
images, labels = next(dataiter)

# flatten images
images = images.view(images.shape[0], -1)

# forward pass, get logits
logits = model(images)

# calculate loss with logits and labels
loss = criterion(logits, labels)

print(loss)
