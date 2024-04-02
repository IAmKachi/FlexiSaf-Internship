import torch
from torch import nn
import torch.nn.functional as F
from  torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import helper


# def activation(x):
#     """Sigmoid activation function"""
#     return 1 / ( 1 + torch.exp(-x))


# def softmax(x):
#     """softmax activation function"""
#     num = torch.exp(x)
#     denom = torch.sum(torch.exp(x), dim=1).view(x.shape[0], 1)

#     return num / denom

# define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# download and load the training data
trainset = datasets.MNIST(r'week_2\input', download=True, train=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim=1)

        return x


model = Network()
# print(model.fc1.weight)
# print(model.fc1.bias)

model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.01)

# resize images to 1D vector
# images = images.view(images.shape[0], -1)
images.resize_(64, 1, 784)

img_idx = 0
ps = model.forward(images[img_idx, :])

img = images[img_idx]
# plt.imshow(img.view(1, 28, 28).numpy().squeeze(), cmap='Greys_r')
# plt.show()
helper.view_classify(img, ps)
plt.show()

# print(type(images))
# # print(images[1])
# print(images.shape)
# print(labels.shape)

# # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
# # plt.show()

# inputs = images.flatten(1)
# n_hidden = 256
# n_outputs = 10

# input_weights = torch.randn(inputs.shape[1], n_hidden)
# hidden_weights = torch.randn(n_hidden, n_outputs)

# input_bias = torch.randn(1, n_hidden)
# output_bias = torch.randn(1, n_outputs)

# hidden_outputs = activation(torch.mm(inputs, input_weights) + input_bias)
# prob_outputs = softmax(torch.mm(hidden_outputs, hidden_weights) + output_bias)

# print(prob_outputs)
# print(prob_outputs.shape)
# print(prob_outputs.sum(dim=1))