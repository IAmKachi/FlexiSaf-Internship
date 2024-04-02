# create neural networks with pytorch
import torch


def activation(x):
    """Sigmoid activation function"""
    return 1 / ( 1 + torch.exp(-x))


torch.manual_seed(7)

# features = torch.randn((1, 5))
# weights = torch.randn_like(features)
# bias = torch.randn((1, 1))

# # output = activation(torch.sum(features * weights) + bias)
# output = activation(torch.mm(features, weights.view((5, 1))) + bias)

features = torch.randn((1, 3))

# define the size of each layer in our network
n_input = features.shape[1]
n_hidden = 2
n_output = 1

# weights for input to hidden layer
W1 = torch.randn(n_input, n_hidden)
# weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

hidden_outputs = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(hidden_outputs, W2) + B2)

print(output)