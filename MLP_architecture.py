"""
This script contains the definition of the MLP model in PyTorch framework.

# Network architecture:
    - 4 layers in total. (3 hidden layers + 1 output layer)
    - Hidden layer activation function: ReLU
    - Hidden layer sizes: 500, 250, 100
    - Output layer activation function: Softmax
    - Output layer size: 10

# Network output - class probabilities.
"""

import torch.nn as nn


# Define MLP model
class MLP(nn.Module):
    # Constructor - defines model layers
    def __init__(self):
        # Call parent constructor (mandatory for pytorch compatibility)
        super(MLP, self).__init__()

        # Model layers

        # Flatten layer - converts 3D input to 1D (input processing)
        # Input: (batch_size, 32, 32, 3) --> Output: (batch_size, 3072)
        self.flatten = nn.Flatten()

        # Fully connected layer 1
        # Input: (batch_size, 3072) --> Output: (batch_size, 500)
        self.fc1 = nn.Linear(32 * 32 * 3, 500)
        # ReLU activation function
        self.relu1 = nn.ReLU()

        # Fully connected layer 2
        # Input: (batch_size, 500) --> Output: (batch_size, 250)
        self.fc2 = nn.Linear(500, 250)
        # ReLU activation function
        self.relu2 = nn.ReLU()

        # Fully connected layer 3
        # Input: (batch_size, 250) --> Output: (batch_size, 100)
        self.fc3 = nn.Linear(250, 100)
        # ReLU activation function
        self.relu3 = nn.ReLU()

        # Output layer (fully connected)
        # Input: (batch_size, 100) --> Output: (batch_size, 10)
        self.output = nn.Linear(100, 10)
        # Softmax activation function
        # dim=1 applies softmax across classes for each sample in the batch
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass
        # Argument x: input tensor to the MLP (batch_size, 3, 32, 32)

        # Input layer (flatten)
        x = self.flatten(x)  # --> (batch_size, 3072)

        # Hidden layers
        # Fully connected layer 1
        x = self.relu1(self.fc1(x))  # --> (batch_size, 500)
        # Fully connected layer 2
        x = self.relu2(self.fc2(x))  # --> (batch_size, 250)
        # Fully connected layer 3
        x = self.relu3(self.fc3(x))  # --> (batch_size, 100)

        # Output layer
        x = self.softmax(self.output(x))  # --> (batch_size, 10)

        return x
