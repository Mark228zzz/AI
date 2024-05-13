from torch import nn


# create a simple module
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 3) # linear layer from 3 inputs to 3 outputs

    def forward(self, x):
        x = self.fc1.relu(self.fc1(x)) # applying ReLU activation function
        return x


# create an instance of the model
module = SimpleNet()
print(module)
