import os
import torch
from torch import nn
from torch import optim


class Percepron(nn.Module):
    def __init__(self):
        super(Percepron, self).__init__()
        self.linear = nn.Linear(2, 1) # linear layer from 2 inputs to 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x)) # Sigmoid activation for binary classification


# create an instance of the model
model = Percepron()

# criterion and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # lr = learn rate

# example data (logical OR)
x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_train = torch.tensor([[0.], [1.], [1.], [1.]])

# learning cycle
for epoch in range(500_000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# module testing
model.eval()

# create data for testing
data = [
    torch.randint(0, 2, (1, 2)).float() for _ in range(25)
]

for test_input in data:
    predicted = model(test_input)
    print(f'Predicted value for input {test_input} is {predicted.item():.0f}')

# save the module
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

torch.save(model, f'{script_dir}/modules/trained_perceptron.pth')
