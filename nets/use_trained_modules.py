import os
import torch


class Percepron(torch.nn.Module): # make sure this class exactly matches the class used when saving the model
    def __init__(self):
        super(Percepron, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def main():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # load the module
    model = torch.load(f'{script_dir}/modules/trained_perceptron.pth')
    model.eval()

    # use the module
    data = [
        torch.randint(0, 2, (1, 2)).float() for _ in range(25)
    ]

    for input_data in data:
        predicted = model(input_data)
        print(f'Predicted value for input {input_data} is {predicted.item():.0f}')

if __name__ == '__main__':
    main()
