import torch

# create a simple tensor
tensor1 = torch.tensor([1, 2, 3])
print(f'{tensor1 = }')

# operations with tensors
tensor2 = tensor1 + tensor1
print(f'{tensor2 = }')

tensor3 = tensor2 * tensor1
print(f'{tensor3 = }')
