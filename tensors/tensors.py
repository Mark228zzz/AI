import torch

# create a simple tensor
tensor1 = torch.tensor([1, 2, 3])
print(f'{tensor1 = }')

# operations with tensors
tensor2 = tensor1 + tensor1
print(f'{tensor2 = }')

tensor3 = tensor2 * tensor1
print(f'{tensor3 = }')

# create a matrix
matrix_tensor = torch.tensor([[1, 2], [3, 4]])
print(f'{matrix_tensor = }')

# zeros
zeros_tensor = torch.zeros((4, 2))
print(zeros_tensor)

# ones
ones_tensor = torch.ones((3, 3))
print(f'{ones_tensor = }')

# rand
rand_tensor = torch.rand((3, 2))
print(f'{rand_tensor = }')

# make a tensor using arange()
arange_tensor = torch.arange(9) # from 0 to 8
print(f'{arange_tensor = }')

# edit scale
arange_tensor = arange_tensor.view(3, 3)
print(f'{arange_tensor = }')

# Tensor concatenation
c_tensor = torch.cat((tensor1.view(1, 3), tensor2.view(1, 3)), dim=0)
print(f'{c_tensor = }')

# slices
tensor4 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f'{tensor4[1, 2]}') # Out: 6
print(f'{tensor4[:2, :2]}') # Out: [[1, 2], [4, 5]]
