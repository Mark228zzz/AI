import torch

# create tensors with requires_grad=True for automatic differentiation
tensor_grad1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor_grad2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f'{tensor_grad1 = }')
print(f'{tensor_grad2 = }')

# operations
tensor_grad3 = tensor_grad1 * tensor_grad2 + tensor_grad2**2
print(f'{tensor_grad3 = }')

# sum
summa = tensor_grad3.sum()
print(f'Sum of tensor_grad3: {summa}')

# automatic calculation of gradients
summa.backward()

# gradient of tensor_grad1
print(f'{tensor_grad1.grad}')

# gradient of tensor_grad2
print(f'{tensor_grad2.grad}')
