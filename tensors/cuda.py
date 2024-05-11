import torch

# creating a tensor using your CPU
tensor_cpu1 = torch.tensor([1, 2, 3, 4, 5])
tensor_cpu2 = torch.rand((5, 5))

# creatnig a tensor using your GPU
if torch.cuda.is_available(): # check if CUDA is available
    tensor_gpu1 = torch.tensor([1, 2, 3, 4, 5], device='cuda')
    tensor_gpu2 = torch.rand((5, 5), device='cuda')
else:
    print('CUDA is not available!')
