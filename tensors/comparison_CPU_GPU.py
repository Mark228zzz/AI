import torch
from time import time

size = (10_000, 10_000)

def main() -> None:
    start_time_cpu = time()
    tensor_cpu = torch.rand(size)
    end_time_cpu = time()

    print(f'Tensor was create on the CPU in {end_time_cpu-start_time_cpu:.2f}s')

    if not torch.cuda.is_available(): raise "CUDA is not available!"

    start_time_gpu = time()
    tensor_gpu = torch.rand(size, device='cuda')
    end_time_gpu = time()

    print(f'Tensor was create on the GPU in {end_time_gpu-start_time_gpu:.2f}s')

if __name__ == '__main__':
    main()
