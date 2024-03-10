import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Check if a GPU is available and select it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create two very large matrices
matrix_size = 41000  # Adjust this value based on your GPU's memory capacity
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Continuously multiply the matrices to utilize the GPU
while True:
    c = torch.matmul(a, b)
    # print('Matrix multiplication performed')
