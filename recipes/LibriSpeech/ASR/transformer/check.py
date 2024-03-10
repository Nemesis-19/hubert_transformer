import torch
import os

# Set the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # This will make only GPU 1 visible to the script

# Alternatively, you can set the device directly in PyTorch without affecting the environment variable
device = torch.device("cuda:0")  # Note: The index is now 0 because CUDA_VISIBLE_DEVICES changes the visible devices

# Now, we demonstrate a simple operation that runs on GPU 1
# Create a tensor and transfer it to the GPU specified by the device
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)

# Perform a simple operation on the GPU
result = tensor * tensor

# Print the result
print(result)

# Note: Ensure that your machine has a GPU and CUDA is installed properly.
