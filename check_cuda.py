import torch
import torch.nn as nn

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the available CUDA devices
    cuda_devices = torch.cuda.device_count()
    
    # Check each CUDA device's compute capability
    for i in range(cuda_devices):
        device = torch.cuda.get_device_name(i)
        compute_capability = torch.cuda.get_device_capability(i)
        print(f"Device {i}: {device}, Compute Capability: {compute_capability}")
else:
    print("CUDA is not available on this system.")

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input.shape)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target.shape)
output = loss(input, target)
output.backward()