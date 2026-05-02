#!/usr/bin/env python
import torch
from models.velocity_architectures.unet import UNet

model = UNet(in_channels=1, out_channels=1, c=32)
model.eval()

# Test with random input
x = torch.randn(2, 1, 28, 28)
t = torch.randn(2, 1)

try:
    with torch.no_grad():
        output = model(x, t)
    print(f'Success! Output shape: {output.shape}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
