import torch
import torch.nn.functional as F
import numpy as np

# Simulate sparse target voxel (C=4, Z=80, H=600, W=200)
# Wait! In dataset_loader.py: 
# target_tensor = torch.from_numpy(target_voxel).permute(3, 2, 0, 1)
# target_voxel is (nx, ny, nz, 4) -> (600, 200, 80, 4)
# permute(3, 2, 0, 1) -> (4, 80, 600, 200) => (C, Z, H, W) where H=600, W=200

target = torch.zeros(1, 4, 80, 600, 200)
# put some 1.0s sparsely
target[0, 0, 40, 300, 100] = 1.0
target[0, 0, 40, 301, 100] = 1.0

target_size = (32, 128, 128)
target_interp = F.interpolate(target, size=target_size, mode='trilinear', align_corners=False)

print("Original max:", target[0, 0].max().item())
print("Interpolated max:", target_interp[0, 0].max().item())
