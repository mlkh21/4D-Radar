#!/usr/bin/env python3
"""
Example: Data loading and inspection
This example shows how to load and inspect NTU4DRadLM dataset.
"""

import os
import sys
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

def main():
    """Main function for data loading example."""
    
    print("=" * 60)
    print("4D Radar Data Loading Example")
    print("=" * 60)
    
    # Dataset path
    dataset_path = "./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre"
    
    if not os.path.exists(dataset_path):
        print(f"\n⚠️  Dataset path not found: {dataset_path}")
        print("Please update the path to your NTU4DRadLM_Pre directory")
        return
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = NTU4DRadLM_VoxelDataset(
        root_dir=dataset_path,
        split='train',
        return_path=True
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("\n⚠️  No samples found in dataset")
        print("Please check:")
        print("  1. Dataset directory structure")
        print("  2. radar_voxel/ and target_voxel/ folders exist")
        print("  3. .npy files are present in both folders")
        return
    
    # Inspect first sample
    print("\n" + "=" * 60)
    print("Inspecting first sample")
    print("=" * 60)
    
    target, radar, path = dataset[0]
    
    print(f"\nFile path: {path}")
    print(f"\nTarget voxel:")
    print(f"  Shape: {target.shape}")
    print(f"  Dtype: {target.dtype}")
    print(f"  Range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"  Memory: {target.element_size() * target.nelement() / 1024 / 1024:.2f} MB")
    
    print(f"\nRadar voxel:")
    print(f"  Shape: {radar.shape}")
    print(f"  Dtype: {radar.dtype}")
    print(f"  Range: [{radar.min():.4f}, {radar.max():.4f}]")
    print(f"  Memory: {radar.element_size() * radar.nelement() / 1024 / 1024:.2f} MB")
    
    # Channel analysis
    print(f"\nChannel analysis (Target):")
    for i in range(target.shape[0]):
        channel_data = target[i]
        non_zero = (channel_data != 0).sum()
        print(f"  Channel {i}: Non-zero voxels: {non_zero} / {channel_data.numel()}")
    
    print(f"\nChannel analysis (Radar):")
    for i in range(radar.shape[0]):
        channel_data = radar[i]
        non_zero = (channel_data != 0).sum()
        print(f"  Channel {i}: Non-zero voxels: {non_zero} / {channel_data.numel()}")
    
    # Sample multiple items
    print("\n" + "=" * 60)
    print("Sampling multiple items")
    print("=" * 60)
    
    sample_size = min(5, len(dataset))
    print(f"\nSampling {sample_size} items...")
    
    for i in range(sample_size):
        target, radar, path = dataset[i]
        print(f"  Sample {i}: Target {target.shape}, Radar {radar.shape}")
    
    print("\n✓ Data loading example completed successfully!")
    print("\nNext steps:")
    print("  1. Visualize the voxels using Open3D or similar tools")
    print("  2. Train a model using the training scripts")
    print("  3. Evaluate model performance on test set")

if __name__ == "__main__":
    main()
