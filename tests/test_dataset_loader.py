"""
Unit tests for NTU4DRadLM dataset loader.
"""

import unittest
import os
import sys
import tempfile
import shutil
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset


class TestNTU4DRadLMVoxelDataset(unittest.TestCase):
    """Test cases for NTU4DRadLM_VoxelDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test scene structure
        scene_dir = os.path.join(self.test_dir, "test_scene")
        os.makedirs(scene_dir)
        
        radar_voxel_dir = os.path.join(scene_dir, "radar_voxel")
        target_voxel_dir = os.path.join(scene_dir, "target_voxel")
        os.makedirs(radar_voxel_dir)
        os.makedirs(target_voxel_dir)
        
        # Create dummy data files
        for i in range(5):
            # Create dummy voxel data (H, W, Z, C) = (32, 32, 16, 4)
            radar_data = np.random.randn(32, 32, 16, 4).astype(np.float32)
            target_data = np.random.randn(32, 32, 16, 4).astype(np.float32)
            
            np.save(os.path.join(radar_voxel_dir, f"frame_{i:04d}.npy"), radar_data)
            np.save(os.path.join(target_voxel_dir, f"frame_{i:04d}.npy"), target_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_dataset_initialization(self):
        """Test dataset can be initialized correctly."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir=self.test_dir,
            split='train'
        )
        
        self.assertEqual(len(dataset), 5)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir=self.test_dir,
            split='train'
        )
        
        target, radar = dataset[0]
        
        # Check shapes (C, Z, H, W)
        self.assertEqual(len(target.shape), 4)
        self.assertEqual(len(radar.shape), 4)
        self.assertEqual(target.shape[0], 4)  # 4 channels
        self.assertEqual(radar.shape[0], 4)   # 4 channels
        
        # Check types
        self.assertIsInstance(target, torch.Tensor)
        self.assertIsInstance(radar, torch.Tensor)
    
    def test_dataset_return_path(self):
        """Test dataset returns path when requested."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir=self.test_dir,
            split='train',
            return_path=True
        )
        
        target, radar, path = dataset[0]
        
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.npy'))
    
    def test_dataset_length(self):
        """Test dataset length is correct."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir=self.test_dir,
            split='train'
        )
        
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.__len__(), 5)
    
    def test_padding(self):
        """Test that voxels are padded correctly."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir=self.test_dir,
            split='train'
        )
        
        target, radar = dataset[0]
        
        # Check that all dimensions are multiples of 32
        for dim in range(1, 4):  # Check Z, H, W (skip C)
            self.assertEqual(target.shape[dim] % 32, 0)
            self.assertEqual(radar.shape[dim] % 32, 0)
    
    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        dataset = NTU4DRadLM_VoxelDataset(
            root_dir="/nonexistent/path",
            split='train'
        )
        
        self.assertEqual(len(dataset), 0)


class TestDatasetUtils(unittest.TestCase):
    """Test cases for dataset utility functions."""
    
    def test_import_modules(self):
        """Test that all required modules can be imported."""
        try:
            from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset
            from diffusion_consistency_radar.cm.radarloader_NTU4DRadLM_benchmark import load_data_NTU4DRadLM
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")


if __name__ == '__main__':
    unittest.main()
