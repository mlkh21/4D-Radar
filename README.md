# 4D Radar Diffusion Model

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.7+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A diffusion model implementation for 4D radar data processing using consistency models and EDM (Elucidated Diffusion Models).

## Overview

This project implements diffusion models for processing 4D radar point cloud data from the NTU4DRadLM dataset. It supports both:
- **Consistency Models (CM)**: For efficient radar data generation
- **EDM**: Elucidated Diffusion Models for high-quality radar enhancement

## Features

- 4D radar voxel processing with 3D U-Net architecture
- Conditional diffusion models for radar-to-lidar translation
- Support for training and inference pipelines
- Evaluation metrics (Chamfer Distance, Hausdorff Distance, F-score)
- Distributed training with PyTorch DDP

## Project Structure

```
4D-Radar-Diffusion/
├── diffusion_consistency_radar/
│   ├── cm/                          # Core model implementations
│   │   ├── dataset_loader.py        # Dataset loading utilities
│   │   ├── karras_diffusion.py      # Diffusion process implementation
│   │   ├── train_util_cond.py       # Training utilities
│   │   └── ...
│   ├── scripts/                     # Training and inference scripts
│   │   ├── cm_train_radar.py        # Consistency model training
│   │   ├── edm_train_radar.py       # EDM training
│   │   ├── image_sample_radar.py    # Inference script
│   │   └── evaluate.py              # Evaluation metrics
│   ├── launch/                      # Launch scripts
│   └── setup.py                     # Package setup
├── NTU4DRadLM_pre_processing/       # Data preprocessing
└── inspect_radar_data.py            # Data inspection tool
```

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mlkh21/4D-Radar-Diffusion.git
cd 4D-Radar-Diffusion
```

2. Install dependencies:
```bash
cd diffusion_consistency_radar
pip install -e .
```

Or install manually:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Organize your NTU4DRadLM dataset in the following structure:
```
NTU4DRadLM_Pre/
├── scene_001/
│   ├── radar_voxel/    # Input radar voxels
│   └── target_voxel/   # Target voxels (lidar)
├── scene_002/
│   └── ...
```

2. Update the dataset path in training scripts or use command-line arguments.

### Training

#### Consistency Model Training
```bash
cd diffusion_consistency_radar
bash launch/train_cd.sh
```

Or with custom parameters:
```bash
python scripts/cm_train_radar.py \
    --dataset_dir /path/to/NTU4DRadLM_Pre \
    --global_batch_size 4 \
    --out_dir ./train_results
```

#### EDM Training
```bash
bash launch/train_edm.sh
```

### Inference

```bash
# Consistency Model inference
bash launch/inference_cd.sh

# EDM inference
bash launch/inference_edm.sh
```

### Evaluation

```bash
python scripts/evaluate.py
```

## Configuration

Key training parameters:
- `--global_batch_size`: Global batch size across all GPUs
- `--image_size`: Input voxel size (default: 64)
- `--num_channels`: Model channel count (default: 128)
- `--use_fp16`: Enable mixed precision training
- `--lr`: Learning rate
- `--ema_rate`: Exponential moving average rate

## Data Format

### Input (Radar Voxel)
- Shape: `(H, W, Z, 4)` where channels are `[Occupancy, Intensity, Doppler, Variance]`
- Stored as `.npy` files

### Output (Target Voxel)
- Shape: `(H, W, Z, 4)` where channels are `[Occupancy, Intensity, Doppler, Mask]`
- Stored as `.npy` files

## Evaluation Metrics

The evaluation script computes:
- **Chamfer Distance**: Measures point cloud similarity
- **Hausdorff Distance**: Maximum distance between point sets
- **F-score**: Precision-recall metric at specified threshold

## Citation

If you use this code in your research, please cite the relevant papers.

## Code Quality

This project follows best practices for research code:
- ✅ Type hints for better code clarity
- ✅ Comprehensive error handling
- ✅ Logging instead of print statements
- ✅ Unit tests for critical components
- ✅ Example scripts for quick start
- ✅ Configuration templates

See [CODE_REVIEW_SUMMARY.md](CODE_REVIEW_SUMMARY.md) for detailed code review results.

## License

Please check the original repository for license information.

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--global_batch_size` or `--microbatch`
2. **Dataset not found**: Update paths in configuration files
3. **Import errors**: Ensure all dependencies are installed

## Contact

For questions or issues, please open an issue on GitHub.
