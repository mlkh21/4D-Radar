"""统一配置管理"""
from dataclasses import dataclass, field
from typing import List, Tuple
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    image_size: int = 64
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    in_ch: int = 4
    out_ch: int = 3
    dims: int = 3

@dataclass
class DataConfig:
    voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    grid_size: Tuple[int, int] = (500, 500)
    max_range: float = 16.0
    batch_size: int = 4
    num_workers: int = 4

@dataclass
class DiffusionConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    weight_schedule: str = "karras"

@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    total_training_steps: int = 600000

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get('model', {})),
            data=DataConfig(**data.get('data', {})),
            diffusion=DiffusionConfig(**data.get('diffusion', {})),
            training=TrainingConfig(**data.get('training', {})),
        )
