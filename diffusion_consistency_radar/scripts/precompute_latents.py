import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import VAE3D, create_ultra_lightweight_vae_config
from cm.dataset_loader import NTU4DRadLM_VoxelDataset

# 配置
VAE_CKPT = "./diffusion_consistency_radar/train_results/vae/vae_best.pt"
DATASET_DIR = "./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre"
OUTPUT_DIR = "./NTU4DRadLM_pre_processing/NTU4DRadLM_Latent"
BATCH_SIZE = 4  # 批处理大小，根据显存调整（8-16）
NUM_WORKERS = 8  # 数据加载线程数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载VAE
print("Loading VAE...")
config = create_ultra_lightweight_vae_config()
vae = VAE3D(**config)
ckpt = torch.load(VAE_CKPT, map_location='cpu')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    vae.load_state_dict(ckpt['model_state_dict'])
else:
    vae.load_state_dict(ckpt)
vae = vae.cuda().eval()

# 使用DataLoader批量加载
class LatentDataset(NTU4DRadLM_VoxelDataset):
    def __getitem__(self, idx):
        target, cond = super().__getitem__(idx)
        return target, cond, self.samples[idx][0]  # 返回路径用于保存

dataset = LatentDataset(root_dir=DATASET_DIR, split='train', use_augmentation=False)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

print(f"Encoding {len(dataset)} samples with batch_size={BATCH_SIZE}...")
for batch_idx, (targets, conds, paths) in enumerate(tqdm(dataloader)):
    targets = targets.cuda()
    conds = conds.cuda()
    
    with torch.no_grad():
        z_targets = vae.get_latent(targets)
        z_conds = vae.get_latent(conds)
    
    # 逐个保存（因为路径不同）
    for i in range(len(paths)):
        scene_name = paths[i].split('/')[-3]
        save_dir = os.path.join(OUTPUT_DIR, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(paths[i]).replace('.npz', '.pt')
        torch.save({
            'target': z_targets[i:i+1].cpu(),
            'cond': z_conds[i:i+1].cpu()
        }, os.path.join(save_dir, filename))
    
    # 定期清理
    if batch_idx % 50 == 0:
        torch.cuda.empty_cache()

print("Done!")