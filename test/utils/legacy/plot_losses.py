import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 从 legacy 子目录向上解析项目根目录，避免依赖当前工作目录。
    base_dir = Path(__file__).resolve().parents[3]
    
    vae_path = base_dir / 'test/result/vae/reconstruction/vae_near40_500_v2/vae/metrics.csv'
    ldm_path = base_dir / 'test/result/archive/ldm_sensor_aware_partial_20260713/ldm/metrics.csv'
    cd_path = base_dir / 'test/mini-test/train_results_near40_loop3/cd/metrics.csv'

    # 读取 CSV 数据
    try:
        vae_df = pd.read_csv(vae_path)
        ldm_df = pd.read_csv(ldm_path)
        cd_df = pd.read_csv(cd_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    # 创建一行三列的图表组合
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. VAE Recon Loss
    axs[0].plot(vae_df['epoch'], vae_df['recon_loss'], label='Recon Loss', color='royalblue')
    axs[0].set_title('VAE Reconstruction Loss', fontsize=14)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('Loss', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    # 2. LDM Loss
    axs[1].plot(ldm_df['epoch'], ldm_df['loss'], label='LDM Loss', color='forestgreen')
    axs[1].set_title('LDM Denoising Loss', fontsize=14)
    axs[1].set_xlabel('Epoch', fontsize=12)
    axs[1].set_ylabel('Loss', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    # 3. CD Loss
    axs[2].plot(cd_df['epoch'], cd_df['loss'], label='CD Loss', color='crimson')
    axs[2].set_title('CD Distillation Loss', fontsize=14)
    axs[2].set_xlabel('Epoch', fontsize=12)
    axs[2].set_ylabel('Loss', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()

    # 优化排版并保存
    plt.tight_layout()
    out_path = base_dir / "test" / "result" / "ldm" / "visualization" / "training_loss_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated and saved plot to: {out_path}")

if __name__ == "__main__":
    main()
