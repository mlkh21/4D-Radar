import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 获取项目的根目录 (假定本脚本放在 test 文件夹下)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    vae_path = os.path.join(base_dir, 'Result/train_results/vae/metrics.csv')
    ldm_path = os.path.join(base_dir, 'Result/train_results/ldm/metrics.csv')
    cd_path = os.path.join(base_dir, 'Result/train_results/cd/metrics.csv')

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
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_loss_curves.png')
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated and saved plot to: {out_path}")

if __name__ == "__main__":
    main()
