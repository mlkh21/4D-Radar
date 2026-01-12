import subprocess
import sys
import os

# 使用清华源加速下载，可换成其他源，如阿里云: https://mirrors.aliyun.com/pypi/simple/
PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

def install_package(package):
    """通过 pip 安装包"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package,
            "-i", PIP_INDEX_URL
        ])
        print(f"安装成功 {package}")
    except subprocess.CalledProcessError as e:
        print(f"安装失败 {package}. 错误: {e}")
        return False
    return True

def main():
    print("开始配置和安装")

    # 确保 pip 是最新的
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip",
            "-i", PIP_INDEX_URL
        ])
    except Exception as e:
        print(f"警告：无法升级 pip： {e}")

    # 核心依赖
    core_packages = [
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
        "pyyaml>=6.0",
        "easydict>=1.9",
        "tqdm>=4.62.0",
        "open3d>=0.15.0",
        "opencv-python>=4.5.0"
    ]

    # 感知丧失
    loss_packages = [
        "piq>=0.7.0"
    ]

    # 分布式训练
    dist_packages = [
        "mpi4py>=3.1.0"
    ]

    # 开发依赖
    dev_packages = [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.960"
    ]

    all_packages = core_packages + loss_packages + dist_packages + dev_packages

    failed_packages = []
    
    for package in all_packages:
        if not install_package(package):
            failed_packages.append(package)

    if failed_packages:
        print("\n" + "="*50)
        print("安装完成但出现错误")
        print("以下软件包安装失败：")
        for pkg in failed_packages:
            print(f"- {pkg}")
        print("请检查您的网络连接或系统依赖性")
        print("对于 mpi4py，请确保安装了 MPI（例如，Ubuntu 上的“sudo apt install libopenmpi dev”）")
        print("="*50)
        sys.exit(1)
    else:
        print("\n" + "="*50)
        print("所有依赖项均已成功安装！")
        print("="*50)
        sys.exit(0)

if __name__ == "__main__":
    main()
