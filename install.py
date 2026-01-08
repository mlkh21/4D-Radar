import subprocess
import sys
import os

def install_package(package):
    """Install a package via pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
        return False
    return True

def main():
    print("Starting configuration and installation...")

    # Ensure pip is up to date
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except Exception as e:
        print(f"Warning: Could not upgrade pip: {e}")

    # Core dependencies
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

    # Perception loss
    loss_packages = [
        "piq>=0.7.0"
    ]

    # Distributed training
    dist_packages = [
        "mpi4py>=3.1.0"
    ]

    # Development dependencies
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
        print("Installation finished with errors.")
        print("The following packages failed to install:")
        for pkg in failed_packages:
            print(f"- {pkg}")
        print("Please check your network connection or system dependencies.")
        print("For mpi4py, ensure you have MPI installed (e.g., 'sudo apt install libopenmpi-dev' on Ubuntu).")
        print("="*50)
        sys.exit(1)
    else:
        print("\n" + "="*50)
        print("All dependencies installed successfully!")
        print("="*50)
        sys.exit(0)

if __name__ == "__main__":
    main()
