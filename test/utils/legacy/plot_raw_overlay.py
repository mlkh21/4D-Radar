import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_calib(calib_file):
    R = np.eye(3)
    T = np.zeros(3)
    with open(calib_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            key = parts[0].strip()
            vals = [float(x) for x in parts[1].strip().split() if x.replace('.','',1).replace('-','',1).isdigit()]
            if key == 'R': R = np.array(vals).reshape(3, 3)
            elif key == 'T': T = np.array(vals)
    return R, T

r = np.load('Data/NTU4DRadLM_Raw/loop3/radar_pcl/1654232877.339592.npy')
l = np.load('Data/NTU4DRadLM_Raw/loop3/livox_lidar/1654232877.314979.npy')
R, T = load_calib('Data/config/calib_radar_to_livox.txt')
r_trans = r.copy()
r_trans[:, :3] = np.dot(r[:, :3], R.T) + T

plt.figure(figsize=(10,10))
plt.scatter(l[:, 0], l[:, 1], s=1, c='blue', alpha=0.5, label='LiDAR')
plt.scatter(r_trans[:, 0], r_trans[:, 1], s=2, c='red', alpha=0.8, label='Radar (transformed)')
plt.xlim(0, 120)
plt.ylim(-20, 20)
plt.legend()
plt.title("Overlay BEV (X-Y)")
plt.savefig("overlay_test.png")

print("Generated overlay_test.png")
