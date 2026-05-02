import numpy as np

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

def transform(pcl, R, T):
    xyz = pcl[:, :3]
    xyz_trans = np.dot(xyz, R.T) + T
    pcl_trans = pcl.copy()
    pcl_trans[:, :3] = xyz_trans
    return pcl_trans

r = np.load('Data/NTU4DRadLM_Raw/loop3/radar_pcl/1654232877.339592.npy')
l = np.load('Data/NTU4DRadLM_Raw/loop3/livox_lidar/1654232877.314979.npy')
R, T = load_calib('Data/config/calib_radar_to_livox.txt')
r_trans = transform(r, R, T)

keep_r = (r_trans[:,0]>=0) & (r_trans[:,0]<120) & (r_trans[:,1]>=-20) & (r_trans[:,1]<20) & (r_trans[:,2]>=-6) & (r_trans[:,2]<10)
keep_l = (l[:,0]>=0) & (l[:,0]<120) & (l[:,1]>=-20) & (l[:,1]<20) & (l[:,2]>=-6) & (l[:,2]<10)

r_f = r_trans[keep_r]
l_f = l[keep_l]

print("R_f mean:", r_f.mean(axis=0)[:3])
print("L_f mean:", l_f.mean(axis=0)[:3])
