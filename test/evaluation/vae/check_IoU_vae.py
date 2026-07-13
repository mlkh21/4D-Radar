import os
import sys
import csv
import torch
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CM_ROOT = PROJECT_ROOT / "diffusion_consistency_radar"
sys.path.insert(0, str(CM_ROOT))
from cm.vae_3d import (

    VAE3D,

    create_ultra_lightweight_vae_config,

    create_lightweight_vae_config,

    create_standard_vae_config,

)
from cm.dataset_loader import load_sparse_voxel, resize_voxel_channels





vae_ckpt = os.environ.get(
    "VAE_CKPT",
    str(PROJECT_ROOT / "test" / "mini-test" / "train_results_mini_near" / "vae" / "vae_best.pt"),
)

data_root = os.environ.get(
    "DATA_ROOT", str(PROJECT_ROOT / "Data" / "NTU4DRadLM_Pre_near40_soft")
)

scene = os.environ.get("SCENE", "garden")

max_files = int(os.environ.get("MAX_FILES", "500"))

thresholds = [float(x) for x in os.environ.get("THRESHOLDS", "0.1 0.2 0.3 0.4 0.5").split()]

config_type = os.environ.get("VAE_CONFIG_TYPE", "ultra_lightweight")

target_size = tuple(int(x) for x in os.environ.get("TARGET_SIZE", "32,128,128").split(","))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if config_type == "ultra_lightweight":

    vae_config = create_ultra_lightweight_vae_config()

elif config_type == "lightweight":

    vae_config = create_lightweight_vae_config()

elif config_type == "standard":

    vae_config = create_standard_vae_config()

else:

    raise ValueError(f"Unknown VAE_CONFIG_TYPE: {config_type}")



model = VAE3D(**vae_config).to(device)



ckpt = torch.load(vae_ckpt, map_location=device)

state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

model.load_state_dict(state)

model.eval()



target_dir = os.path.join(data_root, scene, "target_voxel")

files = sorted(

    f for f in os.listdir(target_dir)

    if f.endswith(".npz") or f.endswith(".npy")

)

files = files[:max_files]



if not files:

    raise RuntimeError(f"No target voxel files found in: {target_dir}")



stats = {

    thr: {

        "intersection": 0,

        "union": 0,

        "gt_occ": 0,

        "pred_occ": 0,

        "frames": 0,

    }

    for thr in thresholds

}



rows = []



with torch.no_grad():

    for name in files:

        path = os.path.join(target_dir, name)



        if path.endswith(".npz"):

            voxel = load_sparse_voxel(path)

        else:

            voxel = np.load(path).astype(np.float32)



        # 原始 voxel: (X,Y,Z,C) -> 训练张量: (C,Z,X,Y)

        x = torch.from_numpy(voxel).permute(3, 2, 0, 1)

        x = resize_voxel_channels(x, target_size, mask_channel=3)

        x = x.unsqueeze(0).to(device)



        z, _ = model.encode(x, deterministic=True)

        recon = model.decode(z)



        gt = x[:, 0] > 0.5



        for thr in thresholds:

            pred = recon[:, 0] > thr



            inter = int((gt & pred).sum().item())

            union = int((gt | pred).sum().item())

            gt_count = int(gt.sum().item())

            pred_count = int(pred.sum().item())



            iou = inter / max(union, 1)

            recall = inter / max(gt_count, 1)

            precision = inter / max(pred_count, 1)



            stats[thr]["intersection"] += inter

            stats[thr]["union"] += union

            stats[thr]["gt_occ"] += gt_count

            stats[thr]["pred_occ"] += pred_count

            stats[thr]["frames"] += 1



            rows.append({

                "frame": os.path.splitext(name)[0],

                "threshold": thr,

                "gt_occ": gt_count,

                "recon_occ": pred_count,

                "intersection": inter,

                "union": union,

                "iou": iou,

                "recall": recall,

                "precision": precision,

            })



out_dir = os.path.join("test", "result", "vae", "evaluation", "vae_iou_eval", scene)

os.makedirs(out_dir, exist_ok=True)



csv_path = os.path.join(out_dir, "vae_reconstruction_iou_per_frame.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

    writer.writeheader()

    writer.writerows(rows)



summary_path = os.path.join(out_dir, "vae_reconstruction_iou_summary.csv")

with open(summary_path, "w", newline="", encoding="utf-8") as f:

    fieldnames = [

        "threshold",

        "frames",

        "mean_iou_global",

        "mean_recall_global",

        "mean_precision_global",

        "gt_occ_total",

        "recon_occ_total",

        "count_ratio",

    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()



    print("\nVAE reconstruction IoU summary")

    print(f"checkpoint: {vae_ckpt}")

    print(f"data: {target_dir}")

    print(f"frames: {len(files)}")

    print("-" * 80)



    for thr in thresholds:

        s = stats[thr]

        iou = s["intersection"] / max(s["union"], 1)

        recall = s["intersection"] / max(s["gt_occ"], 1)

        precision = s["intersection"] / max(s["pred_occ"], 1)

        ratio = s["pred_occ"] / max(s["gt_occ"], 1)



        row = {

            "threshold": thr,

            "frames": s["frames"],

            "mean_iou_global": iou,

            "mean_recall_global": recall,

            "mean_precision_global": precision,

            "gt_occ_total": s["gt_occ"],

            "recon_occ_total": s["pred_occ"],

            "count_ratio": ratio,

        }

        writer.writerow(row)



        print(

            f"thr={thr:.2f} | "

            f"IoU={iou:.4f} | "

            f"Recall={recall:.4f} | "

            f"Precision={precision:.4f} | "

            f"CountRatio={ratio:.4f}"

        )



print("-" * 80)

print(f"Saved per-frame CSV: {csv_path}")

print(f"Saved summary CSV:   {summary_path}")
