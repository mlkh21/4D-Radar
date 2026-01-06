# -*- coding: utf-8 -*-

# 此脚本用于评估EDM模型推理结果的点云质量
# 计算Chamfer距离、Hausdorff距离、F-score等指标
# 评估点云之间的匹配情况

import os
import numpy as np
import math
from scipy.spatial import cKDTree

import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate radar point cloud predictions')
    parser.add_argument('--pred_path', type=str, required=True, help='预测点云路径')
    parser.add_argument('--gt_path', type=str, required=True, help='真实点云路径')
    parser.add_argument('--output_path', type=str, default='./eval_results.json', help='输出路径')
    return parser.parse_args()

def main():
    args = parse_args()

    # pred_pc_list, gt_pc_list = read_inference_data(args.pred_path, args.gt_path)

    metrics = {
        'chamfer_distance': [],
        'hausdorff_distance': [],
        'precision': [],
        'recall': [],
        'fscore': [],
    }
    
    # Placeholder for reading data logic since read_inference_data is not defined in the snippet
    # Assuming read_inference_data exists or logic is similar to before but using args
    pred_pc_list = [] # populate this
    gt_pc_list = [] # populate this
    
    # Original logic adapted (commented out to avoid errors since read_inference_data is missing context)
    # But essentially replacing the hardcoded paths with args.pred_path and args.gt_path
    
    print(f"Evaluation started with Pred: {args.pred_path}, GT: {args.gt_path}")
    
    # for i in tqdm(range(len(pred_pc_list)), desc="Evaluating"):
    #     pred_pc_i = pred_pc_list[i]
    #     gt_pc_i = gt_pc_list[i]


    # 计算Chamfer距离
    # 输入：GT: 真实点云，形状为 (N, 3)
    #      Pred: 预测点云，形状为 (M, 3)
    # 输出：chamfer_dist: Chamfer距离，即两点云之间的平均距离之和
    #       distance_GT_to_Pred: 真实点云到预测点云的距离
    #       distance_Pred_to_GT: 预测点云到真实点云的距离

    kdtree_GT = cKDTree(GT)
    kdtree_Pred = cKDTree(Pred)

    distance_Pred_to_GT, _ = kdtree_GT.query(Pred)
    distance_GT_to_Pred, _ = kdtree_Pred.query(GT)

    chamfer_dist = np.mean(distance_GT_to_Pred) + np.mean(distance_Pred_to_GT) 

    return  chamfer_dist, distance_GT_to_Pred, distance_Pred_to_GT

def evaluate_matches(distance_A_to_B, distance_B_to_A, threshold):

    # 评估点云匹配情况
    # 输入：distance_A_to_B: 真实点云到预测点云的距离
    #      distance_B_to_A: 预测点云到真实点云的距离
    #      threshold: 距离阈值
    # 输出：TP: True Positive数量
    #       FN: False Negative数量
    #       FP: False Positive数量
    #       TN: True Negative数量
    #       precision: 精确率
    #       recall: 召回率

    TP = np.sum(distance_B_to_A <= threshold) # True Positive: 预测点云中距离真实点云小于阈值的点数
    FP = np.sum(distance_B_to_A > threshold) # False Positive: 预测点云中距离真实点云大于阈值的点数
    FN = np.sum(distance_A_to_B > threshold) # False Negative: 真实点云中距离预测点云大于阈值的点数
    TN = 0 # True Negative: 在点云评估中通常不计算TN，因为没有明确的负类定义

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return TP, FN, FP, TN, precision, recall

def read_inference_data(pcl_np_pred_path, pcl_np_gt_path):

    # 读取推理结果点云数据
    # 输入：pcl_np_pred_path: 预测点云的相对路径
    #      pcl_np_gt_path: 真实点云的相对路径
    # 输出：pred_pc_list: 预测点云列表
    #       gt_pc_list: 真实点云列表    

    pred_pc_list = []
    gt_pc_list = []

    for dirpath, dirnames, filenames in os.walk(BASE_PATH):
        for dirname in dirnames:
            if SCENE_NAME in dirname:
                subdir = os.path.join(dirpath, dirname)
                file = os.listdir(subdir + pcl_np_pred_path)
                file.sort(key=lambda x:int(x.split('.')[0]))
                for i in file:
                    path = os.path.join(subdir + pcl_np_pred_path, i)
                    pred_pc = np.load(path)
                    pred_pc_list.append(pred_pc)

                file = os.listdir(subdir + pcl_np_gt_path)
                file.sort(key=lambda x:int(x.split('.')[0]))
                for i in file:
                    path = os.path.join(subdir + pcl_np_gt_path, i)
                    gt_pc = np.load(path)
                    gt_pc_list.append(gt_pc)    

    return pred_pc_list, gt_pc_list
    
def main():

    # 主函数，执行评估流程

    pcl_np_pred_path =  "/pre_pcl_np/"
    pcl_np_gt_path = "/gt_bev_pcl/"
    
    pred_pc_list, gt_pc_list = read_inference_data(pcl_np_pred_path, pcl_np_gt_path)
    
    Chamfer_distance, Hausdorff_distance, prediction, recall, fscore = 0, 0, 0, 0, 0

    for i in range(len(pred_pc_list)):
        print("i", i)
        pred_pc_i = pred_pc_list[i]
        gt_pc_i = gt_pc_list[i]


        if pred_pc_i.shape[0] == 0 or gt_pc_i.shape[0] == 0:
            continue

        Chamfer_distance_i, distance_gt_to_pred, distance_pred_to_gt = cumpute_chamfer_distance(gt_pc_i, pred_pc_i)
        Chamfer_distance = Chamfer_distance + Chamfer_distance_i

        Hausdorff_distance = Hausdorff_distance + np.maximum(np.max(distance_gt_to_pred), np.max(distance_pred_to_gt))

        TP_i, FN_i, FP_i, TN_i, precision_i, recall_i = evaluate_matches(distance_gt_to_pred, distance_pred_to_gt, threshold = DISTANCE_Threshold)
        prediction = prediction + precision_i
        recall = recall + recall_i

        if (recall_i + precision_i) > 0:
            fscore = fscore + 2*precision_i*recall_i / (recall_i + precision_i)


    print("Chamfer_distance", Chamfer_distance/len(pred_pc_list))
    print("Hausdorff_distance", Hausdorff_distance/len(pred_pc_list))
    print("F-score", fscore/len(pred_pc_list))        

if __name__ == "__main__":
    main()
