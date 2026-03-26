# -*- coding: utf-8 -*-

# NOTE: 此脚本用于评估EDM模型推理结果的点云质量
# NOTE: 计算Chamfer距离、Hausdorff距离、F-score等指标
# NOTE: 评估点云之间的匹配情况

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

    # FIXME: 这里仍是评估脚手架，尚未接入实际的输入读取与指标汇总流程。

    metrics = {
        'chamfer_distance': [],
        'hausdorff_distance': [],
        'precision': [],
        'recall': [],
        'fscore': [],
    }
    
    # TODO: 将 pred_path / gt_path 读入逻辑接入为实际实现。
    pred_pc_list = []  # TODO: 填充预测点云列表
    gt_pc_list = []  # TODO: 填充真值点云列表
    
    print(f"Evaluation started with Pred: {args.pred_path}, GT: {args.gt_path}")
    
    # NOTE: 评估主循环应按索引对齐遍历 pred_pc_list 与 gt_pc_list。


    # NOTE: 计算Chamfer距离
    # NOTE: 输入：GT: 真实点云，形状为 (N, 3)
    # NOTE: 预测点云，形状为 (M, 3)
    # NOTE: 输出：chamfer_dist: Chamfer距离，即两点云之间的平均距离之和
    # NOTE: 真值到预测距离 distance_GT_to_Pred: 真值点云到预测点云的最近邻距离
    # NOTE: 预测到真值距离 distance_Pred_to_GT: 预测点云到真值点云的最近邻距离

    kdtree_GT = cKDTree(GT)
    kdtree_Pred = cKDTree(Pred)

    distance_Pred_to_GT, _ = kdtree_GT.query(Pred)
    distance_GT_to_Pred, _ = kdtree_Pred.query(GT)

    chamfer_dist = np.mean(distance_GT_to_Pred) + np.mean(distance_Pred_to_GT) 

    return  chamfer_dist, distance_GT_to_Pred, distance_Pred_to_GT

def evaluate_matches(distance_A_to_B, distance_B_to_A, threshold):

    # NOTE: 评估点云匹配情况
    # NOTE: 输入：distance_A_to_B: 真实点云到预测点云的距离
    # NOTE: 预测到真值距离 distance_B_to_A
    # NOTE: 判定匹配阈值 threshold
    # NOTE: 输出：真正例（TP）数量
    # NOTE: 漏检点（FN）数量
    # NOTE: 误检点（FP）数量
    # NOTE: 真负例（TN）在点云评估中通常不定义
    # NOTE: 精确率 precision
    # NOTE: 召回率 recall

    TP = np.sum(distance_B_to_A <= threshold) # True Positive: 预测点云中距离真实点云小于阈值的点数
    FP = np.sum(distance_B_to_A > threshold) # False Positive: 预测点云中距离真实点云大于阈值的点数
    FN = np.sum(distance_A_to_B > threshold) # False Negative: 真实点云中距离预测点云大于阈值的点数
    TN = 0 # True Negative: 在点云评估中通常不计算TN，因为没有明确的负类定义

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return TP, FN, FP, TN, precision, recall

def read_inference_data(pcl_np_pred_path, pcl_np_gt_path):

    # NOTE: 读取推理结果点云数据
    # NOTE: 输入：pcl_np_pred_path: 预测点云的相对路径
    # NOTE: 真值点云相对路径 pcl_np_gt_path
    # NOTE: 输出：pred_pc_list: 预测点云列表
    # NOTE: 真值点云列表 gt_pc_list

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

    # NOTE: 主函数，执行评估流程

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
