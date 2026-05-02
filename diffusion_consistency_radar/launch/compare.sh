python /home/zxj/catkin_ws/src/4D-Radar-Diffusion/diffusion_consistency_radar/scripts/compare_radar_lidar_images.py \
--pred_pcl_dir /home/zxj/catkin_ws/src/4D-Radar-Diffusion/Result/inference_results/loop3_ldm_eval \
--raw_livox_dir /home/zxj/catkin_ws/src/4D-Radar-Diffusion/Data/NTU4DRadLM_Raw/loop3/livox_lidar \
--output_dir /home/zxj/catkin_ws/src/4D-Radar-Diffusion/Result/comparison_results/loop3 \
--max_files 1 \
--mode 3d \
--point_size 0.8