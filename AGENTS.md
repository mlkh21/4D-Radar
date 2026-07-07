# 项目规则

本项目是 4D Radar / LiDAR / infrared 多模态 3D Occupancy 生成项目。

- 任务：基于机载传感器融合的障碍物地图构建与场景地图更新
  考虑机载传感器噪声大，点云稀疏等问题，开展4D毫米波雷达点云融合滤波、红外与毫米波雷达点云稠密点云生成等方法研究；同时，考虑机载传感器误差和里程计误差，优化概率栅格障碍物地图表征方式；考虑机载飞行平台内存约束，设计实时感知障碍物地图与数字高程地图的融合更新方法。
  解释--参考论文P. R. Florence, J. Carter, J. Ware and R. Tedrake, "NanoMap: Fast, Uncertainty-Aware Proximity Queries with Lazy Search Over Local 3D Data," 2018 IEEE International Conference on Robotics and Automation (ICRA), Brisbane, QLD, Australia, 2018, pp. 7631-7638, doi: 10.1109/ICRA.2018.8463195.
  部分内容参考浙大高飞前期成果https://github.com/ZJU-FAST-Lab/Radar-Diffusion
  思路可以借鉴当前3D占用网络（ Occupancy Network）相关内容，如：MetaOcc: Surround-View 4D Radar and Camera Fusion Framework for 3D Occupancy Prediction with Dual Training Strategies，RadarOcc，LiCROcc等。
- 约束条件
  飞行速度35m/s-70m/s；
  飞行器动力学模拟基于JSBsim；
  整套仿真系统基于ros框架，以服务方式发布航迹点信息，以action方式定义所设计的控制器，与PX4实现硬在环仿真集成。

## 工作原则

- 默认使用中文回答。
- 代码中要写功能注释，代码注释默认使用中文。
- 新增文件中要写文件头注释，说明文件功能，默认使用中文。
- 修改代码前先阅读相关调用链，并说明准备修改的文件。
- 不要大范围重构，优先小步修改。
- 不要删除数据集、checkpoint、训练日志或实验结果。
- 不要自动运行长时间训练命令。
- 测试代码放入 `test/` 文件夹，测试结果放入 `test/result/` 文件夹。
- 运行测试前先说明测试范围。
- 优先写小测试验证修改是否正确，测试结束后清理测试数据，删除不必要的测试文件。
- 修改数据预处理、target 生成、模型结构或评估指标时，需要说明改动对监督信号、体素数量和指标结果的影响。
- 每次修改后在 `TODO/findings.md` 中记录修改内容和发现的问题，在 `TODO/task_plan.md` 中制定后续改进计划，在 `TODO/progress.md` 中记录进展，格式参照文件中原内容。

## 环境

Conda 环境名：

```bash
Radar-Diffusion
```

运行 Python 脚本优先使用：

```bash
conda run -n Radar-Diffusion python <script>
```

例如：

```bash
conda run -n Radar-Diffusion python test/test_sensor_aware_target.py
```
