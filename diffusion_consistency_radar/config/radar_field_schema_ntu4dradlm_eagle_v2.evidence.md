<!-- 文件功能：记录 NTU4DRadLM Oculii Eagle 原始 Radar 字段合同的公开来源、经验交叉检查和适用边界。 -->
# NTU4DRadLM Oculii Eagle 字段合同证据

## 适用对象

- 数据集：NTU4DRadLM。
- ROS topic：`/radar_pcl`。
- 消息类型：`sensor_msgs/PointCloud`。
- 本证据不适用于 `/radar_enhanced_pcl`、其他 Eagle 固件/驱动或其他数据集。

## 设备资料

1. Oculii Corporation，`EAGLE CAN FD USER MANUAL`，Edition 0.6.1，2021-09：
   <https://fccid.io/2AXVNEAGLE/User-Manual/User-Manual-6126195.pdf>
   - 第 5 页给出最大速度范围 `-86.8 m/s` 到 `+86.8 m/s`。
   - 第 13--14 页说明传感器数据相对于传感器本体，而不是车辆中心。
   - 第 23--24 页规定 Doppler 解码单位为 `m/s`，Power 是以 `dB` 表示的检测信噪比，并给出 Range/Alpha/Beta 到 XYZ 的关系。

2. Oculii Corporation，`EAGLE G7 USER MANUAL`，Edition 0.5.41，2021-12：
   <https://snail-radar.github.io/docs/manuals/Manual%20Edition%200.5.41%20Dec2021%20Eagle%20G7.pdf>
   - 第 5 页再次给出 `-86.8 m/s` 到 `+86.8 m/s` 的最大速度范围。
   - 第 25 页再次规定 Doppler 为 `m/s`、Power 为 `dB` SNR。

## 数据集提供方资料

1. NTU4DRadLM 官方仓库：
   <https://github.com/junzhang2016/NTU4DRadLM>
   - 数据集将原始 Radar 标识为 Oculii Eagle，并声明 `/radar_pcl` 为 `sensor_msgs/PointCloud`。

2. NTU4DRadLM 配套 4DRadarSLAM：
   <https://github.com/zhuge2333/4DRadarSLAM>
   - README 明确说明预处理会把 Radar 点云变换到 Livox LiDAR frame。
   - `apps/preprocessing_nodelet.cpp` 的 `cloud_callback` 对 Eagle 消息逐点应用 `Radar_to_livox`，之后才把输出消息标记为 `base_link`；因此输入消息中的 `base_link` 不能解释为“XYZ 已经位于 LiDAR/车体 frame”。
   - `src/radar_ego_velocity_estimator.cpp` 将输入 `Power` 作为 `snr_db`，并在速度估计前对原始 Doppler 取负；结合单位方向向量方程，原始正 Doppler 表示距离增加，即 `away_from_sensor`。

3. NTU4DRadLM 论文：
   <https://doi.org/10.1109/ITSC57777.2023.10422606>
   - 论文将 Radar 设备标识为 Oculii Eagle，原始点字段包含 XYZ、Doppler 和 Power。

## 本地只读交叉检查

- 2026-09-03 全量只读报告：
  `test/result/comparison/alignment_check/raw_radar_contract_empirical_20260903/report.json`
- 报告 SHA-256：`e4d644f33bfb8117d241d227c02ba5ef340641bf5c24e3a92d31338e02912769`。
- garden/loop3 共 12554 个 Radar frame 均为 `/radar_pcl`、`sensor_msgs/PointCloud`、`header.frame_id=base_link`，channels 稳定为 `Doppler/Range/Power/Alpha/Beta`。
- 315641 个抽样点的 Power 为 `0.12--32.97`，与手册所述已解码 dB SNR 数值形态一致；Doppler 为 `-22.9104--8.7912 m/s`，样本极值只作数据分布检查，不用于反推设备量程。
- 对 garden 首个 Radar frame 复核：`abs(norm(xyz)-Range)` 的中位数约 `4.4e-7 m`、最大值约 `7.7e-6 m`；用手册的 Range/Alpha/Beta 公式和固定轴置换可复原消息 XYZ，中位点误差约 `0.0071 m`。
- 项目 Radar→Livox 标定 SHA-256：`e50426daf72ee69a7fe458f1f0caf9060d9c8e2a2548a13e03acdae329cbe08d`。

## 合同结论与边界

- `Power`：检测信噪比，单位 `dB`；输入值已经是解码后的浮点 dB，不再执行 `log1p`。
- `Doppler`：相对传感器的径向速度，单位 `m/s`，正值表示远离传感器。
- `86.8 m/s`：Eagle 标称最大速度范围的绝对上限，用作固定归一化尺度；它不是当前样本观测极值，也不是飞行器速度上限。
- XYZ：物理上位于 Radar 传感器 frame；ROS header 的 `base_link` 是该数据流的标签，不能替代 Radar→LiDAR 外参。
- 若后续发现数据集使用了与上述手册不同的固件解码、`/radar_pcl` 驱动执行了未公开坐标变换，或设备提供方给出相反的 Doppler 符号定义，必须升级 schema/数据协议并重新生成 normalization，不得原地改写本合同。
