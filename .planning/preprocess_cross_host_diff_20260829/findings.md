<!-- 文件功能：记录跨主机预处理差异诊断证据。 -->
# 诊断发现

- 本机与服务器的 policy、核心 Python 脚本、标定和同步表哈希一致；Radar 两场景逐文件一致。
- `preprocess_script=cee103...` 是 `NTU4DRadLM_pre_processing.py`，不是当前服务器启动脚本 `preprocess.sh`。
- 本机 `preprocess-v2.sh` 复用 `Raw_p1_01_candidate`；服务器脚本先在 `Radar` 环境执行 `unpack_rosbag.py` 重新生成 Raw。
- `unpack_rosbag.py` 使用 OpenCV 解码压缩图像并再次写 PNG；核心预处理再灰度读取并从 512x640 缩放到 480x640，能够解释 IR 全帧受环境影响。
- Patchwork++ 头文件明确维护 `update_flatness_`、`update_elevation_`，阈值会自适应更新；核心脚本为每个 worker 创建一个实例并跨帧复用，任务则通过 `imap_unordered` 动态分派。
- 同一 garden 第 18 帧本机复现：fresh/after_0/after_0_to_17 的非地面点数分别为 16598/16561/16555；占用体素相对 fresh 分别差 17/19 个。这证明当前实现不是帧级确定函数。
- 差异比例：garden LiDAR/target/mask 为 59.58%/58.91%/44.46%；loop3 为 5.35%/4.00%/4.76%，符合状态化地面滤波差异向派生监督传播的结构。
- 本机环境：Python 3.8.12、NumPy 1.24.4、SciPy 1.10.1、OpenCV 4.10.0、pypatchworkpp 1.0.4，绑定模块 SHA-256 为 `afab14dc...13b43a`。
