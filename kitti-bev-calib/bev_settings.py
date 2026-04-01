# BEV Settings - 自适应配置
# 
# 使用方式：
#   1. 直接设置 DATASET_TYPE = "kitti" 或 "custom" 
#   2. 或者通过环境变量: export BEV_DATASET_TYPE=custom
#
# 自定义数据集可通过环境变量覆盖范围：
#   export BEV_XBOUND_MIN=0    BEV_XBOUND_MAX=200   BEV_XY_STEP=2.0
#   export BEV_YBOUND_MIN=-100 BEV_YBOUND_MAX=100
#   export BEV_ZBOUND_STEP=4.0
#
# 高分辨率模式示例 (X=[0,100], Y=[-50,50], step=1m → 100x100 grid):
#   export BEV_XBOUND_MIN=0 BEV_XBOUND_MAX=100 BEV_YBOUND_MIN=-50 BEV_YBOUND_MAX=50 BEV_XY_STEP=1.0

import os

# ========== 数据集类型选择 ==========
# "kitti" - 原始KITTI配置 (XY: ±90m)
# "custom" - 自定义数据集配置 (X: 0~200m, Y: ±100m)
DATASET_TYPE = os.environ.get("BEV_DATASET_TYPE", "custom")

# ========== KITTI 原始配置 ==========
KITTI_CONFIG = {
    "xbound": (-90.0, 90.0, 2.0),
    "ybound": (-90.0, 90.0, 2.0),
    "zbound": (-10.0, 10.0, 20.0),
    "d_conf": (1.0, 90.0, 1.0),
    "sparse_shape": (720, 720, 41),
}

# ========== 自定义数据集配置 ==========
_xy_step = float(os.environ.get("BEV_XY_STEP", 2.0))
CUSTOM_CONFIG = {
    "xbound": (
        float(os.environ.get("BEV_XBOUND_MIN", 0.0)),
        float(os.environ.get("BEV_XBOUND_MAX", 200.0)),
        _xy_step,
    ),
    "ybound": (
        float(os.environ.get("BEV_YBOUND_MIN", -100.0)),
        float(os.environ.get("BEV_YBOUND_MAX", 100.0)),
        _xy_step,
    ),
    # zbound步长决定图像BEV的Z分辨率:
    #   原始值20.0 → 1个Z体素 → 高度信息完全丢失 → 模型无法估计Z轴平移
    #   修改为4.0 → 5个Z体素 → 高度信息通过多层BEV特征保留
    #   图像特征在不同Z层的分布变化编码了高度偏移信息
    # 注意: 此设置仅影响图像分支的BEV pool，不影响点云分支
    #   (点云分支通过sparse_shape控制，始终为41个Z层)
    #   支持环境变量覆盖: export BEV_ZBOUND_STEP=20.0 (消融实验用)
    "zbound": (-10.0, 10.0, float(os.environ.get("BEV_ZBOUND_STEP", 4.0))),
    "d_conf": (1.0, 100.0, 1.0),
}

# ========== 根据数据集类型选择配置 ==========
if DATASET_TYPE.lower() == "kitti":
    _config = KITTI_CONFIG
    sparse_shape = _config["sparse_shape"]
else:
    _config = CUSTOM_CONFIG
    # PC voxel size must satisfy: (range / pc_voxel) / 8 == range / img_step
    # so that after SparseEncoder's 3x stride-2 downsampling (8x total),
    # the PC branch output matches the image branch BEV grid.
    _pc_voxel_xy = _xy_step / 8.0
    _pc_voxel_z = float((_config["zbound"][1] - _config["zbound"][0]) / 41)
    sparse_shape = (
        int((_config["xbound"][1] - _config["xbound"][0]) / _pc_voxel_xy),
        int((_config["ybound"][1] - _config["ybound"][0]) / _pc_voxel_xy),
        41,
    )

xbound = _config["xbound"]
ybound = _config["ybound"]
zbound = _config["zbound"]
d_conf = _config["d_conf"]

# ========== 通用参数 ==========
down_ratio = 8

# 自动计算 voxel size
vsize_xyz = [
    (xbound[1] - xbound[0]) / sparse_shape[0],
    (ybound[1] - ybound[0]) / sparse_shape[1],
    (zbound[1] - zbound[0]) / sparse_shape[2],
]
