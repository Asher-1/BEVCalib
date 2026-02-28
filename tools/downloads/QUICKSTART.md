# 快速开始 - Trip 数据下载

## ⚡ 重要说明

**默认行为**：脚本会**只下载**每个 trip 中的以下内容：
- `configs/` → 下载到 `configs/` 目录（先下载，小文件快速）
- `bags/important/` → 下载到 `bags/important/` 目录（后下载）

下载顺序：先下载配置文件，再下载 bag 文件，便于快速查看配置。

如需下载完整 trip，请使用 `--full` 参数。

## 🚀 一分钟快速上手

### 1. 下载单个 trip（只下载 bags/important 和 configs）

```bash
cd /mnt/drtraining/user/dahailu/code/BEVCalib/tools/downloads
python3 download_trips.py YR-P789-19_20260213_012754
```

### 2. 下载多个 trips

```bash
python3 download_trips.py YR-P789-19_20260213_012754 YR-B26A1-1_20251117_031232
```

### 3. 下载完整的 trip（所有数据）

```bash
python3 download_trips.py --full YR-P789-19_20260213_012754
```

### 4. 从文件批量下载

**步骤 1：创建 trip 列表文件**

编辑 `my_trips.txt` 文件：

```text
YR-P789-19_20260213_012754
YR-B26A1-1_20251117_031232
YR-XXX-XX_XXXXXXXX_XXXXXX
```

**步骤 2：执行下载**

```bash
python3 download_trips.py --file my_trips.txt
```

### 5. 检查 trips 是否存在（不下载）

```bash
python3 download_trips.py --check-only YR-P789-19_20260213_012754
```

## 📁 下载结构

### 默认模式（选择性下载）

下载 bags/important 和 configs，保持原有目录结构：

```
trips/
├── YR-P789-19_20260213_012754/
│   ├── bags/
│   │   └── important/             # ✓ 保持 important 目录
│   │       ├── Tiny_Topic_Group/
│   │       ├── Medium_Topic_Group/
│   │       └── ...
│   └── configs/                   # ✓ configs 目录
│       ├── calib.yaml
│       └── ...
└── YR-B26A1-1_20251117_031232/
    └── ...
```

### 完整模式（--full）

下载整个 trip 的所有数据：

```
trips/
├── YR-P789-19_20260213_012754/
│   ├── bags/              # ✓ 下载所有子目录
│   │   ├── important/
│   │   ├── other/
│   │   └── ...
│   ├── configs/           # ✓ 下载
│   ├── logs/              # ✓ 下载
│   └── ...                # ✓ 下载所有
```

## 📝 日志文件

每次运行都会生成日志文件：

```
/mnt/drtraining/user/dahailu/data/bevcalib/download_trips_YYYYMMDD_HHMMSS.log
```

## ❓ 常见命令

| 命令 | 说明 |
|------|------|
| `python3 download_trips.py --help` | 查看完整帮助 |
| `python3 download_trips.py TRIP_NAME` | 下载 bags/important 和 configs（默认） |
| `python3 download_trips.py --full TRIP_NAME` | 下载完整 trip |
| `python3 download_trips.py --check-only TRIP_NAME` | 仅检查不下载 |
| `python3 download_trips.py -f trips.txt` | 从文件读取列表 |
| `python3 download_trips.py -o /custom/path TRIP_NAME` | 指定输出目录 |
| `python3 download_trips.py --interactive` | 交互式输入 |

## ⚠️ 注意事项

1. **默认选择性下载**：脚本默认只下载 `bags/important` 和 `configs` 目录，节省空间和时间
2. **确保已配置 drfile**：首次使用需要运行 `drfile configure` 配置认证信息
3. **检查网络连接**：下载需要访问内网
4. **磁盘空间**：确保有足够的磁盘空间存储 trip 数据
5. **中断下载**：按 `Ctrl+C` 可以中断，已下载的数据会保留

## 🎯 实用技巧

### 批量下载前先检查

```bash
# 先检查哪些 trips 存在
python3 download_trips.py --check-only \
  YR-P789-19_20260213_012754 \
  YR-B26A1-1_20251117_031232 \
  YR-XXX-XX_XXXXXXXX_XXXXXX

# 确认无误后再下载
python3 download_trips.py \
  YR-P789-19_20260213_012754 \
  YR-B26A1-1_20251117_031232
```

### 查看下载进度

下载过程中会实时显示：
- 当前下载的 trip 名称
- 下载进度（X/总数）
- drfile 的输出信息
- 每个 trip 的文件数和目录数统计

### 失败重试

如果部分下载失败，脚本会在最后列出失败的 trips，可以：

```bash
# 查看日志了解失败原因
tail -100 download_trips_YYYYMMDD_HHMMSS.log

# 重新下载失败的 trips
python3 download_trips.py FAILED_TRIP_1 FAILED_TRIP_2
```

## 📚 更多帮助

详细文档请参考：[README_download_trips.md](./README_download_trips.md)
