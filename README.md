# U-Net 入门：行人 / 背景分割

这是一个独立、可直接运行的 PyTorch 小项目。输入一张 RGB 照片，U-Net 为每个像素预测“行人”或“背景”。不依赖预训练权重，不需要注册数据集账号。

**RTX 3090 服务器运行请先阅读 [SERVER_GPU.md](SERVER_GPU.md)**：同步后执行 `python setup_gpu.py`，再用 `.venv/bin/python run_gpu.py` 启动。已有可用环境时直接执行 `python run_gpu.py`。训练默认 CUDA + AMP；GPU 环境安装、自检、选卡和后台运行命令均在该说明中。

## 1. 数据集

采用公开的 **Penn-Fudan Pedestrian Detection and Segmentation** 数据集：170 张图像、345 个行人实例，官方压缩包约 51 MB。

- [数据集发布者网站](https://www.cis.upenn.edu/~jshi/ped_html/)
- [官方压缩包](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)
- [PyTorch 使用此数据集的官方教程](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [AWS 官方示例仓库中的压缩包备份](https://github.com/aws-samples/aws-stepfunctions-byoc-mlops-using-data-science-sdk/blob/master/PennFudanPed.zip)

原始标注中 `0` 是背景、不同非零值是不同的行人。本项目将 `mask > 0` 合并成行人类别，做**二分类语义分割**。完整原图与原始标注保存在 `data/PennFudanPed/`，不会改写。

以固定随机种子 42 划分为 **120 张训练 / 25 张验证 / 25 张测试**，名单保存在 `data/splits.json`。这是本项目的教学划分，不是官方评测协议；部分场景可能相近，分数不代表跨场景泛化能力。来源、压缩包大小和实际下载文件的 SHA-256 保存在 `data/metadata.json`。本次原站下载较慢，实际完整压缩包来自上述 AWS 示例仓库；已校验 ZIP 完整性并与原站下载的前 6,422,528 字节比对一致，未做原站全文件比对。脚本记录本次完整文件的 SHA-256，后续复用 / 下载时会检查这个指纹。

## 2. 环境

建议 Python 3.11–3.13。项目使用独立 `.venv`，运行依赖只有 PyTorch、NumPy、Pillow 和用于 HTTPS 根证书的 certifi，不需要 torchvision、OpenCV 或 Jupyter。

当前这台 Mac 的环境已经安装在项目内。打开终端：

```bash
cd /Users/hr/Desktop/diffusion/unet_starter
source .venv/bin/activate
```

换一台电脑时，先进入复制后的项目目录，再安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 是跨环境的兼容范围；`requirements-lock.txt` 记录本次本机验证的精确版本，同类环境可用 `python -m pip install -r requirements-lock.txt` 复现。虚拟环境本身不能跨电脑直接搬用。Windows 使用 `py -m venv .venv`，PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`，其余 Python 命令相同。

训练默认使用 CUDA，没有可用 GPU 会报错；CPU 调试添加 `--device cpu`。显式传入 `--device auto` 时才按 CUDA → Apple MPS → CPU 选择。预测默认仍自动选择设备。NVIDIA 服务器使用 `setup_gpu.py` 和 `requirements-gpu.txt`，不要直接复用 Mac 的虚拟环境或精确版本锁文件。

## 3. 运行步骤

以下命令均在项目目录、激活环境后执行。下载后的完整数据已放在本项目，重复运行下载脚本会校验并复用本地压缩包，不会再次下载。

```bash
# 1）下载、解压数据，并生成固定划分
python download_data.py

# 原站慢时，可使用 AWS 示例仓库备份（文件已存在时仍复用本地文件）
python download_data.py --source mirror

# 2）训练 20 轮；选择验证集 Dice 最佳的模型
python train.py --epochs 20 --output-dir runs/my_first_run

# 3）预测 8 张测试图并保存对比图
python predict.py --checkpoint runs/my_first_run/best.pt --output-dir runs/my_first_run/predictions
```

服务器推荐 `python run_gpu.py`，它会自动准备数据、训练并预测。原有 macOS / Linux 通用一键入口仍可使用：

```bash
bash run_demo.sh --device auto
# 或先快速验证完整流程：
bash run_demo.sh --epochs 1 --limit-train 16 --device cpu
```

一键脚本将结果写入带时间戳的新目录。手动训练也请使用新的 `--output-dir`；程序会拒绝覆盖已有模型。单轮小样本命令只验证代码能否跑通，不能据此判断分割效果。

预测自己的图片（此模型只学习了行人 / 背景）：

```bash
python predict.py --checkpoint runs/demo/best.pt --image /绝对路径/照片.jpg --output-dir runs/my_photo
```

单图预测会将概率图恢复到原图尺寸，再用 0.5 阈值生成黑白标注图。测试集对比图使用训练尺寸，默认 128×128。

## 4. 文件说明

| 文件 / 目录 | 用途 |
| --- | --- |
| `requirements.txt` | 环境依赖范围 |
| `requirements-lock.txt` | 本机验证环境的精确依赖版本 |
| `requirements-gpu.txt` | 服务器公共依赖，CUDA 版 PyTorch 单独安装 |
| `setup_gpu.py` | Python 环境安装入口，配置 Linux RTX 3090 并自检 |
| `check_gpu.py` | 实际执行 CUDA / AMP 前向和反向传播 |
| `run_gpu.py` | Python 一键运行入口：数据准备、GPU 训练、预测和日志 |
| `setup_gpu.sh` / `run_gpu.sh` | 保留的旧命令兼容入口，内部调用 Python 文件 |
| `SERVER_GPU.md` | RTX 3090 上传、安装、启动和排查说明 |
| `download_data.py` | 从发布者下载完整数据、校验压缩包、生成数据划分 |
| `data.py` | 读取图片、二值化标注、缩放、训练集随机水平翻转 |
| `model.py` | 小型 U-Net 网络 |
| `train.py` | 训练、验证、保存最佳模型、评估测试集 |
| `predict.py` | 加载模型，生成预测标注及对比图 |
| `run_demo.sh` | macOS / Linux 一键运行入口 |
| `test_project.py` | 尺寸、梯度、划分互斥性和标注检查 |
| `RUN_REPORT.md` | 本机实际运行环境、结果与复现命令 |
| `data/PennFudanPed/PNGImages/` | 170 张原始 RGB 图片 |
| `data/PennFudanPed/PedMasks/` | 170 张原始实例标注 |
| `data/splits.json` | 固定训练 / 验证 / 测试名单 |
| `runs/demo/` | 已实际跑过的示例模型与结果 |

每次训练会生成：

```text
runs/你的运行目录/
├── config.json          # 超参数、设备、Python / Torch 版本
├── splits.json          # 本次所用数据划分副本
├── history.csv          # 每轮训练 loss、验证 loss / Dice / IoU
├── best.pt              # 验证集 Dice 最佳的模型权重
├── metrics.json         # 最佳轮次、验证 / 测试指标、训练耗时
└── predictions/         # 运行 predict.py 后生成
    ├── comparison_grid.png   # 原图 | 真实标注 | 预测 | 叠加
    ├── *_comparison.png
    └── *_mask.png            # 黑色背景、白色行人
```

## 5. 网络与参数

这是保留 U-Net 编码器、解码器和跳跃连接的轻量实现：3 次下采样，通道数 16 → 32 → 64 → 128，再逐级上采样、拼接同尺度编码特征，输出 1 通道 logits。上采样采用双线性插值，适合入门阅读；不是原始论文逐层复刻。

- 输入：`[batch, 3, 128, 128]`，RGB 值归一化到 0–1。
- 标签：`[batch, 1, 128, 128]`，0 为背景、1 为行人。
- 损失：`BCEWithLogitsLoss + Soft Dice Loss`。
- 优化器：Adam，默认学习率 `0.001`。
- 评价：阈值 0.5，每张图分别计算 Dice / IoU，再对图片取平均；越接近 1 越好。
- 只用训练集更新参数，用验证集选模型，训练结束后评估最佳模型的测试集指标。

常用参数示例：

```bash
# CPU 上更小的实验
python train.py --epochs 5 --image-size 64 --base-channels 8 --device cpu --output-dir runs/small

# 更大的分辨率，训练会更慢
python train.py --epochs 40 --image-size 256 --batch-size 4 --output-dir runs/larger

# 查看所有参数
python train.py --help
python predict.py --help

# 检查网络和数据
python -m unittest -v test_project.py
```

图像使用双线性插值，标注使用最近邻插值，避免产生不合法的类别值。默认把图片缩放到正方形，便于教学，会改变原图长宽比。固定随机种子有助于复现，但不同设备 / PyTorch 版本的训练结果仍可能不同。`best.pt` 用于推理，不包含恢复优化器状态所需的信息。
