# RTX 3090 服务器运行说明

目标环境：Linux + NVIDIA RTX 3090，单卡训练。训练代码默认 `--device cuda`，启用 FP16 AMP 混合精度；无可用 CUDA 时会直接报错。CPU 调试需要显式加 `--device cpu`。

## 1. 项目同步

本地和服务器已由你现有的同步配置同步。请在服务器进入同步后的 `unet_starter/` 目录，直接使用其中的 Python 文件，无需手动解压。

同步时排除 `.venv/`、`__pycache__/`、`runs/`。不要把 Mac 的 `.venv` 用在 Linux 上，也不要在服务器使用 Mac 的 `requirements-lock.txt` 来决定 CUDA 版本。

```bash
cd /你的服务器项目路径/unet_starter
nvidia-smi
```

## 2. 安装环境

建议使用服务器上的 Python 3.11 或 3.12。默认提供 PyTorch **2.7.1 + CUDA 12.6** 的固定安装组合；该组合列在 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/#v271)。项目无需 torchvision 和 torchaudio。

```bash
python setup_gpu.py
.venv/bin/python run_gpu.py
```

`setup_gpu.py` 会创建 `.venv`、安装 CUDA 版 PyTorch 和公共依赖，然后运行 GPU 自检。`.venv/bin/python run_gpu.py` 使用该环境启动训练，不需要额外的 Shell 启动文件。若需指定 Python：

```bash
python3.11 setup_gpu.py
```

目前只确认了显卡型号，未获知服务器驱动版本。先用 `nvidia-smi` 检查驱动；安装完成以 `check_gpu.py` 实际前向 / 反向运算通过为准。CUDA 运行时和驱动的兼容范围见 [NVIDIA 官方说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。如果管理员要求 CUDA 11.8 构建，可安装同一 PyTorch 版本的对应包：

```bash
python setup_gpu.py --cuda cu118
```

如果已有管理员配置的 PyTorch 环境，激活该环境后，仅安装公共依赖并自检即可：

```bash
python -m pip install -r requirements-gpu.txt
python check_gpu.py
```

`check_gpu.py` 会打印 GPU 名称、显存、PyTorch / CUDA 版本，并执行本项目 U-Net 的 AMP 前向、损失计算、反向传播和优化器更新。成功时输出 `GPU forward/backward OK`。

## 3. 运行

在已配置好依赖的 Python 环境中，一键 GPU 训练（128×128，batch 8，20 轮）：

```bash
python run_gpu.py
```

适合先在 3090 上尝试的更高分辨率配置：

```bash
python run_gpu.py --epochs 40 --image-size 256 --batch-size 8
```

脚本复用已上传的原始 ZIP，校验、解压并生成划分，然后训练和预测。默认运行目录为 `runs/gpu_时间戳_进程号/`，保存模型、指标、对比图、`train.log` 和服务器的 `requirements-server.txt`。不需要再次下载数据，依赖安装需要网络。

如需指定输出目录，直接使用 `--output-dir`（相对路径以项目目录为基准）：

```bash
python run_gpu.py --epochs 40 --image-size 256 --output-dir runs/rtx3090_first
```

多卡服务器选择一张空闲显卡：

```bash
CUDA_VISIBLE_DEVICES=1 python run_gpu.py --epochs 40 --image-size 256
```

这里选择物理 GPU 1，进程内部将它编号为 CUDA 0。脚本是单卡训练，不会自动占用其他显卡；使用调度器时沿用调度器分配的 `CUDA_VISIBLE_DEVICES`。

保持 SSH 断开后继续训练：

```bash
nohup python run_gpu.py --epochs 40 --image-size 256 > launch.log 2>&1 &
tail -f launch.log
```

也可以直接运行训练 / 预测：

```bash
python download_data.py
python train.py --device cuda --epochs 40 --image-size 256 --batch-size 8 --output-dir runs/manual_gpu
python predict.py --device cuda --checkpoint runs/manual_gpu/best.pt --output-dir runs/manual_gpu/predictions
```

不想敲启动命令时，也可以在 IDE 中打开 `run_gpu.py` 点击运行；将解释器设为服务器的 GPU 环境（例如项目 `.venv/bin/python`）。脚本使用自身目录定位数据，所有子步骤使用同一个 Python 解释器。

查看运行参数：

```bash
python run_gpu.py --help
```

## 4. GPU 相关配置

| 选项 | 含义 |
| --- | --- |
| `--device cuda` | 默认训练设备，GPU 不可用则报错 |
| `--device cuda:1` | 指定进程可见的第 2 张 GPU，训练和预测均使用此设备 |
| `--no-amp` | 关闭默认的 FP16 混合精度，以 FP32 训练 |
| `--batch-size 8` | 初始批大小，显存不足时降低到 4 或 2 |
| `--image-size 256` | 使用更清晰的输入，显存和算力开销也会增加 |
| `--num-workers 0` | 数据已预缓存，默认无需多进程；需要时可试 2 或 4 |

CUDA 模式启用 DataLoader pinned memory 和 non-blocking 张量传输；损失始终用 FP32 累加以防 FP16 大掩码求和溢出，验证 / 测试使用 FP32。混合精度使用 [PyTorch 官方 AMP 接口](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

## 5. 验证状态与排查

本地机器是 Mac，没有 NVIDIA CUDA GPU。已检查脚本语法、CPU 回归训练 / 预测和 CUDA 不可用时的报错；实际 CUDA / AMP 测试在本地会明确跳过，需要在服务器运行 `python check_gpu.py` 和 `python -m unittest -v test_project.py`。此前 `RUN_REPORT.md` 中的 49 秒和分割指标来自 CPU 示例，不是 3090 实测成绩。

- `CUDA 不可用`：检查作业是否分配 GPU、是否装了 CUDA 版 torch、驱动和容器是否能访问显卡。
- `CUDA out of memory`：减小 batch size 或分辨率，同时查看是否有其他任务占用显存。
- 多进程加载受限：使用默认 `--num-workers 0`。
- 初始数轮 Dice 较低：小模型从零训练的常见现象，查看完整训练曲线；单轮冒烟验证不代表最终效果。
