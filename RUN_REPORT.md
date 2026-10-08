# 本机运行报告

验证日期：2026-10-07。项目路径：`/Users/hr/Desktop/diffusion/unet_starter`。

## GPU 版本更新

启动方式已进一步改为 Python：运行 `run_gpu.py` 即可串起数据准备、训练和预测，首次安装运行 `setup_gpu.py`。旧 `.sh` 入口仅作兼容转发。Python 启动入口已完成 CPU 单轮端到端验证，生成模型、指标、日志和预测图，并验证子步骤失败能向上传递退出状态、默认 CUDA 不可用时不会创建结果目录。当前改动已经直接写入本地项目，服务器沿用既有同步配置即可。

后续已按 RTX 3090 单卡服务器需求更新：训练默认 CUDA + AMP，增加 Linux 安装脚本、GPU 自检和保存日志的启动入口，详见 `SERVER_GPU.md`。以下 20 轮成绩保留为此前 CPU 基线。

GPU 改动后的本地验证：6 项 unittest 中 5 项通过，1 项真实 CUDA / AMP 运算因本机无 NVIDIA GPU 明确跳过；额外完成了 CPU 单轮训练及预测回归、默认 CUDA 不可用时在创建结果目录前报错、Linux 安装脚本在 Mac 上拒绝运行、三个 Shell 脚本语法检查。真实 RTX 3090 运行尚待服务器自检，本报告没有 GPU 实测速度或精度。

## 环境

| 项目 | 本次实际配置 |
| --- | --- |
| 平台 | macOS / Apple Silicon（arm64） |
| Python | 3.13.5 |
| PyTorch | 2.14.1 |
| NumPy | 2.5.3 |
| Pillow | 12.3.0 |
| certifi | 2026.7.22 |
| 虚拟环境 | 项目内 `.venv/` |
| 运行设备 | CPU，4 个计算线程 |

完整依赖版本见 `requirements-lock.txt`。`pip check` 通过，无依赖冲突。CUDA / MPS 分支未实测；本次完整训练及两种预测方式均使用 CPU。

## 数据已落盘

- 完整 170 张原图及 170 张对应标注已解压到 `data/PennFudanPed/`。
- 压缩包 53,723,336 字节，保留在 `data/PennFudanPed.zip`。
- 来源和完整文件 SHA-256 见 `data/metadata.json`；实际下载来源见 `data/archive_source.json`。
- 本次从 AWS 官方示例仓库取得完整备份，通过 ZIP CRC 校验；与大学原站下载的前 6,422,528 字节一致。大学原站的全文件未下载完成，不能据此声称做过全文件比对。
- 固定种子 42，120 / 25 / 25 张训练 / 验证 / 测试图片，集合之间无重叠。

## 已执行的训练

```bash
cd /Users/hr/Desktop/diffusion/unet_starter
.venv/bin/python train.py --epochs 20 --device cpu --output-dir runs/demo
.venv/bin/python predict.py --device cpu
```

模型含 487,297 个参数；输入分辨率 128×128；batch size 为 8；学习率 0.001；BCE + Soft Dice 损失；不使用预训练模型。

| 指标 | 验证集（25 张） | 测试集（25 张） |
| --- | ---: | ---: |
| Loss | 0.6023 | 0.6197 |
| 平均 Dice | 0.7085 | 0.6795 |
| 平均 IoU | 0.5617 | 0.5298 |

最佳模型来自第 20 轮。训练循环、验证及最终测试耗时合计 **49.07 秒**，不含依赖安装、下载和数据预加载。耗时仅为本机本次结果。

`runs/demo/best.pt` 可以直接用于预测；重新训练请换一个输出目录，例如 `runs/my_first_run`，以免覆盖结果。

## 已完成的验证

- 3 个 unittest 检查全部通过：非整倍数输入尺寸及反向传播、划分互斥性、训练集图片范围和二值标注。
- 20 轮完整训练、验证选模、独立测试集评价全部完成。
- 8 张固定测试图的单张对比图、黑白预测图和组合图已生成。
- 单张图片预测流程通过，预测 mask 成功恢复到输入图片的 570×422 原始尺寸，结果位于 `runs/demo/single_image/`。
- 已查看 `runs/demo/predictions/comparison_grid.png`：输出布局正常，能看到行人分割，同时也有背景误检、边缘较粗和局部漏检。
- `run_demo.sh` 通过 Bash 语法检查；它调用的下载、训练和预测 Python 入口已分别实测。

这是用于理解训练流程的轻量基线，不能当成高精度行人分割系统。测试集指标不用于选择超参数；不同设备和软件版本可能有数值差异。

## 结果入口

- `README.md`：从环境安装到训练 / 预测的完整中文说明。
- `runs/demo/history.csv`：每一轮的训练与验证记录。
- `runs/demo/config.json`：实际超参数和环境。
- `runs/demo/metrics.json`：验证及测试指标原始数值。
- `runs/demo/predictions/comparison_grid.png`：左至右为原图、真实标注、预测标注、预测叠加图。

当前整个项目约 863 MiB，其中虚拟环境约 756 MiB，数据及压缩包约 104 MiB，模型与预测结果约 3 MiB。
