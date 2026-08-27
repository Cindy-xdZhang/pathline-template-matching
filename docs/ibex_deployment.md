# Ibex 部署与流场数据状态

状态日期：2026-08-27，时区 Asia/Riyadh。

## 登录与仓库

正确主机为 `glogin.ibex.kaust.edu.sa`，用户为 `zhanx0o`。本机 SSH config 没有 `ibex` 别名，因此不要写 `ssh ibex`。

```bash
ssh -o BatchMode=yes glogin.ibex.kaust.edu.sa
git clone git@github.com:Cindy-xdZhang/pathline-template-matching.git \
  /home/zhanx0o/pathline-template-matching
cd /home/zhanx0o/pathline-template-matching
bash ibex/validate.sh
```

旧 FMT 目录 `/home/zhanx0o/FMT_Task12_3D_20260823` 没有 `.git`，历史上采用 bundle/archive 部署，不能作为 `git pull` 工作流范例。本项目固定采用 GitHub SSH clone/pull，并让每个实验记录精确 commit。

## Python 环境

```bash
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
```

只读审计实测：Python `3.12.9`、NumPy `1.26.4`、PyTorch `2.6.0+cu118`、netCDF4 `1.7.2`、Numba `0.61.2`、PyYAML `6.0.2`。为生成 IVD-p95 Marching Cubes 等值面，2026-08-27 在同一 `deepvortex` 环境安装并固定 `scikit-image 0.25.2`、`tifffile 2024.9.20`、`imageio 2.37.4`、`lazy-loader 0.5`。首次 pip 解析曾临时选择不兼容的 NumPy `2.4.6`；部署当次已恢复 NumPy `1.26.4` 并把 `tifffile` 固定到兼容版本，最终 `pip check` 返回 `No broken requirements found`。项目依赖同时限制 NumPy `<2.3` 和 tifffile `<2025`，防止再次解析到该冲突。登录节点 `torch.cuda.is_available() == False` 正常；GPU 必须在 Slurm 分配节点内验证。

Slurm 为 `25.05.7`；`gpu4` 和 `debug` partitions 存在，`cuda/11.8` module 可用。任何 GPU 实验仍须在 job 内记录 `hostname` 和 `nvidia-smi`，不能由本清单推断实际设备。

## 原始三维流场

共同目录：`/home/zhanx0o/DeepVortex/FLowDataFolder`。

| Dataset | 原始文件状态 | 实测维度 `(x,y,z,t)` |
|---|---|---|
| `cylinder3d` | 可读：`halfcylinderRe160Resampled.nc` | `(160,60,20,151)` |
| `halfcylinderRe640` | 可读：`halfcylinderRe640resampled.nc` | `(160,60,20,76)` |
| `tangaroa` | 可读：`tangaroa.nc` | `(300,180,120,201)` |
| `deltaWing_resampled` | 可读：`deltaWing_mag0_3reesampled.nc` | `(55,314,55,171)` |
| `smokeBuoyancy` | 可读：`SmokeBuoyancy80_239.nc` | `(47,95,47,160)` |
| `halfcylinderRe6400` | 未找到原始文件 | — |
| `deltaWing_LBM` | 未找到原始文件 | — |
| `f22raptor` | 未找到原始文件 | — |
| `channel` | 未找到 `channel.vtk` | — |
| `boeing747` | 未找到原始文件 | — |

使用 FMT loader 对 Re160 的时间索引 31 读取 2 帧、空间最大维度 8，已得到 shape `[2,7,8,8,3]`、finite `true`。2026-08-27 的首次 clone 验证又用本项目 loader 实际读取 5 个可用 NetCDF 的小窗口；mask、NaN/Inf、轴顺序、物理坐标单调性和等间距检查全部通过。

本机交叉检查也使用相同 loader 实际读取了 2 帧、空间最大维度 8：8 个 NetCDF 通过；`f22raptor` 因 time coordinate 为 masked/non-finite 被拒绝；`channel.vtk` 因本项目尚无 VTK loader 被拒绝。文件存在不等于可正确重新积分。

## 旧 Task5 派生缓存

首次 clone 验证确认 10 个数据集各有 development 6 个 `.npz`、confirmation 4 个 `.npz`，合计 60+40；全部 100 片通过完整内容检查。根目录为：

```text
/home/zhanx0o/FMT_Task12_3D_20260823/outputs/mainExp_Task5_3D_1.1/development_cache
/home/zhanx0o/FMT_Task12_3D_20260823/outputs/mainExp_Task5_3D_1.1/confirmation_cache
```

缓存可用于本项目 bootstrap development，但有两条限制：

1. cache 可用不等于原始 flow 可重新积分；数据报告必须分别给出 raw 和 cache 状态。
2. 旧 confirmation 已被 FMT 项目查看，在本项目中不能称为 sealed confirmation。

项目 validator 会打开全部 100 个 slice，逐个检查 672D Raw、161D FMT、二值标签、seed、scale ID、finite、dataset/phase/ordinal、旧实验 ID，并要求每片 config SHA-256 严格等于 registry 冻结的 canonical digest `e9eae4d2cc0a76ba768aed9a61cbbd430790109593cd4214cfaea93b76f56b4b`；不能用“彼此一致”或每个目录第一片替代 canonical 验证。Ibex 完整验证结果写入 `outputs/Other_ProjectBootstrap_1.1/ibex_data_access.json`，并登记到 `docs/ibex_run_registry.md`。

首次验证的 code commit 为 `f202095c1572c668716a407d987fb0882add3ab6`；报告 SHA-256 为 `609bd188229c3c7a40db2d8ea1648c517a7bd3a0ced5bccec4a0c207b714a84d`。远端仓库位置固定为 `/home/zhanx0o/pathline-template-matching`。

机器相关路径只写在 [config/datasets.yaml](../config/datasets.yaml)，实验 config 不重复硬编码。

## 同步纪律

本地修改先运行测试、commit、push；Ibex 只 checkout 已提交 commit。Slurm job 不允许依赖未提交文件。若必须临时诊断 dirty code，要记录文件 SHA-256 并归入 `Other_...`，不得生成论文主结论。
