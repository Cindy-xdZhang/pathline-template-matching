# mainExp_TemplateMatching_2.1：1000 个弧长尺度的固定 8:2 流场检索

状态：**`frozen_pre_run_not_run`**。本文件、`config/mainExp_TemplateMatching_2.1.yaml` 与实现已冻结 raw-field development 实验；尚未提交 Ibex job，尚无 accuracy 或其他性能结果，也不是 formal confirmation。

## 为什么是 2.1

`mainExp_TemplateMatching_1.2` 使用旧 Task5 cache、固定的旧尺度和七个 physical family leave-one-out。本实验同时改变 primitive 的重采样方法、尺度三元组、library/query population 和流场拆分，因此按 major iteration 新建 `mainExp_TemplateMatching_2.1`，不修改或覆盖 1.2 的代码、配置和结果。

2.1 的尺度三元组是 `(dx_grid_scale, ds_frame_scale, arc_length_grid_scale)`。第三维是目标空间弧长，不是旧协议的固定积分步数；每条线按空间弧长均匀重采样为 32 点。依据项目唯一研究协议，弧长重采样必须先由 `Verify_ArcLengthResampling_1.1` 验证。该 Verify、实现测试和输入门禁未完成前，2.1 主 job 禁止投递。

## 冻结的 8:2 拆分

拆分单位是完整 physical family，不拆 spatial seed、source timeslice 或它后续 12 frame 的积分窗口。十个数据集都已在 FMT 或本项目 1.2 中暴露，因此 test 只是 development test，不是首次读取的 sealed confirmation。

| Partition | Dataset | Physical family |
|---|---|---|
| Train | `cylinder3d` | `half_cylinder` |
| Train | `halfcylinderRe640` | `half_cylinder` |
| Train | `halfcylinderRe6400` | `half_cylinder` |
| Train | `deltaWing_resampled` | `delta_wing` |
| Train | `deltaWing_LBM` | `delta_wing` |
| Train | `f22raptor` | `f22_raptor` |
| Train | `channel` | `channel` |
| Train | `boeing747` | `boeing_747` |
| Test | `tangaroa` | `tangaroa` |
| Test | `smokeBuoyancy` | `smoke_buoyancy` |

Train 有五个 physical family，test 有两个 singleton physical family；两侧没有 family 重叠。任何 scaler、Principal Component Analysis（PCA，主成分分析）、prior、模板抽样或方法选择只能读取 train。Test 的 primitive、标签、coverage、匹配和指标在代码 commit、config 和 input manifest 冻结前均不得用于调整方法。

## Source time、seed 与尺度

每个 dataset 固定四个 source time。对含 `T` 个均匀时间 frame 的 raw field，最后可用 source index 为 `T−13`，四个 index 固定为：

```text
floor(k × (T − 13) / 3),  k = 0, 1, 2, 3
```

要求四个 index 唯一，因此 `T>=16`；选择只依赖 time-axis length，不依赖速度、IVD、有效率或标签。每个 raw spatial axis 使用 `stride_axis=ceil(native_axis_count/96)` 并取 `native_axis[0::stride_axis]`，所以已加载 axis 不超过 96；实际 `[stride_x,stride_y,stride_z]` 必须逐 dataset 写入 manifest。每个 source time 的派生窗口是该 strided spatial volume 上连续 13 frames，而不是 native full-resolution window。

`f22raptor` 是明确记录的坐标例外：其 `x/y/z/t` 一维 coordinate variables 在原文件中全部 masked。为复现旧 FMT 的数值约定，2.1 只对该 dataset 显式使用整数 array index coordinates；其 `dx` 和目标弧长因此是 **spatial index units**，`ds` 是 frame units，不能称为已知物理米/秒。其他 NetCDF 坏坐标仍立即失败，不允许隐式 fallback。`channel` 不是原生时变测量：它由一个 steady structured-grid VTK 经过冻结的159帧 Killing observer 转换得到，manifest 必须标记为 synthetic unsteady observer view。

每个 source time 使用一个 endpoint-inclusive `40×40×40` center-seed grid，共 64,000 seeds。中心 seed domain 从每个已加载空间边界内缩本数据集最大 `dx`，保证七条线的初始点都在域内；与计划中的 NumPy meshgrid 一致，seed 的稳定顺序固定为 z outer、y middle、x inner。

三个冻结的一维数组均在 config 中以 12 位小数 numeric scalar 逐项列出，scale manifest 必须以 fixed-point 12 位小数序列化；不允许运行时再次调用 `linspace`：

- `dx_grid_scale`：0.25 到 1.25，共 10 个值；
- `ds_frame_scale`：0.125 到 0.30，共 10 个值；
- `arc_length_grid_scale`：4 到 12，共 10 个值。

Cartesian product 顺序固定为 dx outer → ds middle → arc inner，`scale_id=((i_dx×10)+i_ds)×10+i_arc`，得到 1000 个唯一 tuple。每个 source time 独立用 `numpy.random.Generator(PCG64(15068))` 生成 64,000 seed permutation，并执行 `assignment[permutation[j]]=j mod 1000`。所以每个 seed 只属于一个尺度，每个 source time 的每个尺度恰有 64 个 assigned seeds；分配不能读取 seed 位置、速度、IVD、类别或积分有效性。

总候选量在过滤前固定为：train `8×4×64,000=2,048,000` 个 primitives，test `2×4×64,000=512,000` 个 primitives。每个 primitive 含七条 pathline。

## 弧长 primitive

令 `h_min` 为数据集三个已加载 strided 空间网格间距的最小值，`Δt_frame` 为均匀 source-frame 间隔。每个 tuple 的物理参数为：

```text
neighbor offset     = dx_grid_scale × h_min
RK4 time step       = ds_frame_scale × Δt_frame
target arc length   = arc_length_grid_scale × h_min
maximum time horizon = 12 × Δt_frame
```

从 `center, x+, x−, y+, y−, z+, z−` 七个初始点 forward RK4。如果下一个完整 RK4 step 会越过 12-frame horizon，则最后一步缩短到准确的剩余时间，禁止越过 horizon。某条积分折线第一次跨过目标累计空间弧长时，在最后一段上做线性精确截断；再在 `s_j=j×target_length/31` 上按累计弧长分段线性插值，得到含两端点的 32 点。descriptor 前丢弃时间通道，并以 center line 的初始 xyz 平移全部七条线。

只有七条线都在 12-frame horizon 内达到目标弧长，primitive 才 valid。任一线提前出域、停滞未达目标、出现非有限值或不能产生 32 个有序样本时，整个 primitive invalid；不得保留部分线，也不得用另一个 seed 或尺度替换。每个 `dataset×time×scale×class` 必须报告 assigned、valid 和 invalid。

## 标签、建库与匹配

标签固定为 source frame 整个已加载 strided spatial volume 上的 Instantaneous Vorticity Deviation（IVD，瞬时涡度偏差）p95：

```text
IVD = ||curl(v) − spatial_mean_loaded_volume(curl(v))||
positive iff trilinear_sample(IVD, center_seed) >= percentile95(IVD volume)
```

每个 dataset 的空间 stride 由 `ceil(native_axis_count/96)` 冻结并记录，percentile method 为 `linear`。空间 mean、p95 threshold 和标签都只在整个已加载的 strided volume 上计算；不得称为 native full-resolution IVD。逐 source time 保存 native/loaded shape、实际 stride、数值 threshold、volume/seed 正类比例和 IVD array SHA-256；标签与 scale assignment 无关。

Train library 在每个 `dataset×source time×scale×class` 内平衡。只有一个 scale stratum 的 positive 和 negative 都非空时才建模板，并且每类恰取一个：

```text
selected_negative = 1
selected_positive = 1
```

若任一类为空，则两类都选 0、完整审计，并且该 stratum 不消耗伪随机数。候选按 seed index 升序；用一个 `numpy.random.Generator(PCG64(15068))`，依次按冻结 train dataset 顺序、source time、scale ID 遍历；只有两类都非空时才依次为 negative、positive 各均匀确定性抽一个 candidate。Train 一共有 `8×4×1000=32,000` 个可能的双类 stratum，因此 global library 最多 `32,000×2=64,000` 个模板。所有已选模板连接成一个跨 dataset、source time 和全部 1000 scale 的 global library。Test 保留全部 valid primitives 及自然类别比例，不做平衡或降采样。Query 的一最近邻搜索不限制为同尺度；最近模板可以来自任意 train scale，但必须返回它的 dataset、time、seed、scale、label 和距离。

四个固定比较臂为 train-label prior、centered Raw 672D + exact one-nearest-neighbor（1NN，一最近邻）、train-only Raw-PCA 161D + exact 1NN、independent FMT 161D + exact 1NN。Raw-PCA 在全部 valid train candidates 上用 float64 两遍 streaming covariance：第一遍累计 count/mean，第二遍累计 `672×672` scatter matrix，再做 deterministic symmetric eigendecomposition（对称特征分解），按 eigenvalue 降序取161维并冻结符号。它与 centered matrix 的右 singular vectors 定义同一个 PCA，但避免显式生成最多约 `2.048M×672` 的 float64 matrix 和巨大 left-singular-vector array；test 不得进入两遍累计。所有 feature normalization 均只从对应的 selected balanced train library 拟合并序列化，test 不得更新。匹配为 global cross-scale exact Euclidean 1NN；近似索引、reject threshold、top-k 和同尺度过滤均禁止。

## 指标与统计边界

按用户要求，普通 accuracy 定义为 valid query 上的 `(TP+TN)/(TP+TN+FP+FN)`。同时必须报告 prior baseline、balanced accuracy、Average Precision（AP，平均精确率）、F1、Area Under the Receiver Operating Characteristic Curve（AUROC，受试者工作特征曲线下面积）、precision、recall 和 coverage。Coverage 定义为 `valid/assigned`；invalid primitive 不进入分类指标分母，但必须进入 coverage 分母。

表格至少给出 per source-timeslice、per test dataset、两个 test physical family 等权宏平均、per scale tuple 和 pooled descriptive 结果。单类 slice/scale 的 AUROC 或 AP 写为 null 并登记原因，不得伪造为 0 或 0.5。普通 accuracy 容易被约 5% 正类比例主导，因此不能脱离 prior、balanced accuracy、AP 和 F1 单独解释。

描述性区间固定用 seed `25068` 做 5000 次 paired source-timeslice bootstrap：在两个 test family 内各对四个 timeslice 有放回重采样，再对两个 family 等权宏平均；NumPy percentile method 固定为 `linear`。只有两个已暴露 test family，因此这些区间不支持一般新 physical family 的 formal confirmation 结论，也没有预注册 pass/fail 判定。

## 固定三联图

两个 test dataset 都固定使用 source ordinal `2`，不得根据 accuracy、Average Precision、F1 或视觉效果选图。每张图的三栏分别是：

1. 同一 immutable cache 中 whole-loaded-volume IVD-p95 的完整等值面，加上 240 条中心 pathlines；
2. FMT global exact one-nearest-neighbor 的 vortex/non-vortex template class assignment，不得称为 KMeans 或 clustering；
3. 相对 IVD-p95 的 TP、FP、FN 和淡化的 TN。

三栏必须使用同一组全部 valid query seed 坐标和相同相机。第一栏的 240 条解释性中心线固定为 120 positive + 120 negative，只使用 seed 坐标、三个 scale index 与 reference class 做确定性 maximin 选择，禁止读取 prediction 或 metric。时间着色必须使用 cache 中保留的 32 个真实弧长采样时刻，不能用 endpoint 线性猜测。每张图保存 scene NPZ、360-dpi PNG、PDF、面板对齐 JSON 和逐文件 SHA-256。三联图只解释空间上的匹配与错误，不替代汇总表的性能证据。

## Hash、输出与不可覆盖规则

运行前必须记录每个 raw source 文件的路径、大小和原始文件 SHA-256。每个 13-frame、max-spatial-dimension-96 的 strided 派生窗口按 canonical little-endian C-contiguous 数组保存 `velocity[T,Z,Y,X,3]`、`x`、`y`、`z`、`time` 的 dtype、shape 和各自 SHA-256，并记录 native/loaded shape、`[stride_x,stride_y,stride_z]` 与 combined content SHA-256。还必须哈希 registry、config、1000-tuple scale manifest、seed/scale assignment、标签、primitive、template library、train-only preprocessing artifacts 和最终结果。

完整 required-output 列表以 config 为准。Ibex home 已接近 soft quota，因此 portable windows、primitive cache 和每个 run 固定写到 `/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_2.1_development/`；run 目录为其中新的 `runs/slurm_JOBID_COMMIT12`。目录已存在即失败，禁止覆盖。Git 仓库只保存小型证据摘要，不复制大型 cache。`result_manifest.json` 和所有必需输出完成并持久化后，最后原子写 `RUN_COMPLETE.json`；缺少 completion marker 的目录不能称 completed。

## 当前投递门禁

1. 必须把冻结的 config、builder、runner、tests 和 Slurm 脚本提交为同一 Git revision，local staging 与 Ibex 都只能使用该 clean revision。
2. 必须先在 Ibex 完成 `Verify_ArcLengthResampling_1.1`；失败则不得生成主实验 cache。
3. Ibex 目前只有 5/10 raw fields。本地已确认的 `halfcylinderRe6400`、`deltaWing_LBM`、`f22raptor`、`channel`、`boeing747` 必须在同一 revision 下生成 portable windows 并上传；其余五个由 Ibex raw fields 生成。
4. 必须在首次读取 test predictions 前冻结 10 个 dataset×4 个 source times 的 portable/input manifests，逐文件验证 config、registry、builder commit 和 SHA-256。
5. 目前仍无 Slurm job、result manifest 或任何性能数字；不得写成已部署、已完成或 formal confirmation。

`evaluation_summary.log` 只是 evaluator 写入 run directory 的内部摘要，不得称为 scheduler stdout。真实 Slurm stdout/stderr 位于仓库 `slurm_logs/`，每个 job 必须在 `docs/ibex_run_registry.md` 登记其绝对路径、终态和文件 SHA-256；结论必须引用这些真实日志。

实现已包含弧长 RK4 primitive builder、portable-window staging、40-way cache builder、bounded-memory train-only Raw-PCA、template library、exact 1NN evaluator、指标、5000 次 bootstrap，以及固定 source ordinal 2 的两张三联图证据链。冻结前完整标准回归为 77/77 通过；这只是运行前代码证据，不是性能结果。
