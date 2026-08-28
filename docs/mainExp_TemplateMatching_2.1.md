# mainExp_TemplateMatching_2.1：1000 个弧长尺度的固定 8:2 流场检索

状态：**`development_completed_confirmation_not_run`**。冻结的 raw-field development 实验已由 Ibex job `50966604` 完成；已产生 accuracy、辅助指标、bootstrap、逐尺度表和两张三联图，但它仍不是 formal confirmation。

## 为什么是 2.1

`mainExp_TemplateMatching_1.2` 使用旧 Task5 cache、固定的旧尺度和七个 physical family leave-one-out。本实验同时改变 primitive 的重采样方法、尺度三元组、library/query population 和流场拆分，因此按 major iteration 新建 `mainExp_TemplateMatching_2.1`，不修改或覆盖 1.2 的代码、配置和结果。

2.1 的尺度三元组是 `(dx_grid_scale, ds_frame_scale, arc_length_grid_scale)`。第三维是目标空间弧长，不是旧协议的固定积分步数；每条线按空间弧长均匀重采样为 32 点。依据项目唯一研究协议，弧长重采样必须先由 `Verify_ArcLengthResampling_1.1` 验证。Ibex job `50966318` 已先行通过该门禁，随后才提交 portable staging、cache 和 evaluation。

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

## Ibex 执行与不可变证据

旧状态是“方法已冻结、尚无性能结果”；当前状态是“development 已完成、formal confirmation 未运行”。变化原因是以下任务按门禁顺序完成；旧状态没有错，只适用于正式投递前。

| 阶段 | Job / 结果 |
|---|---|
| 解析弧长验证 | `50966318`，`COMPLETED`；77/77 tests；1000/1000 primitives valid；最大XYZ误差 `1.1897e-7`，目标弧长误差0 |
| Ibex portable staging | `50966482`，5/5 array tasks `COMPLETED`；与本地5个数据集汇合后10 manifests/40 windows逐文件通过 |
| Primitive cache | `50966524`，40/40 array tasks `COMPLETED`；assigned `2,560,000`，valid `1,615,207`，invalid `944,793` |
| GPU evaluation | `50966604`，`COMPLETED`，exit `0:0`，elapsed `00:11:14`；Tesla V100-PCIE-32GB；batch MaxRSS `9069608K` |
| 重复排队 job | `50966575` 从未开始；`50966604` 成功后主动取消，未产生第二套结果 |

数值 commit 为 `59d54903d1f0f9d7525f69ceed136d08fd6797ed`，config SHA-256 为 `89b66176eb381eddc62d739dacafe86e681b60455f1ffb5e5f34ee9af8c81d1d`。结果目录是 `/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_2.1_development/runs/slurm_50966604_59d54903d1f0`；`result_manifest.json` 的文件/content SHA-256 分别为 `ff49a9ee3d95d9b6b2be4b2345393cf622b1bf63e3e82954ee1798d853b0e557` 与 `201ff46b0db3f4b12b75e82516663d5b613fd4fdd08aff15c7ef744fc4341707`。模板库含41,450个模板，正负类各20,725个；train-only PCA 使用1,370,364个valid train candidates。测试端assigned 512,000、valid 244,843，整体coverage为47.8209%。

冻结 config 中 `evidence_scope.performance_results_exist: false` 是运行前状态，不能在完成后用于判断结果是否存在。config 和源结果不可原地改写；权威完成证据是 `result_manifest.status=development_completed_confirmation_not_run` 与 `RUN_COMPLETE.json` 中相同状态及匹配的两个manifest哈希。

## 主要结果

主表先在每个测试physical family的四个source timeslice上计算指标，再对 `2 families × 4 timeslices` 等权平均。它避免Tangaroa的242,682个valid query按样本数淹没Smoke的2,161个valid query；pooled行只作描述。

| 方法 | Accuracy | Average Precision | F1 | Balanced accuracy | AUROC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train-label prior | 0.5868 | 0.4132 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| Raw672 exact 1NN | 0.4723 | 0.4258 | 0.4683 | 0.5117 | 0.4888 | 0.4130 | **0.9166** |
| Train-only Raw-PCA161 exact 1NN | 0.6496 | 0.5517 | 0.4876 | **0.6045** | **0.6537** | 0.4279 | 0.8207 |
| FMT161 exact 1NN | **0.7049** | **0.6029** | **0.4883** | 0.5686 | 0.5176 | **0.4406** | 0.7186 |

FMT在本次development split上的macro accuracy与Average Precision最高；它与Raw-PCA161的F1几乎相同，同时balanced accuracy、AUROC和recall更低。因此证据是混合的，不能概括成“FMT在所有指标上更好”。

| Test dataset | Assigned | Valid | Coverage | Positive / negative | FMT accuracy | FMT AP | FMT F1 | FMT balanced accuracy | FMT AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tangaroa` | 256,000 | 242,682 | 94.7977% | 11,409 / 231,273 | 0.7180 | 0.4155 | 0.1640 | 0.6564 | 0.5991 |
| `smokeBuoyancy` | 256,000 | 2,161 | **0.8441%** | 1,674 / 487 | 0.6923 | 0.7660 | 0.8105 | 0.5007 | 0.4910 |

Smoke只有0.8441%的assigned primitives满足七条线均达到目标弧长；其高AP/F1只描述这2,161个valid queries，不能外推到其余99.16%的assigned seeds。两流场合并的pooled FMT accuracy/AP/F1为 `0.7177/0.4526/0.1906`，但该行主要反映Tangaroa样本量，因此不是主结论。

5000次、seed `25068` 的paired dataset-source-timeslice bootstrap给出：

| FMT减比较方法 | Metric | Difference | 95% percentile interval |
|---|---|---:|---:|
| FMT − Raw672 | Accuracy | +0.2325 | [+0.1880, +0.2857] |
| FMT − Raw-PCA161 | Accuracy | +0.0552 | [+0.0114, +0.1109] |
| FMT − Raw672 | Average Precision | +0.1771 | [+0.1303, +0.2245] |
| FMT − Raw-PCA161 | Average Precision | +0.0512 | [+0.0234, +0.0826] |
| FMT − Raw672 | F1 | +0.0201 | [+0.0027, +0.0384] |
| FMT − Raw-PCA161 | F1 | +0.0007 | [−0.0212, +0.0278] |
| FMT − Raw-PCA161 | Balanced accuracy | −0.0359 | [−0.1260, +0.0353] |
| FMT − Raw-PCA161 | AUROC | −0.1361 | [−0.2717, −0.0474] |

区间是只对固定的两个已暴露test families及其timeslices做的描述性不确定性估计，不是对新physical family泛化的置信保证。

## 三联图与图件审计

两个test dataset都按预注册规则使用source ordinal 2。每张图的三栏分别是：(a) whole-loaded-volume IVD-p95完整等值面、同一valid query seed背景与固定120正类+120负类中心pathlines；(b) FMT global exact 1NN模板类别分配，不是聚类；(c) 对同一seeds的TP、FP、FN和淡化TN。Tangaroa图中 `TP/FP/TN/FN=1851/19940/37706/966`；Smoke图中为 `305/138/33/49`。

两张PNG均为7560×1800、360 dpi，并各有含可编辑文字的PDF、immutable scene NPZ、render metadata和面板几何JSON。本地下载后按result manifest重新核验全部40个文件，0个大小或SHA-256不一致。PDF最小字体为7 pt；三面板几何在1.5 pt阈值下均为PASS；collision audit均为0 FAIL。Tangaroa的13个与Smoke的9个WARN只涉及3D坐标刻度接触轴面或栅格边缘，逐图检查后未发现文字遮挡。该21×5 inch宽图是FMT式研究诊断图；若作为期刊最终版缩放到双栏宽度，需要另建layout版本并重新检查缩放后的字号，不能直接按比例缩小。

`evaluation_summary.log` 只是 evaluator 写入run directory的内部摘要，不是scheduler stdout。真实Slurm日志、终态、设备与文件SHA-256登记在 `docs/ibex_run_registry.md`；结构化结果摘要位于 `docs/evidence/mainExp_TemplateMatching_2.1_ibex_summary.json`。formal confirmation仍未运行。
