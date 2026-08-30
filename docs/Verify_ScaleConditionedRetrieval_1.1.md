# `Verify_ScaleConditionedRetrieval_1.1`

状态：方法与候选集已冻结，尚未产生 Ibex 性能结果。

冻结配置：`config/Verify_ScaleConditionedRetrieval_1.1.yaml`
配置 SHA-256：`f5dbdae08e2e13140245a6a9fd12dba67b4eaf6a7ae1aaea8d600f89a409a6a2`

## 1. 为什么需要这个验证

`mainExp_TemplateMatching_3.1` 的平衡正负模板 exact one-nearest-neighbor（exact 1NN，穷举欧氏一最近邻）在跨流场测试上产生大量 false positive。`Other_NegativeDistanceSpatial_1.1` 证明“到负类模板的距离”比正负模板距离差具有更稳定的涡区排序；在四个已暴露 flow 的八个 `dataset×source×block` group 上，统一的 `sigma=1 + top 5%` 得到 F1 `0.5451`，高于同组父方法 F1 `0.2278`。但该候选由已查看的标签结果产生，不能证明泛化。

本验证只用 3.1 的八个 train flow，执行严格 nested leave-one-physical-family-out（嵌套按完整物理族留一验证）。外层族的 feature 在完成内层选择前不打开，外层标签在无标签预测文件关闭并计算 SHA-256 前不打开。

## 2. 与 FMT Task1 的差异

旧 FMT Task1 3.3 的十数据集宏平均 F1 为 `0.606360`，但它不是当前跨物理族模板检索任务：

| FMT Task1 3.3 | 本验证 |
|---|---|
| 在每个目标 flow 自身 development features 上拟合 scaler、Principal Component Analysis（PCA，主成分分析）和 KMeans | scaler 与负模板只来自其他物理族 |
| 用目标族标签选择 feature block/PCA，并用目标 flow 标签决定 cluster 的涡语义 | 外层族标签完全不能选择 representation、`k`、空间尺度或 threshold |
| 固定一个 pathline 尺度 | 同时保留 2000 个 `dx×ds×arc length` 尺度 |
| 六个条目使用依赖 query batch mean 的运动学 feature；Boeing 的 F1 `0.8403`来自该 block | 只使用每个 primitive 独立得到的 FMT161 子块 |
| development 与 confirmation 的完整 pathline future windows 重叠 | 完整 physical family 留出 |

因此 Task1 的高值不能作为当前方法的无泄漏基线，也不能把它的 batch-dependent IVD-like feature 复制后仍称为独立模板 descriptor。

## 3. 数据与 nested split

唯一输入 manifest：

```text
/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json
```

文件 SHA-256 为 `e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`；它认证 32 个 train cache。禁止打开 Tangaroa 与 Smoke Buoyancy cache。

| Physical family | Datasets | Cache shards | Valid | Natural negative | Positive |
|---|---|---:|---:|---:|---:|
| `half_cylinder` | `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400` | 12 | 1,309,366 | 1,234,669 | 74,697 |
| `delta_wing` | `deltaWing_resampled`, `deltaWing_LBM` | 8 | 589,051 | 564,364 | 24,687 |
| `f22_raptor` | `f22raptor` | 4 | 437,257 | 417,585 | 19,672 |
| `channel` | `channel` | 4 | 315,580 | 304,927 | 10,653 |
| `boeing_747` | `boeing747` | 4 | 316,358 | 303,084 | 13,274 |

五个外层 fold 依次留出上述一个完整 family。每个外层 fold 内，另外四个 family 再各留一个做 inner validation，剩余三个 family 拟合负类 library 与 scaler。总计 20 个 inner validation folds 与 5 个 outer folds；空间 seed、source time 或 primitive 不做随机拆分。

## 4. 冻结方法

### 4.1 Natural negative library

每个 fit population 使用所有 valid negative primitive，不做类别平衡、不抽样。全局 population mean/std 只在这些负类行上拟合；统计使用 float64、距离输入使用 float32。标准差严格小于 `1e-12` 的 feature 设 effective std 为 1。query 不更新 scaler。

### 4.2 Representation

候选只包含三个由 FMT161 固定列索引得到的块：

- `fmt161`：七条线的全部 `7×23=161` 维；
- `real_neighbor36`：六条邻线各取局部索引 `0,3,6,9,12,15`；
- `chirality_all35`：七条线各取局部索引 `18:23`。

不加入 PCA，也不加入依赖 query batch 统计的运动学 feature。

### 4.3 Scale-conditioned negative distance

每个 query 只与相同 numeric `scale_id` 的负模板比较，score 为第 `k` 个最近负模板的 exact Euclidean distance；`k∈{1,5,15,31}`。四个 `k` 在同一次最大 `k=31` exact top-k pass 中返回。`k=1`保留了上一探索的最近负模板机制；它与其他 `k` 一样可以由 inner labels 选择。

### 4.4 Rank、空间处理与缺少支持

每个 `dataset×source×block` 单独处理。在 fit-only 负类数不少于当前 `k` 时，该行是 supported；supported distance 按“distance 升序、center index 升序”得到 `(r+1)/N_supported` rank。Unsupported 行不能删除，rank 固定为 0。

`sigma∈{0,0.5,1,1.5,2}`：

- `sigma=0`：unsupported 行保持 0，并显式判负；
- `sigma>0`：只在同一个 source/block 的 `40³` center grid 上计算 support-mask-normalized spatial imputation：

```text
G(rank × support_mask) / G(support_mask)
```

Unsupported 行的 denominator 大于 0 时记作 `spatial_imputed`；denominator 为 0 时记作 `unimputable`、score 0、判负。输出分别报告 supported、imputed、unimputable 数量及子集 F1。插值行不能称作 exact-scale k-nearest-neighbor 命中。

困难 inner fit `{delta_wing, channel, boeing_747}` 的精确支持为：545 个尺度负类数为 0，597 个尺度少于 5，600 个尺度少于 15/31，1400 个尺度不少于 31。缺口全部在 expanded block。五个 outer final fit 对全部 2000 个尺度均至少支持 `k=31`。

空间处理使最小 query batch 成为完整 valid `source×block` grid；所以完整 classifier 是 transductive classifier（同组样本共同决定空间 score），不能称为 per-primitive independent classifier。只有 FMT encoder 保持逐 primitive 独立。

### 4.5 Decision 与选择

每个 representation/`k`/`sigma` 包含：

- fixed top 5%：目标数是 `ceil(0.05×N_all_valid_group_rows)`，但只可从 supported 或 imputed 且 score 严格大于 0 的行中选择，最终数量受 eligible 行数上限约束；
- calibrated rank threshold：`0.50–0.99`，步长 `0.01`，比较为 `score >= threshold`；ineligible 行强制判负。

候选总数为 `3×4×5×51=3060`。每个 inner family 内先等权平均全部 `dataset×source×block` groups，再对四个 inner families 等权平均。选择顺序固定为最高 F1、Average Precision、balanced accuracy、precision、recall，最后取字典序最小 candidate ID。这样不会让三个 half-cylinder 数据集获得 singleton family 的三倍权重。

## 5. Outer 信息隔离

每个 Slurm array task 只处理一个 outer family，严格按以下顺序：

1. 只打开另外四个 family 的 features、labels 与身份字段；完成 inner validation。
2. 在另外四个 family 上重新拟合所选 representation 的 final scaler/library。
3. 写入、关闭并 hash `selected_candidate.json`；文件记录 selected candidate、final scaler 和每尺度负类支持。
4. 首次打开 outer NPZ，但只读取 `fmt_features`、scale、center、block、assigned-row identity；不读取 `metadata_json` 或 `valid_labels`。
5. 写入、关闭并 hash `outer_predictions.npz` 与 `outer_prediction_manifest.json`。
6. 在任何 outer reference member 打开前，重新验证 manifest 自哈希、配置/候选/外层族、预测文件大小与 SHA-256，以及 13 个预测数组的 dtype、shape 和逐数组 SHA-256。
7. 第二次打开 outer NPZ，只读取 `valid_labels` 与用于验证其 hash 的 `metadata_json`，再计算指标。

单元测试用 `allow_pickle=False` 下不可打开的 object-array poison members，证明 prediction projection 没有触碰 outer label/metadata members。

## 6. 指标与停止规则

主指标为 outer family macro F1 与 Average Precision；同时报告 accuracy、Area Under the Receiver Operating Characteristic Curve、balanced accuracy、precision、recall、混淆计数和三类 support 状态。每个 outer family 内等权平均 `dataset×source×block` groups，最终五个 family 再等权平均。

只有非-oracle 五折结果同时满足以下条件，才进入新的 `mainExp_TemplateMatching_4.1`：

- family macro F1 ≥ 0.70；
- 至少 4/5 family F1 ≥ 0.65；
- 没有 family F1 < 0.50；
- macro Average Precision ≥ 0.60；
- macro balanced accuracy ≥ 0.70；
- macro precision、recall 均 ≥ 0.60。

Outer 结果出现后不得修改候选、threshold、support 或宏平均规则；任何修改必须建立新版本。即使达到目标，本验证仍只使用历史上已暴露的八个 flow，因此不是 formal confirmation。

## 7. 运行入口与预期产物

每个 outer fold：

```bash
python scripts/run_verify_scale_conditioned_retrieval_1_1.py \
  --config config/Verify_ScaleConditionedRetrieval_1.1.yaml \
  --expected-config-sha256 f5dbdae08e2e13140245a6a9fd12dba67b4eaf6a7ae1aaea8d600f89a409a6a2 \
  --outer-family half_cylinder \
  --device cuda \
  --output-dir /ibex/user/zhanx0o/pathline-template-matching/Verify_ScaleConditionedRetrieval_1.1/runs/UNIQUE_DIRECTORY
```

`RUN_COMPLETE.json` 必须最后写入。当前本地完整回归为 158/158 tests passed；CUDA 分支仍需由 Ibex job 验证。

因同一账户的其他 FMT GPU array 占满 GPU QOS，允许在完全相同的配置、输入和 outer family 上运行一个 CPU profile 副本。CPU 与 GPU 结果必须分别记录设备和数值 commit，不能把不同设备产生的 outer folds 混入同一个五折 aggregate。CPU profile 只用于尽早取得首折机制证据和 wall time；正式五折仍须使用同一种设备完成。

## 8. 运行前的旧分数 oracle 上限诊断

这项诊断只分析已暴露的 `Other_NegativeDistanceSpatial_1.1` 旧分数，不使用也不约束本验证尚未产生的 scale-conditioned 分数。证据为：

```text
outputs/Other_NegativeDistanceSpatial_1.1_download/
  slurm_51039505_7118af6c17b9/oracle_upper_bound.csv
```

文件 SHA-256 为 `1fa00ea04d00d0a879f390d5a59867ed04d960297a72bda27f416a53b799f26f`；数值实验 commit 为 `7118af6c17b964b5561e6e297609f431f81aa020`。过滤 `input_id=main31_train_family_holdouts_source2` 后，用整数混淆计数按 `2TP/(2TP+FP+FN)` 重算，`masked_gaussian_rank_sigma_1` 的八个 `dataset×block` group oracle F1 宏平均仅为 `0.586170054270344856`；最低组 `halfcylinderRe640/expanded_3_1` 为 `0.475285735490574440`，最高组 `cylinder3d/legacy_2_1` 为 `0.712839506172839506`。

这里的 oracle 比可部署阈值更宽松：八个 group 各自读取本组真值、分别枚举并选择最佳 threshold，不是共享一个 threshold。因此它只支持一个否定结论：**在旧分数的组内排序不变时，继续调 threshold 本身不可能把八组宏平均 F1 推到 0.7。** 它不能否定本验证，因为本验证同时改变了自然负类 library、representation、exact same-scale 检索与 `k`。
