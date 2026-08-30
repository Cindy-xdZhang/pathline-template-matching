# `Verify_RawPCANegativeMetric_1.1`

状态：`frozen_pre_run_not_implemented`。本版本已经冻结方法、候选集、输入、
产物和停止规则，但尚未实现、尚未提交 Ibex、尚未运行，因此没有性能结论。

冻结配置：`config/Verify_RawPCANegativeMetric_1.1.yaml`

配置 SHA-256：`6f4718ce6d6385bd0bd5b41a7a04e74cb8f2064fee64097f162999e9eefe6440`

冻结发生在第一次读取 `Verify_PerScaleNegativeMetric_1.1` 的任何 outer
feature、label、prediction、metric 或 summary 之前。冻结时没有读取该父实验的
outer 结果。本版本禁止访问 `tangaroa` 和 `smokeBuoyancy` 的文件、manifest、
cache、feature、label、prediction 或 metric；即使未来成功，也只能作为八个已暴露
train flow 上的 development 证据，不是 formal confirmation。

## 1. 要验证的唯一问题

父版本使用三种固定 FMT representation，并在 inner-family validation 中选择其中
一种。本版本只改变 representation：

```text
{fmt161, real_neighbor36, chirality_all35} 三选一
→ 单一固定的 train-only Raw672 Principal Component Analysis 161D
```

Principal Component Analysis（PCA，主成分分析）在这里是一个无监督线性投影：
只用当前 nested fit families 的 Raw672 feature 拟合 161 个主方向，然后用同一投影
变换 fit、inner query 或 outer query。它不读取标签来拟合主方向。

以下部分完全继承 `Verify_PerScaleNegativeMetric_1.1`：natural-negative library、
逐精确尺度收缩 diagonal variance、fit-negative 尾概率、exact same-scale retrieval、
`k`、spatial sigma、decision rules、nested physical-family split、两级等权宏平均、
tie-break 和停止规则。

本版本不把 Raw-PCA 加入父版本的三个 FMT representation 后共同搜索。那样会同时
改变 representation 和候选数量，无法判断改善究竟来自 Raw-PCA 还是更多选择机会。
本版本只有 `raw_pca161` 一个 representation。

## 2. 输入 cache 合同

唯一输入是 `mainExp_TemplateMatching_3.1` 已认证的32个 train cache：

- input manifest：
  `/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_50998592_260a07ad380d/train_cache_input_manifest.json`
- 文件大小：`24,009` bytes；SHA-256：
  `e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`；
- schema：`pathline_template_matching.long_arc_train_cache_input.v1`；
- row count：`32`；rows content SHA-256：
  `ceb6d0e3fb7a2c90fcaae98583f8d1def7ee75fa7968f38d2821ee3040ae156f`；
- manifest 明确记录 `test_dataset_access=false`；cache commit 为
  `260a07ad380d64fc300cabe8926244e92d8ba04a`，3.1 config SHA-256 为
  `771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e`。

`raw_features` 由 `src/pathline_template_matching/phase21_pipeline.py` 中冻结的
cache builder 产生：先用 `centered_xyz` 把完整 primitive 减去中心 pathline 的第一个
XYZ sample，再把 `7×32×3` 按 C order 展平成672维 `float32`。因此一个有效 cache 的
主数组合同为：

| Member | dtype | shape | 本版本用途 |
|---|---|---|---|
| `raw_features` | `float32` | `[N,672]` | PCA fit/transform |
| `valid_scale_id` | `int32` | `[N]` | exact-scale metric |
| `valid_center_seed_index` | `int64` | `[N]` | 空间 group 与 tie-break |
| `valid_scale_block_index` | `int8` | `[N]` | legacy/expanded block |
| `valid_assigned_row_index` | `int64` | `[N]` | 行身份认证 |
| `valid_labels` | `bool` | `[N]` | PCA 关闭后选择 natural negatives 或评测 |

`fmt_features` 不得由本版本打开。每个 cache 必须先验证 manifest 中的文件大小和
文件 SHA-256；fit cache 还必须把 metadata 中 `raw_features` 的 array SHA-256 与实际
数组重新计算比较。Outer label-free projection 在预测阶段不能打开含正类计数的
`metadata_json`，因此先依赖完整 cache 文件 SHA-256、固定 dtype/shape 和行身份合同；
只有预测完成并认证后，才重新打开 metadata 与 label 做最终核验。

32个 train cache 的已认证总有效行数为 `2,967,612`，按完整 physical family 为：

| Physical family | Dataset 数 | Cache 数 | Valid Raw672 rows |
|---|---:|---:|---:|
| `half_cylinder` | 3 | 12 | 1,309,366 |
| `delta_wing` | 2 | 8 | 589,051 |
| `f22_raptor` | 1 | 4 | 437,257 |
| `channel` | 1 | 4 | 315,580 |
| `boeing_747` | 1 | 4 | 316,358 |
| **Total** | **8** | **32** | **2,967,612** |

这些数只描述已经存在的 train cache population，不是性能结果。

## 3. Nested PCA fit population

Outer unit 与 inner unit 都是完整 physical family，顺序固定为：

```text
half_cylinder, delta_wing, f22_raptor, channel, boeing_747
```

每个 outer fold 有四个 inner folds。一个 inner PCA 只用
`all families − outer family − inner family` 的三个 fit families；final PCA 只用
`all families − outer family` 的四个 nonouter families。PCA population 是这些 fit
families 的**全部 valid Raw672 rows**，不按类别平衡、不抽样，也不打开 label。

当前 outer/inner 的精确 PCA sample counts 冻结如下；实现必须逐项断言：

| Outer | Inner held out | PCA fit rows |
|---|---|---:|
| `half_cylinder` | `delta_wing` | 1,069,195 |
| `half_cylinder` | `f22_raptor` | 1,220,989 |
| `half_cylinder` | `channel` | 1,342,666 |
| `half_cylinder` | `boeing_747` | 1,341,888 |
| `half_cylinder` | **final** | 1,658,246 |
| `delta_wing` | `half_cylinder` | 1,069,195 |
| `delta_wing` | `f22_raptor` | 1,941,304 |
| `delta_wing` | `channel` | 2,062,981 |
| `delta_wing` | `boeing_747` | 2,062,203 |
| `delta_wing` | **final** | 2,378,561 |
| `f22_raptor` | `half_cylinder` | 1,220,989 |
| `f22_raptor` | `delta_wing` | 1,941,304 |
| `f22_raptor` | `channel` | 2,214,775 |
| `f22_raptor` | `boeing_747` | 2,213,997 |
| `f22_raptor` | **final** | 2,530,355 |
| `channel` | `half_cylinder` | 1,342,666 |
| `channel` | `delta_wing` | 2,062,981 |
| `channel` | `f22_raptor` | 2,214,775 |
| `channel` | `boeing_747` | 2,335,674 |
| `channel` | **final** | 2,652,032 |
| `boeing_747` | `half_cylinder` | 1,341,888 |
| `boeing_747` | `delta_wing` | 2,062,203 |
| `boeing_747` | `f22_raptor` | 2,213,997 |
| `boeing_747` | `channel` | 2,335,674 |
| `boeing_747` | **final** | 2,651,254 |

已经存在的 `mainExp_TemplateMatching_3.1/preprocessing_artifacts.npz` 用全部八个 train
flow 拟合。虽然它没有使用 Tangaroa/Smoke，但在本实验的 family-held-out 定义下仍看过
当前 inner/outer family 的 Raw feature distribution，因此会造成 transductive leakage，
禁止复用。2.1 的 PCA 也因 primitive population 和 physical-family exposure 不同而禁止
复用。

## 4. 确定性 PCA 数值定义

实现必须复用或逐项等价实现
`phase21_pipeline.py` 中已经验证的 `StreamingCovariancePCA` 与
`fit_streaming_covariance_pca`，不得改成 randomized、incremental 或 approximate PCA。

Cache 迭代顺序固定为 family order、family 内 dataset order、source ordinal 升序、
cache 内原始 valid-row order。算法为：

1. 第一遍以 `float64` 累计672维 feature sum 和 `int64` row count，得到
   `mean64 = feature_sum / N`。
2. 第二遍仍按同一 cache 顺序，每个 cache 用连续 `8192` 行 chunk；先把 Raw672 转成
   `float64`，减 `mean64`，再以 `float64` 累计
   `scatter += centered.T @ centered`。两遍 row count 必须相等。
3. 用 `0.5×(scatter+scatter.T)` 强制数值对称，调用 `numpy.linalg.eigh`。
4. 以 stable `argsort(-eigenvalue)` 降序。若最小 eigenvalue 小于
   `-max(1e-12, max(largest_eigenvalue,0)×1e-10)`，立即失败；容差内负值固定截为0。
5. 取前161个 eigenvector，并转置为 `[161,672]` components。
6. 每个 component 取第一个绝对值最大的 loading 作为 pivot；若 pivot 为负，整行乘
   `-1`，若 pivot 为0则符号视为 `+1`。
7. Singular values 为前161个截断后 eigenvalue 的平方根；explained variance ratio
   为每个前161 eigenvalue 除以全部672个截断 eigenvalue 之和。总和为0时 ratio 全0。

序列化时 mean 与 components 为 `float32`，singular values 与 explained variance
ratio 为 `float64`。Query transform 精确为：

```text
contiguous float32((raw_float32 - mean_float32) @ components_float32.T)
```

不做 PCA whitening，也不做额外 global post-PCA standardization。之后只使用父版本
已经冻结的 fit-negative exact-per-scale scaler。由于 query 与 exact-scale negative
library 使用同一 PCA 和同一逐尺度 mean，PCA global mean 会在同尺度差分中代数抵消；
PCA subspace 与方向仍会改变 diagonal metric，因此仍是需要实验检验的机制。

## 5. Negative metric 与 tail calibration

PCA 关闭后，才允许打开相应 fit families 的 `valid_labels`。只有 natural negative rows
进入 library。对 PCA161 的每个 coordinate，逐 exact scale 使用父版本的 `ddof=0`
local variance，并以 `lambda=64` 在 variance domain 向同 block、逐尺度分别中心化后的
within-scale pooled prior 收缩；block-other 为空时才允许 global-other，其他尺度也为空
时才 local-only。Standard deviation 严格小于 `1e-12` 时替换为1。

Exact same-scale retrieval、`k={1,5,15,31}`、显式 leave-one-out self exclusion、
duplicate-zero-distance 保留、tail probability/anomaly、local/block/global reference
收缩、support 和 calibration mode 全部逐字继承父版本，不在本版本重新选择。

## 6. 唯一候选集

Representation 固定为 `raw_pca161`，因此候选数为：

```text
1 representation × 4 k × 5 sigma × (1 fixed-top-5% + 50 thresholds)
= 1,020 candidates
```

Spatial sigma 固定为 `0, 0.5, 1.0, 1.5, 2.0` grid cells。Threshold 固定为
`0.50–0.99`、步长 `0.01`。Fixed-top rule 仍取完整
`dataset×source ordinal×scale block` group 的 `ceil(5%×all valid rows)`，只从有
calibration support 或 spatial imputation 且 score 严格为正的行中选择；tie-break 为
score 降序、center index 升序。

选择仍先对每个 inner family 内的 dataset×source×block groups 等权，再对四个 inner
families 等权。Tie-break 顺序固定为 F1、Average Precision、balanced accuracy、
precision、recall、candidate ID 字典序。Outer label 或 metric 不得进入选择。

## 7. Artifact 与访问门禁

Final PCA 新增两个不可覆盖 artifact：

| File | 固定内容 |
|---|---|
| `final_pca.npz` | `mean_float32[672]`、`components_float32[161,672]`、`singular_values_float64[161]`、`explained_variance_ratio_float64[161]`、三个 scalar `sample_count_int64/input_width_int32/output_width_int32` |
| `final_pca_manifest.json` | fit family/cache population、Raw hashes、solver、全部 array dtype/shape/hash、NPZ hash、自哈希和访问状态 |

PCA schema 固定为 `pathline_template_matching.raw_pca161.v1`。Manifest 必须绑定本
实验 config、32-cache input manifest、parent cache commit/config、outer family、按顺序的
fit families 与 caches、每 cache 的 file/raw-array SHA 和 row count、总 sample count、
完整数值规则，并明确记录 `labels_opened_for_pca=false` 与
`outer_raw_features_opened=false`。

每个 inner fold 只保留 fit audit，不持久化四套大模型；`inner_fit_audits.json` 的固定
fit count 从父版本的 `4 inner×3 representations=12` 改为本版本的
`4 inner×1 representation=4`。每项必须绑定 PCA sample count、PCA array hashes、
fit cache identities、PerScale scaler audit 与 tail calibrator audit。

一个 outer fold 的固定17个文件为：

```text
inner_group_metrics.csv
inner_candidate_summary.csv
inner_fit_audits.json
final_pca.npz
final_pca_manifest.json
final_per_scale_scaler.npz
final_per_scale_scaler_manifest.json
final_tail_calibration.npz
final_tail_calibration_manifest.json
selected_candidate.json
outer_predictions.npz
outer_prediction_manifest.json
outer_group_metrics.csv
outer_summary.json
outer_reference_access_audit.json
result_manifest.json
RUN_COMPLETE.json
```

其中 `result_manifest.json` 必须认证15个非自身、非 completion artifacts。
`selected_candidate.json` 必须同时绑定 final PCA、PerScale scaler、tail calibrator 和
完整 inner evidence 的 file/manifest SHA-256。Scaler、calibrator、outer prediction 和
result manifests 也必须直接或经认证链绑定 PCA identity。

Final/outer 访问顺序固定为：

1. 只从四个 nonouter families 拟合 final PCA；原子写入、关闭、逐成员认证并重建。
2. 用认证后的 PCA transform nonouter rows，拟合并认证 final PerScale scaler 与 tail
   calibrator。
3. 写入并认证绑定全部 inner/final artifacts 的 selected candidate。
4. 此后才可第一次打开 outer `raw_features`、scale、center、block 与 assigned-row
   identity；仍禁止 label 和 metadata。
5. 写入并逐数组认证 outer predictions 与 manifest。
6. Postvalidation 必须 fresh reload label-free outer Raw672，用认证 PCA transform，重建
   scaler/calibrator，重算 score 和 prediction 并逐数组完全一致。
7. 只有第6步通过后，才允许打开 outer `valid_labels` 与 `metadata_json` 并计算指标。

Aggregator 必须再次执行相同的 file-set、commit/config/input、PCA/scaler/calibrator、
label-free prediction 与 fresh-label metric 认证；不能只汇总 runner 写出的 summary。

## 8. 成功、否证与提前停止

完整五折必须同时满足：

| 条件 | 门槛 |
|---|---:|
| Five-family macro F1 | `>=0.70` |
| Family F1 `>=0.65` 的数量 | `>=4/5` |
| Minimum single-family F1 | `>=0.50` |
| Five-family macro Average Precision | `>=0.60` |
| Five-family macro balanced accuracy | `>=0.70` |
| Five-family macro precision | `>=0.60` |
| Five-family macro recall | `>=0.60` |

任何一项失败都表示本版本没有达到目标。提前停止只能由数学证书触发：任一已完成
family F1 `<0.50`；已有两个 family F1 `<0.65`；或把全部未运行 family 的相应指标设为
理论上限1后，five-family macro 仍不可能达到门槛。其他情况必须继续完整五折。

本版本失败只否证“固定 Raw-PCA161 加父 PerScale NegativeTail”这一项，不结束项目追求
F1 `0.7–0.8` 的总目标，也不能反向证明 FMT representation 正确。

## 9. 资源预估与已知限制

32个 train cache 文件共 `8,198,604,568` bytes。全部 Raw672 若同时常驻约
`7.429 GiB`；一个 final nonouter population 为 `4.15–6.64 GiB` Raw672，投影后为
`0.99–1.59 GiB` PCA161。因为压缩 NPZ 不能真正 memory-map，runner 必须按 cache
stream PCA passes，并及时释放 Raw arrays，禁止无必要地同时常驻全部 Raw672。

直接执行四个 inner PCA 加一个 final PCA 时，每个 outer fold 的 scatter 计算相当于
处理四倍 nonouter row population，约 `3.0–4.8×10^12` 个 multiply-accumulate terms。
这是资源估计，不是 wall-time 结果。首次 Ibex profile 应申请32 CPU、128 GB、12 h，
用 GNU `time -v` 分别记录 PCA、runner postvalidation 与 aggregate 的 elapsed 和 MaxRSS；
不得因排队或运行时间改用泄漏的全八族 PCA。

PCA 最大化 fit data 的总体方差，不保证保留区分 IVD-p95 正类的低方差方向；161维截断
可能改善噪声，也可能删除涡旋相关信号。因此当前只有“可运行且无 nested leakage”的
依据，没有“会优于 FMT 或达到 F1 0.7”的结论。任何 PCA width、whitening、family
weighting 或 fit-negative-only PCA 的尝试都必须另建版本，不能在本版本看到结果后修改。
