# `Verify_NegativeTailCalibration_1.1`

状态：完整五个 outer-family folds 已认证；冻结成功规则失败，本版本停止并进入结果可见前已冻结的 `Verify_PerScaleNegativeMetric_1.1`。

冻结配置：`config/Verify_NegativeTailCalibration_1.1.yaml`

配置 SHA-256：`4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b`

## 1. 第一性原理问题

IVD-p95 的正类约占每个流场时刻的最高 5%，而 FMT 距离在不同 `dx×ds×arc length` 尺度上没有共同物理单位。`Verify_ScaleConditionedRetrieval_1.1` 已禁止跨尺度最近邻，但仍把 1000 个尺度的距离混在一个 query group 内做百分位 rank。这有两个后果：

1. 每个 query group 的 score 被强制变成近似均匀分布，绝对异常强度被删除；
2. 同一 rank 可能来自完全不同的尺度内负类距离分布，不能表示“相对该尺度背景有多异常”。

旧 `Other_NegativeDistanceSpatial_1.1` 的更宽松 oracle 诊断已显示：让八个 group 各自读取真值并分别选 threshold，`sigma=1` 旧分数的宏平均上限仍只有 `0.586170054270344856`。因此只调 threshold 不能达到 0.7；必须改变 score 的跨尺度含义或尺度内排序。本验证只检验前者，不同时修改 descriptor。

## 2. 唯一数值变化

相对 `Verify_ScaleConditionedRetrieval_1.1`，唯一改变是：

```text
query group 内 supported distance percentile rank
→ fit-negative-only scale-tail anomaly
```

以下内容全部保持不变：自然负类 library、negative-only global mean/std、三个 FMT representation、exact same-scale Euclidean distance、`k={1,5,15,31}`、五个 spatial sigma、fixed top-5%、threshold `0.50–0.99`、nested physical-family split、宏平均、tie-break 和停止规则。候选数仍为 3060。

本版本禁止同时加入 Principal Component Analysis（PCA，主成分分析）、逐尺度 feature scaler、kinematic feature、跨尺度检索或新的候选网格。这样结果只能回答“负类尾概率校准是否修复跨尺度 score 不可比”。

## 3. Fit-negative 尾概率

全局 scaler 仍只拟合一次完整 fit-negative population。对尺度 `s` 的标准化负类 feature 集合 `L_s`，为每个负类行显式排除它自己，再计算第 `k` 近邻距离 `T_s,k,i`。实现必须将当前 row 的距离设为正无穷；不能简单取包含自身搜索的第 `k+1` 项，因为重复 feature 会使零距离 tie 中的自身位置不确定。

对一个升序 reference 数组 `A` 和 query distance `d`：

```text
tail_probability(d) = (1 + count(t in A where t >= d)) / (|A| + 1)
tail_anomaly(d)     = count(t in A where t <  d)       / (|A| + 1)
```

使用 `>=` 使相同距离获得更保守的较大尾概率。距离越大，`tail_anomaly` 越大，因此现有 Average Precision、Area Under the Receiver Operating Characteristic Curve 与 threshold 方向不变。这里的 leave-one-out 只排除当前邻居行；不会为每一行重新拟合 scaler。

## 4. 固定收缩与支持

Scale ID `0–999` 为 legacy block，`1000–1999` 为 expanded block。局部 reference 为同尺度的全部 leave-one-out 距离；block-other reference 汇集同 block 其他尺度，必须排除当前尺度；只有 block-other 为空时才允许回退到两个 block 其他尺度的 global-other reference。

收缩强度固定为 `lambda=64`。这是 3.1 设计中每个 `source×block×scale` 的 assigned-row 数，不由任何标签或结果选择。局部 reference 数为 `m` 时，局部权重固定为 `m/(m+64)`。

| Fit-negative 数 `n_s` | Exact query 第 k 距离 | Local reference | 处理 |
|---:|---|---|---|
| `n_s < k` | 不可计算 | 不可用 | retrieval unsupported；不得用 pooled reference 伪造距离 |
| `n_s = k` | 可计算 | 不可用 | 只用 block/global reference 回退 |
| `n_s >= k+1` | 可计算 | `n_s` 个值 | local 与 broader reference 固定收缩 |

Calibration mode 固定为 `0=no calibration`、`1=local/block shrink`、`2=local/global shrink`、`3=local only`、`4=block fallback`、`5=global fallback`。没有 reference 时保存 `tail_probability=1`、`tail_anomaly=0` 并标记 calibration unsupported；该行不能直接判正。

## 5. 空间处理与分类

禁止再次对 query distance 或 tail anomaly 做 group rank。`tail_anomaly` 直接进入：

```text
G(tail_anomaly × calibration_support) / G(calibration_support)
```

`sigma=0` 时 score 只依赖当前 primitive 与 fit-negative calibrator；正 sigma 仍依赖完整 `dataset×source×block` grid。Fixed top-5% 也依赖完整 query group。因此 tail calibrator 可以逐 primitive 应用，但完整候选集合仍是 transductive classifier，不能称独立逐 primitive classifier。

## 6. Nested 信息隔离

每个 inner fold 只用三个 fit families 的自然负类拟合 scaler、negative library 与 calibration references。Inner validation family 的标签只参与候选指标。选择完成后，用四个 nonouter families 重新拟合 final model，并在打开任何 outer feature member 前写完、关闭和认证：

- `final_tail_calibration.npz`
- `final_tail_calibration_manifest.json`
- `selected_candidate.json`

随后 outer 第一次 projection 只允许读取 feature、scale、center、block 与 row identity。`outer_predictions.npz` 和 manifest 完整关闭、逐文件与逐数组 SHA-256 验证后，才允许第二次 projection 打开 `valid_labels` 和 `metadata_json`。

## 7. 评价与停止规则

五个 outer physical families、两级等权宏平均与停止规则完全继承父验证：family macro F1 至少 0.70、至少 4/5 family F1 至少 0.65、任何 family 不低于 0.50，同时 Average Precision 至少 0.60、balanced accuracy 至少 0.70、precision 与 recall 均至少 0.60。

即使通过，本版本仍只使用历史上已暴露的八个 train flows，只能成为 `mainExp_TemplateMatching_4.1` 候选，不能称 formal confirmation。

## 8. 预注册限制

- Tail transform 在同一尺度内是单调变换，不能修复 FMT feature 本身错误的尺度内排序。
- `n_s=k` 时 block/global fallback 比较跨尺度 absolute distance，是已冻结近似，不具备严格经验覆盖保证。
- Held-out physical family 存在 distribution shift，因此不能宣称 conformal coverage。
- Exact leave-one-out 增加约 `O(D sum_s n_s^2)` 的计算；若 Ibex profile 证明成本不可接受，必须另建版本使用预先冻结的 deterministic subsample，不能在 1.1 中静默近似。

## 9. 实现与部署门禁

- 核心 calibrator：commit `e9380e4`，`src/pathline_template_matching/negative_tail_calibration.py`。
- 认证 runner：初始实现 commit `907a371`，最终部署 hardening commit `bd3037c`；文件 SHA-256 `ab62453215a7ecf508aad50e94e244093d898c2baa148908c215e71ce994b6d5`。
- 聚合认证器：commit `bd3037c`，`scripts/aggregate_verify_negative_tail_calibration_1_1.py`，文件 SHA-256 `212e402cf287f780a0e8def4949a38dfde1d96d59b27ad61d50c35dff7730e58`。
- Ibex CPU profile wrapper：commit `bd3037c`，文件 SHA-256 `e6921f54b702cc29a920db2f8e919d942e6249bf8e5352b428ecb185ae3fbc0e`；固定 runner/aggregator/config/input hashes，并分别用 GNU `time -v` 记录数值、postvalidation 与聚合认证耗时和峰值内存。
- 2026-08-30 最终本地完整门禁：203/203 tests 通过，耗时 137.510 秒；聚合专项 7/7、`py_compile`、`git diff --check` 与 Git Bash `bash -n` 均通过；聚合器独立安全复审无剩余 P1/P2。
- Inner selection 由落盘的 3060 个候选、122,400 条 group 记录和 12 个 fit audit 重新认证；最终 selected summary 直接取自认证后的 CSV 数值，禁止混用内存副本。
- Outer label gate 会重新加载完整 label-free outer scope，以认证后的 calibrator 重算 distance、tail、support、spatial score 与 prediction，并逐数组核对；完成后才允许读取 label member。
- CPU 环境审计在 CUDA 可见但 requested device 为 CPU 时不会把 CPU device 传给 CUDA API；该边界已有独立回归测试。
- 完整五折 array wrapper：numerical commit `e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c`；文件 SHA-256 `62283450f94fc9070452caa5defec27d12b980f1d881a6cacae7904a564ebb52`。每个 task 强制验证同一个 40 位 expected commit，防止共享 clone 在等待期间切换版本。
- Complete-five-fold wrapper SHA-256 `ade47a71ce913d19986aaf1ddea9d0ce8f249b93225fe53e4fb5fb8a69a119b8`；以 `afterok` 依赖启动，聚合器拒绝混合 commit、config 或 input manifest。
- Ibex jobs `51059479_[0-4]` 实测最长 wall time `00:12:50`、最高 batch MaxRSS `21,534,488K`；job `51059491` 的 fresh 五折聚合 wall time `00:02:59`、batch MaxRSS `12,076,244K`。因此 12 h / 128 GB 请求有充分余量。

本节只记录实现与运行门禁；性能结果分别保留在第 11、12 节。

## 10. 首折认证与提前停止合同

首个且唯一允许的单折 profile 固定为 `half_cylinder`。聚合器必须由命令行给出 40 位 expected numerical commit，并在打开任何 cache、outer feature 或 label 前认证 commit、config、input manifest、13 个 fold 文件和 11 个 result artifacts。随后重新执行 label-free prediction；只有该链逐数组认证后才允许打开 labels，并以 fresh evaluation 逐字段复算 group CSV 与 summary。

单折认证输出固定为五个不可覆盖文件：

```text
outer_family_summary.csv
single_fold_authentication_report.json
early_stop_certificate.json
aggregate_manifest.json
AGGREGATE_COMPLETE.json
```

Schema 分别使用 `pathline_template_matching.negative_tail_single_fold_authentication_report.v1`、`pathline_template_matching.negative_tail_early_stop_certificate.v1`、`pathline_template_matching.negative_tail_aggregate_manifest.v1` 与 `pathline_template_matching.negative_tail_aggregate_complete.v1`；completion 必须最后写入，且发布前后重验其余文件 size、SHA-256、自哈希与完整集合。

单折报告禁止推断五折成功。`stop_version=true` 只允许在已认证部分结果已数学证明最终不可能满足冻结规则时出现：任一已完成 family F1 `<0.50`；已有两个 family F1 `<0.65`；或把全部未运行 family 的相关指标设为理论上限 1 后，五-family macro 仍低于门槛。否则必须继续其他 folds。完整成功判断只允许 `complete-five-fold` 模式在五个唯一 physical families 齐全时产生。

## 11. 首折认证结果：继续，不作五折结论

Ibex job `51058757` 使用 numerical commit `a076240b76dac8a598fc785916c48dc0edc65398`，outer family 为 `half_cylinder`。作业 `COMPLETED 0:0`，elapsed `00:11:51`；203/203 job 内测试在 151.720 秒内通过。数值 runner、label-free postvalidation、fresh-label single-fold aggregate 分别耗时 7:16.22、0:37.89、1:03.84，三段 MaxRSS 分别为 8,596,984、3,150,936、3,394,096 KiB；Slurm batch MaxRSS 为 13,432,572 KiB。

认证选择为 `chirality_all35 / k=15 / sigma=1.0 / fixed top-5%`。`1,309,366` 个 query 全部具有 retrieval 与 calibration support，无 spatial imputation。

| 指标 | `half_cylinder` family macro |
|---|---:|
| Accuracy | 0.949973 |
| Average Precision | 0.572055 |
| F1 | 0.537691 |
| Balanced Accuracy | 0.742397 |
| Precision | 0.575906 |
| Recall | 0.507304 |

独立下载复核通过精确 13 个 fold 文件、11 个 result artifacts、5 个 aggregate 文件、全部文件 SHA-256 与 JSON 自哈希。Result manifest SHA-256 为 `f64d675896cd83a24debb7d7902425ce04f75d7f07a497e79f102d01c8f5ddf9`；aggregate completion SHA-256 为 `1559a5bbf595cf6e4d696c773a57573ee0ca2b7a0d10362df1e0c8f39bfb13a6`。

冻结的单-family 下限为 F1 0.50；本折为 0.537691，因此 `early_stop_certificate.json` 认证 `stop_version=false`。这不是“方法通过”：Average Precision、precision、recall 和 F1 仍低于完整五折目标，且剩余四折未知。唯一合法结论是继续运行其余四个 physical families。

## 12. 完整五折认证结果：停止本版本

方案 A 使用同一 numerical commit `e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c` 重跑全部五折。Ibex array `51059479_[0-4]` 的五个 task 均 `COMPLETED 0:0`；task `0–4` 依次为 `half_cylinder, delta_wing, f22_raptor, channel, boeing_747`，elapsed 分别为 `00:04:44, 00:04:52, 00:05:37, 00:12:38, 00:12:50`。带 `afterok:51059479` 的 complete-five-fold aggregator job `51059491` 于最后一折结束后一秒启动，并在 `00:02:59` 后 `COMPLETED 0:0`。

聚合器对每折重新认证精确 13 个文件、11 个 result artifacts、完整 3060-candidate inner evidence、selected candidate、final calibrator 与 label-free outer prediction；只有这些全部通过后才打开 outer labels，并 fresh 重算 group metrics、family summary 与停止规则。全部 query 的 retrieval/calibration support 都为 100%，spatial imputation 与 unimputable fraction 均为 0，因此本次失败不是 coverage 或 fallback 缺失。

| Outer family | Selected candidate | F1 | Average Precision | Balanced Accuracy | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `half_cylinder` | `chirality_all35 / k=15 / sigma=1 / top-5%` | 0.537691 | 0.572055 | 0.742397 | 0.575906 | 0.507304 |
| `delta_wing` | `real_neighbor36 / k=31 / sigma=1 / top-5%` | 0.770469 | 0.889834 | 0.930098 | 0.693821 | 0.876126 |
| `f22_raptor` | `real_neighbor36 / k=15 / sigma=0.5 / top-5%` | 0.513438 | 0.517664 | 0.759232 | 0.486965 | 0.545332 |
| `channel` | `real_neighbor36 / k=5 / sigma=1 / top-5%` | 0.253965 | 0.188286 | 0.639626 | 0.210811 | 0.320059 |
| `boeing_747` | `real_neighbor36 / k=1 / sigma=1 / top-5%` | 0.626794 | 0.711794 | 0.862038 | 0.554014 | 0.747191 |
| **Family macro** | — | **0.540472** | **0.575927** | **0.786678** | **0.504303** | **0.599202** |

五折 family-macro accuracy 为 `0.957953`、Area Under the Receiver Operating Characteristic Curve 为 `0.946831`；由于正类稀少，这两个高值不能替代 F1、Average Precision、precision 与 recall。冻结规则中只有 macro balanced accuracy `>=0.70` 通过；macro F1 `>=0.70`、4/5 family F1 `>=0.65`、minimum family F1 `>=0.50`、macro Average Precision/precision/recall `>=0.60` 均失败。实际只有 1/5 family F1 `>=0.65`，最低 `channel` F1 为 `0.253965`，因此 `all_success_conditions_pass=false`。

首个旧 commit profile 与本次新 commit 的 `half_cylinder` 指标逐项相同，支持该折结果可重复；但完整五折显示明显的 physical-family heterogeneity：`delta_wing` 达到目标区间，而 `channel` 排序崩溃。稳定结论是负类尾概率校准修复了跨尺度 score 的可比性，但没有普遍修复同一尺度内的 FMT feature weighting/排序。本版本不能达到目标，下一步按结果可见前已冻结的顺序检验逐尺度 negative variance metric；不得根据这些 outer 指标修改其 `lambda=64`、候选集或停止规则。

完整证据已下载到 `outputs/Verify_NegativeTailCalibration_1.1_jobs51059479_51059491_download`。本地验收精确通过 69 个结果文件、55 个 artifact identity、42 个 JSON 自哈希，总字节 `1,725,432,861`；独立从每折 `outer_group_metrics.csv` 按冻结 group 等权口径重算 family 与五折 macro，逐项一致。关键 SHA-256：

- `AGGREGATE_COMPLETE.json`: `e826fce87b59fa1931509a601c3374aa4f03756c0ab21499963c80758a030db6`
- `aggregate_manifest.json`: `2628da64a95ef02ed85cb8c9f8e759d2a0aac41faf52077ae54513409a998a8e`
- `aggregate_summary.json`: `486435f0e491726b28180ddcfcf3dc5e67c1a3b909bc5cff78de6e99af26f094`
- `outer_family_summary.csv`: `ea031d868d3c4dbacd7d8f87aa0c8fcda3c678a113a3aaa8a3784de9df4d6e81`
