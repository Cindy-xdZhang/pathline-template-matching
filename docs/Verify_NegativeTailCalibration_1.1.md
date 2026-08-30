# `Verify_NegativeTailCalibration_1.1`

状态：数值方法与候选集已在读取任何 `Verify_ScaleConditionedRetrieval_1.1` outer 结果前冻结；实现与 Ibex 运行尚未完成。

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

