# `Verify_PerScaleNegativeMetric_1.1`

执行状态：`fold_array_completed_aggregation_failed_pre_metric_acceptance`。配置中的预注册字段仍保持 `status: frozen_pre_run_not_implemented`，以保留冻结 SHA-256；方法、候选集与输出认证合同是在读取任何 `Verify_NegativeTailCalibration_1.1` outer 指标前冻结的。五折 array `51063738` 的五个 task 均已完成，但依赖聚合 `51063753` 在 fresh replay 中因 `authenticated spatial score does not match tail transform` 失败。尚未读取或接受任何 outer metrics，因此仍无性能结论。

实现路径：`src/pathline_template_matching/per_scale_negative_metric.py`、`scripts/run_verify_per_scale_negative_metric_1_1.py`、`scripts/aggregate_verify_per_scale_negative_metric_1_1.py`、`ibex/verify_per_scale_negative_metric_1.1_all_folds.sh` 与 `ibex/verify_per_scale_negative_metric_1.1_aggregate_five.sh`。提交前本地完整标准库回归为 220 项；numerical commit 为 `809ffa3b9490ca4f5b0817d77759b5d88cce628c`。实际设备和终态见 `docs/ibex_run_registry.md`；未经 fresh-replay 认证的 fold 结果不得转写为指标或结论。

冻结配置：`config/Verify_PerScaleNegativeMetric_1.1.yaml`

配置 SHA-256：`b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79`

## 1. 第一性原理问题

`Verify_NegativeTailCalibration_1.1` 使不同尺度的距离经过 fit-negative 尾概率后可比较，但尾概率变换在同一尺度内是单调的，不能改变该尺度内 query 与 negative library 的距离次序。其父度量使用所有 fit-negative rows 拟合一组 global diagonal population variance；若同一 FMT feature 在不同 pathline 尺度上的自然波动不同，这组全局方差可能给某一尺度内不稳定的 feature 过高权重，并给该尺度内稳定而有区分力的 feature 过低权重。

本验证只回答一个问题：用 fit-negative 数据为每个精确 scale 拟合收缩后的 diagonal within-scale population variance，是否能改善同尺度 Euclidean distance 的排序以及 held-out-family 指标？它不改变 FMT representation，也不引入正类监督或 query-batch 统计。

## 2. 唯一数值变化

相对 `Verify_NegativeTailCalibration_1.1`，唯一变化是：

```text
global negative-only diagonal population variance
→ fit-negative exact-per-scale shrunk diagonal within-scale population variance
```

以下内容完全继承父验证：fit-negative 尾概率定义与 leave-one-out self exclusion、三个 FMT representation、exact same-scale retrieval、`k={1,5,15,31}`、五个 spatial sigma、fixed top-5%、threshold `0.50–0.99`、nested physical-family split、两级等权宏平均、tie-break 和停止规则。候选仍为

```text
3 representations × 4 k × 5 sigma × (1 fixed-top-5% + 50 thresholds) = 3060.
```

`lambda=64` 是冻结常数，不扫描 lambda、metric 或 scaler 网格。本版本禁止同时加入 Principal Component Analysis（PCA，主成分分析）、learned metric、kinematic feature、descriptor 修改、跨尺度检索或任何候选网格变化。

## 3. Fit-negative 逐尺度 diagonal variance

以下拟合对每个 inner fit 独立执行；final fit 则只用四个 nonouter families 重拟合。输入只允许相应 fit families 的自然负类。对 representation 的每个 feature coordinate 独立应用以下公式。

设精确尺度 `s` 的 fit-negative rows 为 `x_(s,i)`，数量为 `n_s>0`。Local mean 和 `ddof=0` population variance 固定为：

```text
mu_s = (1 / n_s) sum_i x_(s,i)
v_s  = (1 / n_s) sum_i (x_(s,i) - mu_s)^2
```

Scale ID `0–999` 为 legacy block，`1000–1999` 为 expanded block。令 `B(s)` 为同 block 中除 `s` 外、实际具有 fit-negative rows 的尺度。Block-other prior 必须先分别减去每个其他尺度自己的 local mean，再 pooling within-scale residual：

```text
v_block_other(s)
  = [sum_(t in B(s)) sum_i (x_(t,i) - mu_t)^2]
    / [sum_(t in B(s)) n_t].
```

禁止先把其他尺度合并后减一个共同均值，也禁止把尺度均值之间的差异放入 prior variance。只在 `sum_(t in B(s)) n_t = 0` 时，才允许使用两个 block 中全部其他尺度的 global-other prior；global-other 使用完全相同的 per-scale centering 与 pooled within-scale residual 公式，仍排除当前尺度。若 block-other 和 global-other 都为空，则记录 `local_only`，不得伪造 broader rows。

有 broader prior `v_prior(s)` 时，冻结收缩为：

```text
w_s          = n_s / (n_s + 64)
v_shrunk(s)  = w_s v_s + (1 - w_s) v_prior(s)
std_raw(s)   = sqrt(v_shrunk(s))
std_eff(s,j) = 1,                 if std_raw(s,j) < 1e-12
               std_raw(s,j),      otherwise.
```

收缩必须先在 variance 域完成，再开平方；不能对 standard deviation 直接插值。严格小于 `1e-12` 才替换为 `1`。若没有 broader prior，`local_only` 的 `v_shrunk(s)=v_s`，同时保存该 fallback mode。

Query 与 library 必须使用同一份 fit-only exact-scale `mu_s` 和 `std_eff(s)`：

```text
z_s(x) = (x - mu_s) / std_eff(s).
```

对同一尺度的 Euclidean distance，`mu_s` 在差分中理论上严格相消：

```text
z_s(q) - z_s(l) = (q - l) / std_eff(s).
```

因此本机制来自逐尺度 effective standard deviation，而不是 mean shift。Mean 仍须序列化，以保证 query/library transform 可重建并能被逐字段认证。

## 4. 无 local row 与 retrieval/calibration 支持

若 `n_s=0`，该尺度没有可拟合的 mean、variance 或 negative library。Block/global prior 只能为已有 local library 的尺度提供 variance 收缩，绝不能生成 synthetic library row；该尺度的 raw retrieval 必须标为 unsupported。序列化表可以为缺失尺度保存明确的未使用 numeric placeholder，但 `local_support=false` 与 `mode=no_local_rows` 必须控制运行时行为。

父验证的 `n_s` 与 `k` 支持合同不变：

| Fit-negative 数 `n_s` | Exact query 第 k 距离 | Local tail reference | 处理 |
|---:|---|---|---|
| `n_s < k` | 不可计算 | 不可用 | retrieval unsupported；不得用 pooled row 或 pooled distance 伪造第 k 距离 |
| `n_s = k` | 可计算 | 不可用 | tail calibration 只用 block/global reference 回退 |
| `n_s >= k+1` | 可计算 | `n_s` 个 leave-one-out 值 | local 与 broader tail reference 按父验证固定收缩 |

每个 fit-negative row 的 tail calibration distance仍只排除 self row，并保留其他重复 feature 形成的零距离邻居。Per-scale scaler 在完整 fit-negative population 上只拟合一次；leave-one-out 不得逐行重拟合 mean、variance 或 standard deviation。Tail probability、anomaly score、`lambda=64` tail-reference shrinkage、fallback 顺序与 calibration modes 均逐字段继承 `Verify_NegativeTailCalibration_1.1`。

## 5. Nested 信息隔离与序列化

Inner fold 只能用三个 fit families 的自然负类拟合 per-scale scaler、negative library 与 tail references。Inner validation family 的 feature 只能用于预测，label 只能用于候选评价；outer family 不得参与选择。

选择结束后，必须用四个 nonouter families 重拟合 final scaler、library 与 tail calibrator。打开任何 outer feature member 前，以下文件必须全部写入临时文件、关闭、原子发布并认证：

- `final_per_scale_scaler.npz`
- `final_per_scale_scaler_manifest.json`
- `final_tail_calibration.npz`
- `final_tail_calibration_manifest.json`
- `selected_candidate.json`

Final scaler 必须保存完整 2000-scale 状态：scale ID、local/block-other/global-other row counts、local support、scaler mode、local mean、local variance、prior variance、shrunk variance 与 effective standard deviation。Manifest 必须记录 config/input manifest、fit family set、representation、candidate identity、每个数组的 name/dtype/shape/SHA-256，以及 support/mode counts。`selected_candidate.json` 必须绑定 final scaler manifest 与 final tail calibration manifest 的 SHA-256。

Outer gate 继承父验证并增加 scaler 认证：上述 final artifacts 未完成关闭、哈希和逐字段重建验证前，禁止打开 outer feature member。`outer_predictions.npz` 与 manifest 完成认证后，才允许第二次 projection 打开 outer `valid_labels` 和 `metadata_json`。

## 6. 空间处理、评价与停止规则

Tail anomaly 仍直接进入 calibration-support-mask-normalized Gaussian：

```text
G(tail_anomaly × calibration_support) / G(calibration_support).
```

禁止 query-group rank。`sigma=0` 只依赖当前 primitive 与 fit model；正 sigma 和 fixed top-5% 仍依赖完整 `dataset×source×block` query group，所以完整候选不能称独立逐 primitive classifier。

五个 outer physical families、两级等权宏平均与停止规则均不变：family macro F1 至少 0.70、至少 4/5 family F1 至少 0.65、任何 family 不低于 0.50，同时 Average Precision 至少 0.60、balanced accuracy 至少 0.70、precision 与 recall 均至少 0.60。Outer 结果出现后禁止修改 threshold、variance 公式、fallback 或候选集。

即使通过，本版本仍只使用历史上已暴露的八个 train flows，只能成为后续主实验候选，不能称 formal confirmation。

## 7. 预注册限制

- 本版本只改变 diagonal feature weighting，不引入 coordinate covariance；相关 feature 的重复计权仍可能存在。
- Per-scale mean 理论上在 exact same-scale distance 中相消；若观测到结果依赖 mean shift，说明实现违反了同尺度或同 scaler 合同。
- `n_s=1` 时 local variance 为零，结果主要由冻结的 broader prior 决定；这不是新增 library support。
- 本版本在任何 `Verify_NegativeTailCalibration_1.1` outer 指标可见前冻结。当前实现已完成；首次 Ibex fold array 完成但依赖聚合认证失败，终态见第 8 节。尚无本版本性能结论；预注册配置中的历史状态字段不得为反映实现进度而改写。

## 8. Ibex 首次运行终态（不含 outer metrics）

Slurm array `51063738_[0-4]` 于 2026-08-31 00:46:13 +03:00 同时开始，五个 task 全部 `COMPLETED`、exit `0:0`。按 task 0–4：

| task / held-out family | node | elapsed | end | batch MaxRSS |
|---|---|---:|---|---:|
| `0 / half_cylinder` | `cn509-07-r` | `00:10:44` | `00:56:57` | `13478416K` |
| `1 / delta_wing` | `cn509-05-l` | `00:12:07` | `00:58:20` | `19409736K` |
| `2 / f22_raptor` | `cn509-04-l` | `00:12:27` | `00:58:40` | `20518168K` |
| `3 / channel` | `cn509-03-r` | `00:13:31` | `00:59:44` | `21470428K` |
| `4 / boeing_747` | `cn514-06-r` | `00:10:35` | `00:56:48` | `21643012K` |

依赖聚合 `51063753` 于 00:59:45 在 `cn504-17` 开始，01:04:54 以 `FAILED`、exit `1:0` 结束，elapsed `00:05:09`，batch MaxRSS `12250816K`。失败发生在 fresh replay 的认证阶段，精确错误为：

```text
authenticated spatial score does not match tail transform
```

这说明当前 fold artifact 尚未通过冻结的 fresh-replay 合同；它不说明逐尺度度量的性能好坏。五个 fold 目录及失败聚合目录必须原样保留，不覆盖、不删除，也不得把其中任何 outer metric 读出或接受为证据。后续工作限定为 label-free 诊断：只定位已认证 spatial score 与由 tail transform 重放所得 score 不一致的首个 identity、字段与数值来源；在诊断完成、修复另行版本化并重新认证以前，本版本保持“无 outer 性能结论”。
