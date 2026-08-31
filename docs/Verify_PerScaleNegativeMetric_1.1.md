# `Verify_PerScaleNegativeMetric_1.1`

执行状态：`completed_stopped_after_authenticated_five_fold_failure`。配置中的预注册字段仍保持 `status: frozen_pre_run_not_implemented`，以保留冻结 SHA-256；方法、候选集与输出认证合同是在读取任何 `Verify_NegativeTailCalibration_1.1` outer 指标前冻结的。五折 array `51063738` 的五个 task 均已完成，但依赖聚合 `51063753` 在 fresh replay 中因跨CPU Gaussian浮点末位差失败。两项label-free诊断与validation-only hardening随后完成；clean rerun array/aggregate `51064965/51064966`通过完整认证后才首次读取本版本 outer metrics。权威结果见第9节。

实现路径：`src/pathline_template_matching/per_scale_negative_metric.py`、`scripts/run_verify_per_scale_negative_metric_1_1.py`、`scripts/aggregate_verify_per_scale_negative_metric_1_1.py`、`ibex/verify_per_scale_negative_metric_1.1_all_folds.sh` 与 `ibex/verify_per_scale_negative_metric_1.1_aggregate_five.sh`。原数值方法commit为 `809ffa3b9490ca4f5b0817d77759b5d88cce628c`；不改变数值方法的validation hardening commit为 `eba96eb8bb2a20e0e41318cee0a6406e70605b66`，clean deployment commit为 `e919c2e27b8c8157435d40da350866864721ac51`。实际设备和终态见 `docs/ibex_run_registry.md`。

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
- 本版本在任何 `Verify_NegativeTailCalibration_1.1` outer 指标可见前冻结。首次 Ibex fold array 完成但依赖聚合认证失败，历史终态见第8节；随后 validation-only hardening 和 clean rerun 的权威性能终态见第9节。预注册配置中的历史状态字段不得为反映实现或结果进度而改写。

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

### 8.1 Label-free 浮点可移植性诊断

只读诊断 job `51064502` 在与失败聚合相同的 `cn504-17`（AMD EPYC 9655）上，只打开五折的 `outer_prediction_manifest.json` 与 `outer_predictions.npz`，未打开 outer labels、metrics、result 或 aggregate summary。它以1线程重放由已保存 tail anomaly 得到的 Gaussian spatial score：前四折逐位一致；原来在 AMD EPYC 7702 生成的 Boeing 折有78,874个 score 发生末位差，最大绝对差 `5.551115123125783e-16`、最大6 ULP，denominator最大5 ULP，但五折最终布尔 prediction 均为0个变化。

随后 job `51064646` 在同一 `cn504-17`、32线程环境中完成全部五折的完整 label-free query replay。Folds 0–3 的全部字段逐位一致；Boeing fold仍只有 `spatial_score` 和 `spatial_denominator` 不同，最大分别为6和5 ULP。`raw_negative_distance`、tail probability/anomaly、全部 identity/support/mode/imputation、group audit和最终 prediction逐位一致。结合 Intel Xeon Gold 6248 上 alternate-SIMD 的同类无标签复现，根因是 `np.exp` 生成 Gaussian kernel及后续 NumPy 运算在不同CPU/SIMD路径上的少量 ULP 差异，而当前认证错误地对连续 `float64` 数组要求 `np.array_equal`。

因此认证修订只允许 `spatial_score` 和 `spatial_denominator` 各最多8 ULP，并继续要求它们的dtype、shape、finite、nonnegative和zero mask完全一致；artifact/manifest SHA-256以及 raw distance、tail values、全部离散状态、group audit和最终 prediction仍为exact。这是validation-only hardening，不改变Gaussian scorer、分数、候选、split或指标。边界在读取任何 outer metrics 前由jobs `51064502/51064646`冻结。旧fold与失败聚合保持不变；必须从新的clean commit重跑五折及聚合后才能读取指标，目前仍无性能结论。

## 9. Clean rerun、认证结果与停止结论

Validation-hardened clean rerun 使用 exact commit
`e919c2e27b8c8157435d40da350866864721ac51`。Array
`51064965_[0-4]` 的五项均 `COMPLETED 0:0` 且
`label_free_postvalidation=passed`。聚合 `51064966` 于
2026-08-31 02:06:25--02:09:33 +03:00 在 `cn604-11` 完成，exit
`0:0`，32 CPU、32 GB，batch MaxRSS `12023788K`。聚合重新执行 scaler、
calibrator、outer query、prediction authentication、label gate、逐组指标和层级宏平均，
而不是信任 fold 内嵌 summary。

| Outer physical family | Selected candidate | AP | F1 | BA | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `half_cylinder` | `chirality_all35`, `k=5`, `sigma=1.5`, top 5% | 0.568894 | 0.542668 | 0.744518 | 0.582349 | 0.511196 |
| `delta_wing` | `real_neighbor36`, `k=15`, `sigma=1.0`, top 5% | 0.901151 | 0.781242 | 0.936346 | 0.703648 | 0.888110 |
| `f22_raptor` | `real_neighbor36`, `k=31`, `sigma=0.5`, top 5% | 0.495125 | 0.503504 | 0.753988 | 0.477131 | 0.535359 |
| `channel` | `real_neighbor36`, `k=5`, `sigma=1.0`, top 5% | 0.179057 | 0.244366 | 0.633502 | 0.202731 | 0.308229 |
| `boeing_747` | `real_neighbor36`, `k=5`, `sigma=1.0`, top 5% | 0.703612 | 0.618757 | 0.857668 | 0.546225 | 0.738860 |
| equal-family macro | — | 0.569568 | 0.538108 | 0.785204 | 0.502417 | 0.596351 |

停止规则只有 balanced accuracy 通过；macro F1、Average Precision、precision、
recall、至少4/5 family达到0.65及任一family不低于0.50均失败。相对直接父版本
`Verify_NegativeTailCalibration_1.1` 的 macro F1 `0.540472`，本版本为
`0.538108`，变化 `−0.002364`。因此逐尺度 diagonal variance 没有解决主要问题，
不得继续在这个版本上选择 lambda、方差公式或阈值。

认证证据：

- config/input/input-rows SHA-256：`b469b909…` / `e57d6b52…` /
  `ceb6d0e3…`；
- aggregate completion/manifest/summary/table file SHA-256：
  `b6621789…` / `1bbd13bf…` / `60a8dcbd…` / `c5262c87…`；
- remote复核五折精确75个文件的size/SHA、所有JSON self-hash、commit/config/family/
  input引用链均PASS；
- 本地下载后重新认证4个aggregate文件、五份outer-group CSV artifact hash，按每个
  `dataset×source×block` 重新计算family F1与五family等权宏平均，误差小于
  `1e-12`；本地路径为
  `outputs/Verify_PerScaleNegativeMetric_1.1_job51064966_download` 与
  `outputs/Verify_PerScaleNegativeMetric_1.1_job51064966_fold_metrics_download`。

当前结论是负结论：瓶颈不在全局方差与逐尺度方差之间的选择。下一步仅进入在本
版本 outer 指标可见前已冻结的两个表示级验证：
`Verify_EarlyOppositePairKinematics_1.1` 与
`Verify_RawPCANegativeMetric_1.1`。Tangaroa 与 Smoke 继续禁止访问。
