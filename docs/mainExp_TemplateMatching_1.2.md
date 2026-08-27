# mainExp_TemplateMatching_1.2：空类 library stratum 的可审计处理

状态：**cache-backed development 已完成；formal confirmation 未运行且仍禁止访问**。完整运行配置为 `config/mainExp_TemplateMatching_1.2_development.yaml`；未在该文件重定义的 descriptor、scale、split、metric、bootstrap 和三联图规则全部继承 1.1。

## 修订原因

`mainExp_TemplateMatching_1.1` 的 Ibex job `50930751` 在任何性能指标产生前失败：`channel/ordinal0/lib_o025_d0125_n48` 的 library stratum 有 228 个 negative、0 个 positive。1.1 规定任一空类 stratum 立即失败，因此 1.1 保持失败状态，不原地修改。

1.2 只改变一个建库规则：

```text
m = min(512, available_negative, available_positive)
if m > 0: 两类各确定性选择 m 个模板
if m = 0: 该 flow×source-time×scale stratum 两类都选择 0 个模板，并写入 audit
```

不得只保留非空类；这样会引入类别偏置。被跳过 stratum 的 query 不删除、不降采样，仍进入全部主指标。该 stratum 的 library-side rows 仍可进入不使用 query 的无监督 Raw-PCA 拟合以及 constant prior；exact 1NN 的 normalization 仍只从实际选中的平衡模板拟合。

每折 manifest 必须报告 `skipped_library_stratum_count` 和 `skipped_library_candidate_count`；`audit_counts.csv` 必须保留 aggregate、negative、positive 三行，并标记 `stratum_status=skipped_empty_class`、两类 `selected_count=0`。如果全部 stratum 都被跳过，或最终 library 不再同时包含两类，仍然 fail closed。

## 未改变的协议

- 7 个 physical family leave-one-out；seen-scale 与 exposed-development unseen-scale 分开；
- 旧 development ordinals 4–5 禁止进入主指标；sealed confirmation access 禁止；
- 四臂为 constant prior、Raw672 exhaustive 1NN、library-only Raw-PCA161 exhaustive 1NN、FMT161 exhaustive 1NN；
- query 使用所有 valid primitive 和自然类别比例；
- physical-family macro、5000 次 paired source-timeslice bootstrap、percentile 95%/NumPy `linear` 不变；
- 每个 dataset×regime 固定 ordinal 2 的三联图，不按性能选图。

本 development 版本仍只能产生描述性证据。它不能替代新的 flow family sealed confirmation，也不能回答连续任意尺度、不同 descriptor 或重新积分 raw primitive 的问题。

## Ibex 运行证据

`mainExp_TemplateMatching_1.2` 于 2026-08-27 在 Ibex job `50932239` 完成：

| 项目 | 证据 |
|---|---|
| 状态 | `COMPLETED`，exit `0:0`；最终研究状态 `development_completed_confirmation_not_run` |
| 时间 | 18:04:42–18:17:29 +03:00；elapsed `00:12:47` |
| 设备 | `gpu510-32`；1×Tesla V100-PCIE-32GB；16 CPU；64 GB；batch MaxRSS `19,743,764K`（约 18.83 GiB） |
| 数值代码 | Git commit `700d392b590f46a68f8ef6e973524ee0a7886c62` |
| config | SHA-256 `1af4bd91bcc9621570a91748c8c4bbc9493a76d17feed55ea03188742607f72f` |
| 输入 | 100 个 cache files、390,140 个 primitives；input manifest content SHA-256 `3a05640211341eb4b8ac4c4dedbe5851bb3aa1303bf84c6b48561f45deb9cb1f` |
| 运行前门禁 | 44/44 tests 通过；CUDA matcher gate 通过，`TF32=False`、float32 matmul precision `highest`、deterministic algorithms enabled |
| 结果 | result manifest file SHA-256 `217cdaf7e1551bab588788fba6e1f2e24f41c5bba0e20f52e3f3dcbcb5bce3d0`；canonical content SHA-256 `9ae7de1790682aacb25d46284d76d3bbadfccbfe6040f57dcdf4c2f7364df906` |

冻结的 pre-run config 把 `evidence_scope.performance_results_exist=false` 原样复制进 immutable result manifest；该字段只记录运行前状态，完成后已陈旧，不能用于判断结果是否存在。权威完成字段是 result manifest 顶层 `status=development_completed_confirmation_not_run`、`run_state.json` 的同名状态及其 required-output audit。已哈希的 config/result 不原地改写；这一差异在结构化 summary 中显式登记。

每折平衡模板数为 11,906–17,530。已知空类 stratum 在 channel 被留作 query 时不进入 library，因此 held-out `channel` 折的 skip 为 0；其余六折都审计到同一个 228-candidate stratum，并按 1.2 规则将两类模板选择数都设为 0。该数据仍进入 Raw-PCA 拟合候选、constant prior 候选和相应 query，未被从评测总体删除。

## 主结果

主推断表采用 `physical_family_macro`：先在每个 physical family 内对 source-timeslice 指标做宏平均，再对 7 个 family 宏平均。Average Precision（AP，按连续分数衡量正类排序）与 F1 是主指标；Area Under the Receiver Operating Characteristic Curve（AUROC，受试者工作特征曲线下面积）等为辅助指标。

| 尺度 regime | 方法 | AP | F1 | AUROC | Precision | Recall | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| seen | Library prior | 0.0579 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| seen | Raw 672D + exact 1NN | 0.2045 | 0.1791 | 0.6270 | 0.1080 | 0.8644 | 0.5969 |
| seen | Raw-PCA 161D + exact 1NN | 0.3037 | 0.2233 | 0.7074 | 0.2316 | 0.7969 | 0.6342 |
| seen | **FMT 161D + exact 1NN** | **0.3563** | **0.3039** | **0.7619** | 0.2297 | 0.7340 | **0.6980** |
| unseen | Library prior | 0.0581 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| unseen | Raw 672D + exact 1NN | 0.1897 | 0.1723 | 0.5928 | 0.1104 | 0.8413 | 0.5680 |
| unseen | **Raw-PCA 161D + exact 1NN** | **0.3649** | **0.3059** | 0.7257 | **0.2934** | 0.7555 | **0.6797** |
| unseen | FMT 161D + exact 1NN | 0.2923 | 0.2349 | **0.7534** | 0.1554 | **0.7905** | 0.6650 |

### 配对 bootstrap 差值

下表是 `FMT − comparator`，使用 5000 次 physical-family-stratified、paired source-timeslice bootstrap 和 percentile 95% interval；用户尚未冻结把区间符号当作正式 pass/fail gate 的规则，因此只作描述。

| Regime | 指标 | Comparator | 点估计 | 95% interval |
|---|---|---|---:|---:|
| seen | AP | Raw 672D | +0.1518 | [+0.1363, +0.1666] |
| seen | AP | Raw-PCA 161D | +0.0527 | [+0.0359, +0.0691] |
| seen | F1 | Raw 672D | +0.1248 | [+0.1135, +0.1359] |
| seen | F1 | Raw-PCA 161D | +0.0806 | [+0.0700, +0.0911] |
| unseen | AP | Raw 672D | +0.1026 | [+0.0940, +0.1111] |
| unseen | AP | Raw-PCA 161D | **−0.0726** | **[−0.0833, −0.0622]** |
| unseen | F1 | Raw 672D | +0.0626 | [+0.0591, +0.0659] |
| unseen | F1 | Raw-PCA 161D | **−0.0711** | **[−0.0760, −0.0660]** |

## 结果解释与反例

当前 development run 有三个可直接复核的观察结果；由于通过规则未冻结，它们不构成主命题的 pass/fail 判定：

1. FMT 161D 在 seen-scale 的 AP/F1 点估计都高于 Raw 672D 和同维 Raw-PCA 161D，对应差值区间全为正。
2. FMT 在 unseen-scale 的 AP/F1 点估计仍高于 Raw 672D，但低于 Raw-PCA 161D，对应 FMT−Raw-PCA 区间全为负；“FMT 的 unseen-scale AP/F1 高于 Raw-PCA”没有在本次 development run 中被观察到。
3. FMT unseen-scale 的 AUROC 为 0.7534，高于 Raw-PCA 的 0.7257，而 AP 为 0.2923、低于 0.3649。AUROC 和 AP 都来自连续分数排序；二者方向不同，反映 ROC 与 precision-recall 在类别不平衡下强调的排序位置不同。固定的“一最近模板类别直接作为预测”另产生较低 precision/F1 和部分 flow 的大量 false positive；调整该二值决策不能被假定会修复 AP。

`counterexamples.csv` 完整保留 480 个 `FMT <= comparator` 的 flow/scale/metric 条目。最严重的 unseen-scale AP 反例是 `cylinder3d/eval_o125_d015_n40`：FMT 0.0552、Raw-PCA 0.7070，差值 −0.6518。Smoke 的 unseen-scale 固定三联图显示 2,586 个 false positive，只得到 168 TP、13 FN；这直接说明当前固定二值决策在该 scene 的 precision/F1 行为较差，不用于解释阈值无关的 AP。

## 三联图与布局修订

每个 dataset×regime 固定 source ordinal 2，共 10×2=20 张图，不按性能选样本。三栏共享同一 evaluated seed 坐标、正交相机和物理 bounds：

1. 背景流场：whole-loaded-volume IVD p95 等值面、240 条 center pathlines，以及全部 evaluated seeds；5 个 raw-backed dataset 的两个 regime 共 10 张图使用真实重建等值面，另外 10 张明确标记 positive-seed fallback。
2. 模板类别分配：蓝色 non-vortex、红色 vortex。这里不是重新运行 K-Means 聚类，而是显示预计算的 FMT exact one-nearest-neighbor binary template assignment；这样才与本项目方法一致。
3. 误差分析：TN 淡蓝、TP 红、FP 紫色三角、FN 橙色 `x`；严格使用同一 seed 顺序与 IVD p95 reference。

原始 Ibex 数值产物未覆盖。首次图的标题接触画布上边界，因此另建只改布局、不重算指标的 `Other_MainExp12FigureLayout_1.1`：renderer commit `cfe2afcf01133a3a7034db05175710e8f9dd70fe`；20 张图仍为 7560×1800，所有图第一像素行均为空白，counts、camera、bounds 和 scene hashes 与原图一致。重绘 manifest content SHA-256 为 `63961fd290b05ed714b8761a8b22c98741a2450e4459969bbda44a0beeeed93b`，file SHA-256 为 `8fd85b9612d025bcfe51362853748878cdedaab21507534b31adb0161b93b1c5`。

机器可读汇总见 `docs/evidence/mainExp_TemplateMatching_1.2_ibex_summary.json`。完整结果在 Ibex：`/home/zhanx0o/pathline-template-matching/outputs/mainExp_TemplateMatching_1.2_development/runs/slurm_50932239_700d392b590f`。本地下载保留表格、报告、20 张图和 scene；完整 per-query/fold artifacts 仍以 Ibex result manifest 为准。
