# mainExp_TemplateMatching_1.2：空类 library stratum 的可审计处理

状态：**cache-backed development 协议已冻结、尚未运行；formal confirmation 禁止**。完整运行配置为 `config/mainExp_TemplateMatching_1.2_development.yaml`；未在该文件重定义的 descriptor、scale、split、metric、bootstrap 和三联图规则全部继承 1.1。

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
