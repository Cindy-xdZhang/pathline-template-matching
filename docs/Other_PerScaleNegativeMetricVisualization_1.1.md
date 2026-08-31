# `Other_PerScaleNegativeMetricVisualization_1.1`

状态：**`COMPLETED_LOCAL_REPORTING_QA_PASS`**。本版本为已认证
`Verify_PerScaleNegativeMetric_1.1` 的下游报告，不重新训练、拟合、选择候选、
调阈值或按显示效果挑图。预测来自 Ibex clean deployment commit
`e919c2e27b8c8157435d40da350866864721ac51`，冻结 config SHA-256 为
`b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79`。

报告目录：

```text
outputs/Other_PerScaleNegativeMetricVisualization_1.1_local_reporting_20260831
```

报告脚本为 `scripts/render_per_scale_negative_metric_visualizations.py`，源码 SHA-256
为 `e41d478cff25200221569642222d62d447c00baef91534ef3d3a35640c6c683c`；该脚本最早
由 commit `c678baa68d686982f968c9030e84277168da0e06` 锁定。主实验预测在 Ibex jobs
`51064965_[0-4]` 运行，并由 `51064966` 完成五折认证；父空间 scenes 来自
`Other_MainExp31FamilyHeldOutVisualization_1.1` job `51029080`。本次只在本地对这些
已认证产物做确定性渲染和交付质量检查，因此没有新增 Slurm job。

## 图的固定定义

四个 query flow 固定为 `cylinder3d`（Re160）、`halfcylinderRe640`（Re640）、
`halfcylinderRe6400`（Re6400）和 `boeing747`，source ordinal 固定为2。每个 flow
必须同时报告两个互斥尺度块：

- `legacy_2_1`：原1000个尺度，scale ID 0–999；
- `expanded_3_1`：新增1000个尺度，scale ID 1000–1999。

禁止在看到结果后只挑一个 block，也禁止跨 block 投票或聚合，所以交付物固定为
4 flow × 2 block = 8张三联图。每张图三栏为：

1. IVD-p95 等值面与固定240条中心 pathlines；
2. 所有 valid query rows 的 FMT PerScale 模板二分类；
3. 同一 rows 相对 IVD-p95 标签的 TP/FP/FN/TN 空间分解。

第二栏不是聚类。第三栏固定编码为：TP 红圆、FP 紫色三角、FN 橙色 `x`、TN 淡蓝点。
IVD 是 Instantaneous Vorticity Deviation（瞬时涡量偏差），标签为 whole-volume
`||curl(v)-spatial_mean(curl(v))||` 的第95百分位阈值。

三个 cylinder flow 的完整 `half_cylinder` family 从 library、normalization 和
calibration 中排除；Boeing 查询时完整排除 `boeing_747` family。half-cylinder fold
固定选择 `chirality_all35, k=5, sigma=1.5, top-5%`；Boeing fold 固定选择
`real_neighbor36, k=5, sigma=1.0, top-5%`。这是 nested family selection 的真实结果，
不得为了图面统一而改写。

## 分类结果

Average Precision（AP）是按分类分数计算的精确率–召回率面积汇总；balanced
accuracy（BA）是正负类召回率的平均；F1 是 precision 与 recall 的调和平均。

| Flow | Block | Valid rows | AP | F1 | BA | Precision | Recall | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cylinder3D Re160 | legacy | 60,560 | 0.7586 | 0.6797 | 0.8059 | 0.7444 | 0.6254 | 2,254 | 774 | 56,182 | 1,350 |
| Cylinder3D Re160 | expanded | 43,347 | 0.7100 | 0.6184 | 0.7522 | 0.7694 | 0.5169 | 1,668 | 500 | 39,620 | 1,559 |
| Cylinder3D Re640 | legacy | 60,555 | 0.5480 | 0.5229 | 0.7361 | 0.5532 | 0.4959 | 1,675 | 1,353 | 55,824 | 1,703 |
| Cylinder3D Re640 | expanded | 42,463 | 0.4341 | 0.4597 | 0.6962 | 0.5099 | 0.4185 | 1,083 | 1,041 | 38,834 | 1,505 |
| Cylinder3D Re6400 | legacy | 62,313 | 0.5611 | 0.5121 | 0.7398 | 0.5196 | 0.5048 | 1,619 | 1,497 | 57,609 | 1,588 |
| Cylinder3D Re6400 | expanded | 57,906 | 0.5545 | 0.4878 | 0.7229 | 0.5048 | 0.4719 | 1,462 | 1,434 | 53,374 | 1,636 |
| Boeing 747 | legacy | 61,432 | 0.8030 | 0.7206 | 0.8576 | 0.7113 | 0.7303 | 2,185 | 887 | 57,553 | 807 |
| Boeing 747 | expanded | 17,601 | 0.6549 | 0.5664 | 0.8578 | 0.4574 | 0.7435 | 403 | 478 | 16,581 | 139 |

不能从表中断言 legacy 尺度普遍优于 expanded 尺度，因为两个 block 的尺度集合、
valid population 和 pathline 几何不同；图只显示当前冻结方法在各自 block 上的实际结果。
Boeing legacy 是唯一 F1 超过0.70的固定图；Re640 expanded 最低，为0.4597。

## 认证与交付质量检查

报告保留全部 `406,177` 个 valid query rows，没有为了显示丢弃样本。八个 scene 与
prediction 的 dataset、source ordinal/index、block、center seed、assigned row、scale ID
和 block index 均完成 exact ordered identity join；8/8 指标在 `1e-12` 容差内从图中
reference/prediction 重算通过。source、240条 pathlines、IVD mesh、camera 和 bounds 均未改变。

交付 QA 结果：

- 8/8 panel-alignment PASS，物理容差1.5 pt；
- 8/8 PDF 可审计，最小文字7 pt，无低于5 pt字形；
- PyMuPDF rendered-collision audit 为0 FAIL；89个 warning 均为3D raster 边界与数值刻度的
  接触，逐图复核后接受；
- 8/8 PNG 成功解码，均为7560×1800、RGBA、360 dpi；
- 8/8 原始 PNG 逐图目视 PASS；SVG/PDF 文字可编辑。

关键文件 SHA-256：

- `input_manifest.json`：`6f41b997fb7d76bf8bf5d4e82e5e8650dc78956ae5e2e09736adcce88195f842`；
- `per_figure_metrics.csv`：`e60ba2c50d2f6d3e46ed71354434330c10d9a5cb8f22d3f7b3c17728c32a4229`；
- `result_manifest.json`：`0d2a1e0c0f930c2eac3dc2d54998964dd439f5013c581c290cb30f30fd28a0aa`；
- `delivery_qa_summary.json`：`3aa37062f7e00e2bd5c9a2032222fc4ec77822e61c83e0144b901cfe0db38987`。

原始 `RUN_COMPLETE.json` 的状态是 `complete_pending_local_pdf_collision_qa`；随后生成的
`delivery_qa_summary.json` 已完成该后置门禁并记录 `delivery_status=PASS`。两者描述的是
先后两个阶段，不构成结论矛盾。

## 结论边界

这些结果是 `family-held-out exposed-development fixed-source reporting`。完整 physical
family 已从对应 library 中排除，但四个 flow 的数据和 fitted outer-fold classifier 已在
开发阶段暴露；固定 top-5% 判决还依赖完整 query group。因此它们不能称 sealed
confirmation、任意单 primitive 独立分类器或2000尺度统一投票结果。

结构化摘要见
`docs/evidence/Other_PerScaleNegativeMetricVisualization_1.1_local_summary.json`。
