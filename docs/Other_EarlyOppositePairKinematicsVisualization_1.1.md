# `Other_EarlyOppositePairKinematicsVisualization_1.1`

状态：**`COMPLETED_LOCAL_REPORTING_QA_PASS`**。本版本是已认证
`Verify_EarlyOppositePairKinematics_1.1` 的固定下游可视化，不重新训练、拟合、
选择候选、调阈值或按图面挑结果。EarlyOppositePair 在原 FMT 表示后追加固定 4D
seed-time opposite-pair kinematics：用中心点六个轴向邻点的速度中央差分计算涡量范数、
应变范数、带符号散度与带符号 Q criterion。

预测来自 numerical commit
`2c3774dca0d81db8edd5645e63576526b9e276f7`，预测 config SHA-256 为
`e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767`。报告配置
SHA-256 为 `0b5053cdd2342fcd65950b82f08b520de4c8a2717c44ad15a5d13babd0caf1c8`。

报告目录：

```text
outputs/Other_EarlyOppositePairKinematicsVisualization_1.1_local_reporting_20260831
```

报告器由 Git commit `7e2d7c7e44385f91414c6e4ec347f88e16da7466` 固定，脚本
`scripts/render_early_opposite_pair_kinematics_visualizations.py` 的 SHA-256 为
`fe0d34ba2859f33f06fbfdec1e2b4756c4d9abd3f5f0d9f44ee6c295bf3f2226`。父空间
scenes 来自 Ibex job `51029080`；half-cylinder 与 Boeing 预测分别来自
`51070299_0` 和 `51070386_4`，完整五折认证来自 `51070392`。本次只在本地对已认证
产物做确定性连接、渲染和交付检查，因此没有新增 Slurm job。

## 图的固定定义

固定 query 为 `cylinder3d`（Re160）、`halfcylinderRe640`（Re640）、
`halfcylinderRe6400`（Re6400）与 `boeing747`，source ordinal 固定为2。每个 flow
同时保留两个互斥尺度块：

- `legacy_2_1`：原1000个尺度，scale ID 0–999；
- `expanded_3_1`：新增1000个尺度，scale ID 1000–1999。

两个 block 的有效 primitive population 不同，禁止看到结果后只选一个 block，也禁止
跨 block 投票或合并，所以交付物固定为4个 flow × 2个 block = 8张三联图。每张图为：

1. whole-volume IVD-p95 等值面与固定240条中心 pathlines；
2. 全部 valid query rows 的 FMT EarlyOppositePair 模板二分类；
3. 同一 rows、顺序、相机与边界下的 true positive（TP）、false positive（FP）、
   false negative（FN）和 true negative（TN）空间分解。

IVD 是 Instantaneous Vorticity Deviation（瞬时涡量偏差），定义为
`||curl(v)-spatial_mean(curl(v))||`；p95 标签使用 whole-volume 第95百分位阈值。
第二栏是分类，不是聚类。第二栏红色表示预测涡区、蓝色表示预测非涡区；第三栏为
TP 红圆、FP 紫色三角、FN 橙色叉号、TN 淡蓝点。图内没有独立图例，因此独立引用图片时
必须连同本段或等价 caption 一起使用。

三个 cylinder flow 固定使用
`chirality_all35_plus_seed4, k=31, sigma=0.5, top-5%`；Boeing 固定使用
`real_neighbor36_plus_seed4, k=31, sigma=0.5, top-5%`。这些是已完成 nested
physical-family selection 的真实候选，不为可视化改变。

## 分类结果

Average Precision（AP）汇总 precision–recall 曲线；balanced accuracy（BA）是正类与
负类召回率的平均；F1 是 precision 与 recall 的调和平均。Coverage 以每个
`dataset×block` 的64,000个 assigned rows 为分母，invalid rows 不进入分类指标。

| Flow | Block | Valid / 64,000 | Coverage | Accuracy | AP | F1 | BA | Precision | Recall | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cylinder3D Re160 | legacy | 60,560 | 94.63% | 0.9712 | 0.7815 | 0.7367 | 0.8338 | 0.8068 | 0.6779 | 2,443 | 585 | 56,371 | 1,161 |
| Cylinder3D Re160 | expanded | 43,347 | 67.73% | 0.9553 | 0.7674 | 0.6406 | 0.7623 | 0.7970 | 0.5355 | 1,728 | 440 | 39,680 | 1,499 |
| Cylinder3D Re640 | legacy | 60,555 | 94.62% | 0.9543 | 0.6285 | 0.5682 | 0.7588 | 0.6011 | 0.5388 | 1,820 | 1,208 | 55,969 | 1,558 |
| Cylinder3D Re640 | expanded | 42,463 | 66.35% | 0.9437 | 0.5197 | 0.4928 | 0.7122 | 0.5466 | 0.4486 | 1,161 | 963 | 38,912 | 1,427 |
| Cylinder3D Re6400 | legacy | 62,313 | 97.36% | 0.9694 | 0.7782 | 0.6981 | 0.8364 | 0.7083 | 0.6882 | 2,207 | 909 | 58,197 | 1,000 |
| Cylinder3D Re6400 | expanded | 57,906 | 90.48% | 0.9649 | 0.7432 | 0.6610 | 0.8114 | 0.6840 | 0.6394 | 1,981 | 915 | 53,893 | 1,117 |
| Boeing 747 | legacy | 61,432 | 95.99% | 0.9836 | 0.9093 | 0.8338 | 0.9178 | 0.8229 | 0.8449 | 2,528 | 544 | 57,896 | 464 |
| Boeing 747 | expanded | 17,601 | 27.50% | 0.9747 | 0.7884 | 0.6873 | 0.9396 | 0.5551 | 0.9022 | 489 | 392 | 16,667 | 53 |

Boeing legacy 的 F1 最高，为0.8338；Re640 expanded 最低，为0.4928，图中对应较多
FN。Boeing expanded 的 recall 很高，但 precision 只有0.5551，而且 coverage 仅27.50%；
它的高 accuracy 和 BA 只适用于17,601个 valid rows，不能外推到未积分成功的46,399行。

在同一固定 source 2 上，Early 的八行 F1 均高于旧 PerScale 图：Re160
legacy/expanded 分别增加 `+0.0570/+0.0222`，Re640 为 `+0.0453/+0.0331`，Re6400 为
`+0.1860/+0.1732`，Boeing 为 `+0.1131/+0.1209`。八行等权平均从0.5710升至
0.6648，描述性差值为+0.0938。该比较使用已暴露的固定 source 与已选择分类器，并且
representation、`k` 和 `sigma` 同时变化，只能说明完整 Early 方法在这些图上的结果更好，
不能作为新的无偏确认，也不能把提升归因于单个组件。

## 认证与交付质量检查

报告连接并保留全部 `406,177` 个 valid query rows，没有为显示丢弃样本。八个 scene 与
prediction 的 dataset、source ordinal/index、block、center seed、assigned row、scale ID
和 block index 均完成 exact ordered identity join；8/8 指标在 `1e-12` 绝对容差内从图中
reference/prediction 重算通过。240条 pathlines、IVD mesh、camera 和 bounds 均未改变。

交付检查结果：

- 权威绘图源码预检18 PASS、3个已复核 warning、0 FAIL；
- 8/8 panel alignment 严格检查通过，16次比较均无 warning/fail；
- 8/8 PDF 可审计，最小文字7 pt，无低于5 pt字形；
- PDF rendered-collision audit 为0 FAIL；89个 warning 均为3D数值刻度或轴字符接触
  raster/fill 边缘，结合八张原图复核后接受；
- 8/8 PNG 完整解码，均为7560×1800、RGBA、约360 dpi；
- 8/8 原图逐图检查没有裁切、空面板、相机错位、边界不一致或异常 pathline；图内缺少
  独立图例的 warning 通过强制相邻 caption 解释处理。

关键文件 SHA-256：

- `input_manifest.json`：`40335898add75ed1e69caeeeff0fb9a374507bd0e49b38271f9e4b8f2f5ade63`；
- `per_figure_metrics.csv`：`d6c88d93c16d7ad33c25466c6f2e56516dd52eb7a417f2a81480434be67517e0`；
- `visualization_manifest.json`：`8d261f6dc2eca555c7a54e9237e852814c5ad73636bc7b503af25bc05ef4c2ee`；
- `result_manifest.json`：`75cf4bb0b37294735eb3800f5157fd1ecd4ca37c2d96c6cd89cd5bba7aabdb7d`；
- `RUN_COMPLETE.json`：`bb94c75f5cca3a18456df7680655dd6f1d09f4ad21bd3488eb429d8b5ec27dfd`；
- `delivery_qa_summary.json`：`18b821975a1e292ac5c4513a2298830314cff8dbe53562c8b80882cd6c1dcb2c`。

原始 `RUN_COMPLETE.json` 状态是 `complete_pending_local_pdf_collision_qa`；随后生成的
`delivery_qa_summary.json` 已完成该后置门禁并记录 `delivery_status=PASS`。两者是先后阶段，
不构成状态矛盾。

## 结论边界

这些图是 `family-held-out exposed-development fixed-source reporting`。对应完整 physical
family 已从 library、normalization 和 calibration 中排除，但四个 query flow、fitted
outer-fold classifier 与 source 2 都已暴露；固定 top-5% 判决还依赖完整 query group。
因此结果不能称 sealed confirmation、任意单 primitive 独立分类器、聚类或2000尺度统一投票。

完整五-family `Verify_EarlyOppositePairKinematics_1.1` 的 macro F1 为0.639163，仍低于
预注册成功阈值0.70。这些固定 source 图解释了方法在哪里正确或错误，不改变父验证版本
“完整五折失败并停止”的结论。

结构化摘要见
`docs/evidence/Other_EarlyOppositePairKinematicsVisualization_1.1_local_summary.json`。
