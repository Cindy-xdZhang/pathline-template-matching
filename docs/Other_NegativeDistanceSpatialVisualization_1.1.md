# Other_NegativeDistanceSpatialVisualization_1.1：四个指定流场的当前候选分类三联图

状态：**`completed_family_held_out_exposed_development_visualization`**。唯一配置为
`config/Other_NegativeDistanceSpatialVisualization_1.1.yaml`，冻结 SHA-256
为 `82b92a52690eab3883287dc71a8ac2c57a691062188b0629ae83e331c6252c5c`。

## 研究问题

本版本只回答一个空间解释问题：当完整 query physical family 不进入模板库时，
当前固定候选
`nearest non-vortex FMT-template distance + within-group rank + mask-normalized Gaussian sigma=1 + fixed top-5%`
在 `cylinder3d`（Re160）、`halfcylinderRe640`、
`halfcylinderRe6400` 和 `boeing747` 的哪些位置与
whole-loaded-volume IVD-p95 一致或不一致？

它不是新的分类方法版本，不重新积分 pathline、不重建 FMT feature、不重新选
sigma 或 threshold。候选来自已经完成的
`Other_NegativeDistanceSpatial_1.1`，因此图只能称为
**family-held-out exposed-development visualization**，不能称为 formal
confirmation、无偏模型选择或跨物理族泛化证明。

## 冻结输入

1. 八个父 scene 来自
   `Other_MainExp31FamilyHeldOutVisualization_1.1` Ibex job `51029080`。
   查询 Cylinder/Re640/Re6400 时完整排除 `half_cylinder` family；查询 Boeing
   时完整排除 `boeing_747` family。父 scene 的 bounds、全部 valid seeds、
   IVD-p95 mesh、reference、240 条中心 pathlines、相机和 block 身份必须逐数组
   保持不变。
2. 当前候选 prediction 来自
   `Other_NegativeDistanceSpatial_1.1` Ibex job `51039505` 的不可变
   `predictions.csv`，文件 SHA-256 为
   `cc4651baefeabda0c4570ad4a3f8a6b855e2616e835af02994294a2095f177f2`。
3. score 固定为 `masked_gaussian_rank_sigma_1`；prediction 固定为
   `masked_gaussian_rank_sigma_1__fixed_top_fraction_0.05`。禁止读取 oracle
   threshold，禁止按图、dataset 或 block 改 threshold。

Runner 必须按
`dataset × source ordinal × block × center index` 精确连接 prediction 与父
scene；duplicate、missing、extra 或顺序身份不一致均 fail closed。每图 panel b/c
使用该 block 的全部 valid rows，不进行可视化下采样。

## Figure contract

- 核心结论：当前固定候选能产生可定位的 family-held-out 分类和错误分布，但
  不达到项目的 F1 目标，而且错误随 flow 与 scale block 改变。
- Results-level 问题：完整目标 family 不进入库时，分类在何处与 IVD-p95
  一致或不一致？
- 图件类型：`image plate + quantification`。
- 后端：Python / Matplotlib。
- 图件单位：固定 source ordinal `2` 的一个 `dataset × scale block`；四个
  dataset 各有 `legacy_2_1` 与 `expanded_3_1`，共八张，禁止跨 block
  叠画、投票或聚合。
- panel a：父 scene 的 IVD-p95 等值面与 240 条按 reference class 和几何位置
  预先选择的中心 pathlines；120 positive + 120 negative 只作解释性背景，
  不代表自然类别比例。
- panel b：固定 negative-distance spatial top-5% 候选对全部 valid rows 的
  class assignment；不是 clustering。
- panel c：同一 rows、同一顺序、同一相机和 bounds 的 TP/FP/FN/TN。
- 导出：scene NPZ/manifest、含可编辑文字且三维 marks 栅格化的 SVG/PDF、
  360 dpi PNG、panel-alignment JSON 与 render metadata。

## 预注册的结果边界

`Other_NegativeDistanceSpatial_1.1` 已报告这八组等权宏平均
Accuracy/AP/F1/BA/precision/recall 为
`0.9527/0.5955/0.5451/0.7562/0.5710/0.5350`。该数字已经暴露，
所以本版本不能利用图再次选择候选。图的目的只是显示空间错误结构；即使某一张
图较好，也不能覆盖 Re640 等反例或冒充多 source 聚合证据。

## Ibex 运行与逐图结果

Ibex job `51045480` 在 commit
`520dd9e7fdb2db5be017f0796f0d5f8f6735f8c8` 上完成，节点为
`cn604-05`，使用16个CPU core与64 GB申请内存；elapsed `00:02:37`，
batch MaxRSS `1,991,452 KiB`。job内163/163 tests通过，8张图共覆盖
406,177个valid query rows。

| Flow | Scale block | Valid / 64,000 | Coverage | Accuracy | Average Precision (AP) | F1 | Balanced accuracy | Area Under the Receiver Operating Characteristic Curve (AUROC) | Precision | Recall | TP / FP / TN / FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Re160 | `legacy_2_1` | 60,560 / 64,000 | 94.6250% | 0.9611 | 0.7133 | 0.6451 | 0.7889 | 0.9822 | 0.7064 | 0.5935 | 2,139 / 889 / 56,067 / 1,465 |
| Re160 | `expanded_3_1` | 43,347 / 64,000 | 67.7297% | 0.9481 | 0.6677 | 0.5831 | 0.7363 | 0.9570 | 0.7256 | 0.4874 | 1,573 / 595 / 39,525 / 1,654 |
| Re640 | `legacy_2_1` | 60,555 / 64,000 | 94.6172% | 0.9482 | 0.5406 | 0.5101 | 0.7297 | 0.9416 | 0.5396 | 0.4837 | 1,634 / 1,394 / 55,783 / 1,744 |
| Re640 | `expanded_3_1` | 42,463 / 64,000 | 66.3484% | 0.9379 | 0.4072 | 0.4402 | 0.6867 | 0.9256 | 0.4882 | 0.4007 | 1,037 / 1,087 / 38,788 / 1,551 |
| Re6400 | `legacy_2_1` | 62,313 / 64,000 | 97.3641% | 0.9545 | 0.6232 | 0.5513 | 0.7601 | 0.9719 | 0.5594 | 0.5435 | 1,743 / 1,373 / 57,733 / 1,464 |
| Re6400 | `expanded_3_1` | 57,906 / 64,000 | 90.4781% | 0.9477 | 0.5665 | 0.4945 | 0.7263 | 0.9637 | 0.5117 | 0.4784 | 1,482 / 1,414 / 53,394 / 1,616 |
| Boeing 747 | `legacy_2_1` | 61,432 / 64,000 | 95.9875% | 0.9669 | 0.7341 | 0.6642 | 0.8275 | 0.9771 | 0.6556 | 0.6731 | 2,014 / 1,058 / 57,382 / 978 |
| Boeing 747 | `expanded_3_1` | 17,601 / 64,000 | 27.5016% | 0.9573 | 0.5112 | 0.4722 | 0.7940 | 0.9278 | 0.3814 | 0.6199 | 336 / 545 / 16,514 / 206 |

八组指标逐项重算后均在 `1e-12` 容差内与固定父
`per_group_metrics.csv` 一致。四个flow中，`legacy_2_1` 的F1均高于
`expanded_3_1`；Boeing `legacy_2_1` 最好（F1 `0.6642`），Re640
`expanded_3_1` 最低（F1 `0.4402`）。两个block的valid population不同，尤其
Boeing expanded coverage只有`27.5016%`，因此该差异不能单独归因为尺度方法优劣。

## 产物、哈希与图形QA

- result / visualization / `RUN_COMPLETE.json` 文件SHA-256分别为
  `14190a2ceba035f0cd1f4279eaeaaefe732e8f0519c4d24359b70fc4c931cde6`、
  `cbf0875ecf71c8eb79dea62b9e75669a3f0891f97375d9ffdcdfa2297fbfe3e9`、
  `88bc0e9e5c83265dbbfd431275789f760240cd93c7cf8f41bd10d6aacf4d3274`。
- stdout / stderr SHA-256分别为
  `cebd75037b5ec06294b4f0a1347259ffb082f6e8285fa1bd6b967da067064a0d`、
  `1fc2d6eaa8467332753fcc0eb6dde6a63c7e07a8c6be9c520790e07341f6ab55`。
- 本地下载后，62/62 manifest artifacts的size和SHA-256全部复核通过。
- 科研图源码预检为18 PASS、3个预期WARN、0 FAIL；WARN仅对应360 dpi而非
  600 dpi、没有TIFF以及本版本未声明期刊固定栏宽。
- 8/8 panel alignment在strict模式通过；8/8 PDF的最小文字为7 pt，高于5 pt
  门槛；collision audit为0 FAIL、95 WARN。逐张检查WARN后确认它们都是三维
  坐标刻度与填充/栅格图层边界的预期相交，没有文字互相覆盖、被路径穿过或被
  页面裁剪。8/8原始360 dpi PNG已逐张目视检查。
- Ibex output：
  `/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeDistanceSpatialVisualization_1.1/runs/slurm_51045480_520dd9e7fdb2`。
- 本地下载：
  `outputs/Other_NegativeDistanceSpatialVisualization_1.1_download`。
