# Other_NegativeDistanceSpatialVisualization_1.1：四个指定流场的当前候选分类三联图

状态：**`frozen_pre_run_not_run`**。唯一配置为
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

## 完成后必须补充

- numerical Git commit、Ibex job ID、node 与实际资源；
- scheduler stdout/stderr 路径和 SHA-256；
- 八张逐图指标与父 `per_group_metrics.csv` 的一致性检查；
- visualization/result manifest 与 `RUN_COMPLETE.json` SHA-256；
- 八张 panel alignment、PDF 文字、collision audit 和逐图目视检查结果；
- 本地下载路径。
