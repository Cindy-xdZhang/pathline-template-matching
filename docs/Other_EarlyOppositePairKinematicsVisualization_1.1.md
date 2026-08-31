# `Other_EarlyOppositePairKinematicsVisualization_1.1`

状态：**`FROZEN_PRE_RUN_NOT_RUN`**。本版本是已认证
`Verify_EarlyOppositePairKinematics_1.1` 的固定下游可视化，不重新训练、拟合、
选择候选、调阈值或按图面挑结果。

## 固定问题与证据范围

在 `cylinder3d`（Re160）、`halfcylinderRe640`、`halfcylinderRe6400` 和
`boeing747` 的固定 source ordinal 2 上，最新已认证模板分类结果在空间中哪里与
IVD-p95 一致，哪里产生 false positive 或 false negative？

这是 `family-held-out exposed-development visualization`：half-cylinder 查询时
完整排除 `half_cylinder` family，Boeing 查询时完整排除 `boeing_747` family；但四个
flow 与拟合后的 outer-fold classifier 都已暴露，不能称 sealed confirmation。

## 冻结图件

每个 flow 的 `legacy_2_1`（scale ID 0–999）和 `expanded_3_1`（1000–1999）分别出图，
共八张。禁止看到结果后删去任一 block，也禁止跨 block 投票或合并。每张图固定三栏：

1. 未改变的 whole-volume IVD-p95 等值面与 reference-only 预选的240条中心 pathlines；
2. 同一全部 valid query rows 的 EarlyOppositePair FMT 模板二分类；
3. 同一 rows、顺序、相机和 bounds 下的 TP/FP/FN/TN 分解。

第二栏是 template class assignment，不是 clustering。三个 cylinder flow 固定使用
`chirality_all35_plus_seed4, k=31, sigma=0.5, top-5%`；Boeing 固定使用
`real_neighbor36_plus_seed4, k=31, sigma=0.5, top-5%`。这些候选来自已完成的 nested
family selection，不为可视化改变。

配置：`config/Other_EarlyOppositePairKinematicsVisualization_1.1.yaml`，冻结 SHA-256 为
`0b5053cdd2342fcd65950b82f08b520de4c8a2717c44ad15a5d13babd0caf1c8`。运行、指标、
图件哈希和交付审计将在实际完成后追加，历史冻结字段不改写。
