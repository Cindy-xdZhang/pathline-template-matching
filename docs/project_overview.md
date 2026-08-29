# Pathline Template Matching：项目总览

## 1. 最根本的研究问题

给定多个已知三维流场，我们从涡区域和非涡区域采集大量 pathline primitive，用已加载空间体的瞬时涡量偏差第 95 百分位（whole-loaded-volume Instantaneous Vorticity Deviation p95，IVD p95）产生二分类标签，再用无可训练参数的 FMT 生成特征库。FMT 沿用原项目名称；由于原文档存在不同历史全称，本项目不展开该缩写。

面对一个未参与建库、方法选择或阈值选择的新流场，同样把 query primitive 编码成特征，检索库中欧氏距离最近的模板，并继承其涡/非涡标签。

本项目要回答的不是“FMT 能否进入分类网络”，而是：

> 在明确冻结的流场 family、邻居距离、积分步长和积分总长度范围内，training-free FMT 特征空间中的最近模板，能否可靠地代表同一种局部流动模式？

## 2. 与 FMT 项目的关系

FMT 项目提供三个已验证的基础部件：

1. 三维 primitive 固定为中心线及 `x±、y±、z±` 六邻居，共 7 条线；各线积分后采样为 32 点。
2. 无可训练参数的三维 Fourier 描述符及其刚体平移、固定旋转不变性测试。
3. Task5 的可变尺度构造：邻居距离、Runge–Kutta fourth-order method（RK4，四阶龙格–库塔积分）的步长和积分步数可变，但网络输入固定为 `7×32×3`。

FMT Task5 的 `mainExp_Task5_3D_1.1` 在 10 个数据条目上支持“FMT 改善可变尺度监督 IVD 分类”，其相对同维 Raw Principal Component Analysis（PCA，主成分分析）residual 的 dataset-macro F1/Average Precision 增益为 `+.0891/+.1116`。这只是新项目的动机证据，不能当作最近邻模式匹配的结果。来源与适用边界见 [source_provenance.md](source_provenance.md)。

## 3. 冻结的数据契约

### 3.1 Primitive

- 3D 线顺序固定为 `center, x+, x−, y+, y−, z+, z−`。
- 1.x旧cache为每个seed分配`(offset_grid_scale, dt_scale, integration_steps)`；当前raw-flow-backed 2.1/3.1使用`(dx_grid_scale, ds_frame_scale, target_arc_length_grid_scale)`，其中第三项是每条线的目标累计空间弧长。
- 线数固定为 7，每线采样点数固定为 32。积分器输出 `7×32×4=(x,y,z,t)`；描述符只读取 `7×32×3=(x,y,z)`，时间仅用于物理时间与重采样审计。
- `mainExp_TemplateMatching_1.1/1.2`沿用FMT Task5的rounded-integration-index采样且保持不变。2.1经`Verify_ArcLengthResampling_1.1`后使用目标弧长精确截断与32点等弧长重采样；3.1经`Verify_LongArcHorizon_1.1`后把最大future horizon从12扩为48个source-frame intervals。不同版本的primitive/cache身份不得混用。
- 当前 FMT 基线的六个邻居共享一个距离。若未来允许六个距离分别变化，必须升方法版本并增加 config 字段和测试。

### 3.2 标签

在 primitive 的 seed time 上计算

```text
ω(x,t) = curl(v(x,t))
IVD(x,t) = ||ω(x,t) − mean_loaded_volume(ω(t))||
label(seed) = IVD(seed) >= percentile95(IVD volume)
```

这里的 “whole field” 实际指 loader 读取并可能按 stride 降采样后的完整空间体，不等于原始全分辨率数据。首版名称因此固定为 `whole_loaded_volume_ivd_p95`，避免混淆。

### 3.3 默认 FMT 描述符

`fmt_independent_3d_161d_sha256_25fce29499c9089e` 使用 6 个 Fourier 频率、Gram rotation invariants、chirality 和跨邻居逐槽排序，宽度为：

```text
每线: 6×3 Gram slots + 5 chirality slots = 23
中心和六邻居: 7×23 = 161
```

它没有可训练参数，也不读取同 batch 的其他 primitive，因此同一个 query 单独编码、混在任意 batch 中编码或分 chunk 编码必须逐位一致。`neighbor_weight=1`、`neighbor_scale=1` 与旧 Task5 cache 的生成代码一致；改动任一参数都会产生新的内容派生 descriptor ID，不能与旧 cache 混检索。

FMT Task5 的主 268 维配方由 `161D base + 63D time-local Gram + 44D kinematic` 拼接。44 维 kinematic 块用当前 batch 的平均涡量；同一 query 会因 batch 组成改变。它不进入本项目首版主基线。

## 4. 模式匹配基线与当前版本

```mermaid
flowchart LR
  A["已知流场：library families"] --> B["可变尺度 7×32×3 primitives"]
  B --> C["IVD p95 labels"]
  B --> D["161D independent FMT"]
  C --> E["分 flow/time/scale/class 平衡建库"]
  D --> E
  E --> F["只用 library 拟合逐维 mean/std"]
  G["未见 physical family queries"] --> H["同一 161D FMT 与冻结 scaler"]
  F --> I["Exact Euclidean 1-nearest-neighbor"]
  H --> I
  I --> J["标签 + 最近模板 metadata + distance margin"]
```

1.1因一个library stratum缺正类而在指标前fail closed；1.2保留旧Task5 cache语义并完成7-family leave-one-out development。随后2.1实现raw-flow-backed空间弧长primitive与固定8:2完整流场拆分；当前完成版本是[mainExp_TemplateMatching_3.1](mainExp_TemplateMatching_3.1.md)：H48、49帧窗口、保留2.1的1000个tuple并新增1000个长弧tuple，重新建立全部train preprocessing与96,160个平衡模板。Ibex job `50999189`完成1,024,000个assigned test rows的评测与4张固定`dataset×block`三联图。

- 1.x模板库在每个`flow×source time×scale tuple`取`m=min(512,n_positive,n_negative)`个正类和负类。2.1/3.1在每个双类非空的`dataset×source time×scale tuple`中正负各取1个；单类stratum两类均不取，但仍保留candidate、coverage和prior审计。
- 标准化的均值和标准差只从 library feature 拟合；query 不得更新它们。
- 匹配器是精确欧氏一最近邻。二分类连续分数为 `最近负类距离 − 最近正类距离`；分数大于零等价于最近模板属于正类。
- 当前主实验不启用unknown/reject threshold；拒识属于后续独立版本。
- 主对照包含 672 维 centered Raw、只在 library 拟合的 161 维 Raw-PCA、161 维 FMT，以及只用未平衡 library 候选标签比例的常数 prior。所有检索对照使用各自 library-only preprocessing 和相同 exact one-nearest-neighbor。

## 5. 数据拆分与证据等级

旧 FMT 的 10 个数据条目和全部 Task5 scale tuple 已被开发过程查看过，只能作为本项目 development 资源，不能称为全项目未见 confirmation。

1.x cache-backed开发评测采用leave-one-physical-family-out。2.1/3.1改为固定8:2完整流场拆分：`cylinder3d, halfcylinderRe640, halfcylinderRe6400, deltaWing_resampled, deltaWing_LBM, f22raptor, channel, boeing747`建库，`tangaroa, smokeBuoyancy`测试；完整physical family不跨两侧。正式confirmation仍必须来自新的、从未读取的flow families；在完整方法、代码commit和manifest冻结前不得读取其raw field、query feature、有效率、标签或指标。

旧cache的train/validation/historical-confirmation分工只适用于1.x。2.1/3.1从每个raw flow冻结4个完整source windows；3.1每窗49帧、同一40³ center grid分配到legacy/expanded两个block，每个source共128,000 assigned rows。

每个版本的query population使用测试侧全部valid primitives，保持自然类别比例，不按标签平衡或下采样；必须报告assigned、valid、invalid和自然正负类数量。

必须同时防止三种泄漏：

1. 同一 physical family 跨 library/query；
2. 同一 pathline 的完整 source window 跨拆分；
3. 同一 scale tuple 在“未见尺度”实验中换名后跨拆分。

## 6. 评价方式

主要指标：Average Precision（AP，按连续分数衡量正类排序）和 F1。辅助指标：Area Under the Receiver Operating Characteristic Curve（AUROC，受试者工作特征曲线下面积）、precision、recall、balanced accuracy。

所有结果都要给出：逐 flow、dataset macro、physical-family macro、逐尺度 tuple，以及按 source timeslice 配对 bootstrap 的 95% confidence interval（置信区间）。不能只报告 primitive-level 随机 bootstrap，因为同一时间片内样本相关。

`mainExp_TemplateMatching_3.1`的exposed-development主估计中，FMT161的Accuracy/Average Precision/F1为`0.6041/0.3621/0.3787`。FMT相对Raw672的多数差值区间为正，但相对Raw-PCA161的Average Precision、F1、Area Under the Receiver Operating Characteristic Curve和recall更差；Smoke expanded-block coverage仅`0.5727%`且arc length `80 h_min`没有有效query。因此结果不支持“FMT总体最佳”或“长弧普遍改善”。完整数值与哈希见[3.1实验文档](mainExp_TemplateMatching_3.1.md)；formal confirmation未运行。

## 7. 当前代码边界

已经迁移并测试：

- 独立 161 维 FMT descriptor；
- 尺度 tuple 校验与均衡分配；
- fail-closed NetCDF window loader；
- whole-loaded-volume IVD；
- 三维RK4、7-line primitive构造、旧rounded-index路径以及当前目标弧长精确截断与`7×32×4=(x,y,z,t)`等弧长重采样；FMT view为前三通道`7×32×3`；
- library-only standardization、精确 1NN、class-distance margin、无 pickle 保存；
- 1.x cache-backed leave-one-family-out evaluator，以及2.1/3.1 raw portable-window、primitive-cache、固定8:2 evaluator、四方法对照、逐query/timeslice/flow/family/block/tuple表、成对bootstrap、反例表和哈希链；
- 固定样本三联图：IVD-p95等值面+pathlines、FMT template class assignment、TP/FP/FN/TN；3.1按`test dataset×scale block`输出4张并保留immutable scenes；
- Ibex原始数据、portable windows、new raw caches与旧Task5 cache的区分验证及分阶段完成门禁。

尚缺新的、从未读取的physical families及其sealed confirmation manifest/first-read流程，所以当前完成的raw-flow-backed development仍不能宣称formal confirmation或整个研究目标完成。

这些应按 [experiment_log.md](experiment_log.md) 中的版本顺序实施，不能直接在旧 confirmation 上调参。

## 8. 文档权威顺序

各文档职责不能互相替代：[research_tasks_and_protocol.md](research_tasks_and_protocol.md) 管研究与泄漏规则；具体版本文档和 config 管该版本方法；[experiment_log.md](experiment_log.md) 管当前结论及修订；[ibex_run_registry.md](ibex_run_registry.md) 管逐 job 原始证据；本 overview 和 README 只负责导航。若同一事项冲突，先停止实验并在这些文件中显式修订，不能静默选一个版本继续。
