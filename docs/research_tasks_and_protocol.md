# Pathline Template Matching 唯一研究协议

本文件是项目任务、拆分和结论边界的唯一正式定义。其他文档若与本文件冲突，以本文件为准。

## 1. 核心任务

### 三维 Pathline Template Matching：未见流场中的有标签模板检索

长期目标允许连续变化的尺度；首版只研究 config 中冻结的有限尺度 tuple。每个 primitive 统一采样成 `7×L×3`。无可训练参数的 FMT 把每个 primitive 独立编码为 feature vector；有标签 library 保存 feature、IVD p95 标签和来源 metadata；query 通过最近邻检索输出涡/非涡标签。

核心比较：

1. 标签先验；
2. centered Raw geometry + exact one-nearest-neighbor（1NN，一最近邻）；
3. 只在 library 拟合的 161 维 Raw Principal Component Analysis（PCA，主成分分析）+ exact 1NN；
4. independent FMT + exact 1NN；
5. development-only 冻结后才允许加入的其他 descriptor。

允许的核心结论仅为：在明确的数据 family、尺度范围、采样方式、标签和 distance 下，FMT template retrieval 是否优于 Raw retrieval。不得把 FMT Task5 的监督网络结论写成模板检索结论。

## 2. 实验版本

- 论文核心实验：`mainExp_[name]_x.y`
- 组件或设计验证：`Verify_[name]_x.y`
- 非核心探索：`Other_[name]_x.y`
- 消融：`Ablation_[name]_x.y`

小数点前为 major iteration，小数点后为 minor iteration。任何会改变 feature、label、split、library population、normalization、distance、score 或评价数据的修改都必须产生新版本和新输出目录。

## 3. 数据与拆分纪律

1. 主开发评测按 physical family 做 leave-one-family-out，不允许随机拆 primitive 或空间 seed。
2. 一个 source timeslice 及其全部 pathline future window 必须属于同一拆分。
3. scale tuple 以数值三元组 `(neighbor distance, integration step, integration steps)` 判等，不能只按名称判等。
4. confirmation physical family 必须在完整方法、代码 commit 与 manifest 冻结后首次读取；冻结前不得读取 raw field、query feature、valid rate、标签或指标。
5. 一旦查看 confirmation 标签或指标，该数据以后只能算已暴露 development 数据。修改方法后必须更换新 confirmation。
6. 旧 FMT 的 10 个 flow 条目、Task5 development/confirmation cache 和所有已报告 scale tuple 对本项目都属于已暴露 development 资源；旧缓存中的 `confirmation` 只是历史目录名，不是本项目的 sealed confirmation。

## 4. Primitive 与尺度

- 3D 主任务固定 7 条线：`center, x+, x−, y+, y−, z+, z−`。
- 每条线最终固定 `L=32`；积分器输出 `(x,y,z,t)`，输入 descriptor 前只保留 `(x,y,z)`，时间通道仅用于物理时间和重采样审计。
- 相对 tuple 固定为 `(offset_grid_scale, dt_scale, integration_steps)`；逐数据集必须另存实际邻居距离、物理时间步长、积分步数、总积分时间、重采样方法和有效线长度。
- scale assignment 必须与空间位置和 IVD 标签独立；过滤出域 primitive 后还要报告每个 `scale × class` 的 assigned、valid 和 invalid 数量。
- rounded index、物理时间插值和弧长插值是三个不同方法版本。替代方法先用 `Verify_...` 检验；若被主方法采用，必须再升级 `mainExp_...`。

## 5. 标签

首个主实验固定 `whole_loaded_volume_ivd_p95`：

```text
IVD = ||curl(v) − spatial_mean_loaded_volume(curl(v))||
positive iff IVD(seed, seed_time) >= percentile95(IVD volume)
```

必须在 config 和 cache metadata 中记录空间 stride、坐标、边界、IVD percentile、数值 threshold 和正类比例。原始全分辨率 IVD 与 stride-loaded-volume IVD 不得混名或混表。

## 6. 独立 query 要求

主 descriptor 必须满足：同一个 primitive 单独编码、与任意其他 primitive 合批编码、或按不同 chunk 编码，输出在冻结容差内相同。

任何使用 batch mean、Batch Normalization running statistics、可训练 `torch.nn.Parameter` 或 query-set statistics 的 encoder 都不能称为“独立 training-free template descriptor”。它可以进入单独版本，但必须改名并写清最小 query batch 契约。

## 7. Library 与匹配

1. Library 按 `flow × source time × scale × class` 进行冻结的数量控制；不得让 query/test 决定采样数。
2. Feature normalization 只能由 library 拟合并序列化；query 不得更新。
3. 1NN 基线使用 exact Euclidean distance；近似索引必须先证明 recall，并作为新版本。
4. 二分类 score 固定为 `d(nearest negative) − d(nearest positive)`；score 大于 0 判正类，完全等距的 0 固定判负类。
5. top-k、distance weighting、feature block weighting 和 reject threshold 都属于需要 development-only 验证的新版本。
6. 每条命中必须返回最近模板的 dataset、family、source time、seed、scale tuple、label 和距离，支持错误分析。
7. 每格抽样数固定为 `m=min(512,n_positive,n_negative)`；任一类别为空则该格失败并登记，不能静默跳过。候选先按 source ordinal 和 seed index 稳定排序，再用冻结 seed 抽样。
8. Library 和 query 必须携带由完整 encoder 参数计算的 descriptor ID；ID 不同即拒绝比较。

## 8. 指标与统计

- 主要：Average Precision、F1。
- 辅助：Area Under the Receiver Operating Characteristic Curve、precision、recall、balanced accuracy。
- 分解：逐 flow、dataset macro、physical-family macro、逐 scale tuple。
- 不确定性：以 source timeslice 为配对单位的 bootstrap 95% confidence interval。
- 任何 scaler、threshold、descriptor、metric 或 library size 的选择都只能使用 development。

## 9. 证据与运行记录

每项方法结论必须同时指向：experiment version、config SHA-256、Git commit、dataset/cache manifest、逐 query 或逐 timeslice 表、汇总 JSON/CSV、设备、stdout/stderr。

失败、取消、超时、无效、负结果和反例不得删除。旧结论被修订时，必须并列记录“旧结论、当前结论、改变原因、旧结论错误在哪里”。

所有 Ibex scheduler 进程必须在提交后立即登记到 `docs/ibex_run_registry.md`。不能以最终成功 job 覆盖失败 job。
