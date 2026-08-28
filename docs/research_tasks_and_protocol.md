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
3. scale tuple 必须按对应实验版本的数值三元组判等，不能只按名称判等：1.x 使用 `(neighbor distance, integration step, integration steps)`；2.1 使用 `(neighbor distance dx, RK4 time step ds, target spatial arc length)`。
4. confirmation physical family 必须在完整方法、代码 commit 与 manifest 冻结后首次读取；冻结前不得读取 raw field、query feature、valid rate、标签或指标。
5. 一旦查看 confirmation 标签或指标，该数据以后只能算已暴露 development 数据。修改方法后必须更换新 confirmation。
6. 旧 FMT 的 10 个 flow 条目、Task5 development/confirmation cache 和所有已报告 scale tuple 对本项目都属于已暴露 development 资源；旧缓存中的 `confirmation` 只是历史目录名，不是本项目的 sealed confirmation。

## 4. Primitive 与尺度

- 3D 主任务固定 7 条线：`center, x+, x−, y+, y−, z+, z−`。
- 每条线最终固定 `L=32`；积分器输出 `(x,y,z,t)`，输入 descriptor 前只保留 `(x,y,z)`，时间通道仅用于物理时间和重采样审计。
- 1.x 的相对 tuple 固定为 `(offset_grid_scale, dt_scale, integration_steps)`；2.1 的相对 tuple 改为 `(dx_grid_scale, ds_frame_scale, target_arc_length_grid_scale)`。逐数据集必须另存实际邻居距离、物理 RK4 时间步长、实际积分步数、累计空间弧长、终止时间、重采样方法和七条线的有效状态。
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
7. 每格抽样数固定为 `m=min(512,n_positive,n_negative)`；候选先按 source ordinal 和 seed index 稳定排序，再用冻结 seed 抽样。1.1 在任一类别为空时失败并登记；当前 1.2 必须按第11节将两类都选择0个模板并完整审计，不能只保留非空类或静默丢失证据。
8. Library 和 query 必须携带由完整 encoder 参数计算的 descriptor ID；ID 不同即拒绝比较。

## 8. 指标与统计

- 主要：Average Precision、F1。
- 辅助：Area Under the Receiver Operating Characteristic Curve、precision、recall、balanced accuracy。
- 分解：逐 flow、dataset macro、physical-family macro、逐 scale tuple。
- 不确定性：以 source timeslice 为配对单位的 bootstrap 95% confidence interval。
- 任何 scaler、threshold、descriptor、metric 或 library size 的选择都只能使用 development。

## 9. 证据与运行记录

每项方法结论必须同时指向：experiment version、config SHA-256、Git commit、dataset/cache manifest、逐 query 或逐 timeslice 表、汇总 JSON/CSV、设备，以及 scheduler 实际生成的 stdout/stderr 路径与 SHA-256。runner 自写的摘要文件不能冒充 scheduler 日志。

失败、取消、超时、无效、负结果和反例不得删除。旧结论被修订时，必须并列记录“旧结论、当前结论、改变原因、旧结论错误在哪里”。

所有 Ibex scheduler 进程必须在提交后立即登记到 `docs/ibex_run_registry.md`。不能以最终成功 job 覆盖失败 job。

## 10. `mainExp_TemplateMatching_1.1` development-only 部署边界

`mainExp_TemplateMatching_1.1` 可以先运行 cache-backed development phase，但这不等于完整主实验或 formal confirmation。该阶段只允许读取已经暴露的旧 Task5 cache，运行协议以 `config/mainExp_TemplateMatching_1.1_development.yaml` 为准：

- 旧 development ordinals `0–3` 同时提供非留出 family 的 library 和留出 family 的 seen-scale query；
- 旧 development ordinals `4–5` 只允许 `Verify_...` 使用，禁止进入 mainExp 主指标；
- 旧历史 confirmation ordinals `0–3` 只作为 exposed-development unseen-scale query，不能称 sealed confirmation；
- 主指标标签固定读取 cache `reference`，不得重算或替换；三联图允许从 raw field 重建 IVD-p95，只用于 fail-closed 一致性审计和等值面绘制，审计结果绝不替换主指标标签；
- 七个 physical family 做 leave-one-family-out；library 可按冻结规则平衡，query 必须使用全部 valid primitive 并保持自然类别比例；
- 四臂固定为 constant library prior、672D centered Raw、library-only Raw-PCA 161D、independent FMT 161D；后三个 retrieval arms 使用 exhaustive Euclidean 1NN，prior 是常数分数而不是 1NN；
- bootstrap 固定 seed `25068`、5000次，在每个 physical family 内以 source timeslice 为成对重采样单位；95% 区间固定为 percentile interval，NumPy percentile method 为 `linear`；
- 主表 physical-family macro 固定为“每个 family 内 source-timeslice metric 宏平均，再跨7个 family 宏平均”，以保证 paired bootstrap 与主表估计量完全一致；pooled-query family 指标另报但不冒充该置信区间的点估计；
- 用户尚未冻结置信区间通过规则，因此本阶段只能输出 `descriptive_only_pending_user_ci_decision`，不得宣告主命题通过或失败。

Development 表格必须将 seen-scale 与 unseen-scale 分开，保留所有 flow、family、tuple 和反例。三联图不是汇总证据，也不得按性能挑选：每个数据集在 seen/unseen 两种 regime 都固定 source ordinal `2` 并覆盖全部 scale tuple。三栏固定为：

1. IVD-p95 reference 与 cached center pathlines；raw field 可访问时叠加 whole-loaded-volume IVD-p95 isosurface，否则明确标记 seed-reference fallback；
2. FMT exact-1NN vortex/non-vortex class assignment，不得称为 KMeans 或 clustering；
3. FMT TP/FP/FN/TN，TN 可以淡化。

每图固定显示240条中心线，使用 seed `15068` 的 deterministic stratified maximin 选择；三栏必须共享相同 seed、camera 和 physical bounds。Raw-PCA 必须进入表格，可另做独立比较图，但不进入用户指定三联图。

该 development phase 不得访问任何 sealed confirmation 路径。若 development 结果促使 descriptor、library、normalization、distance、score、threshold、resampling 或数据拆分发生改变，必须创建新的 `mainExp_...` 版本后才能建立 formal confirmation manifest。

## 11. `mainExp_TemplateMatching_1.2` 对 1.1 失败的冻结修订

1.1 job `50930751` 在产生任何性能指标前发现一个 library stratum 缺 positive 并按协议失败。1.1 证据和配置保持不变。1.2 只把空类处理改为：若 `m=min(cap,n_negative,n_positive)=0`，该 `flow×source-time×scale` stratum 两类都选择0个模板，并在 audit/manifest 保留候选数、空类和跳过原因；不得只选择非空类。Query 仍保留全部 valid primitive。Raw-PCA 的无监督拟合和 constant prior 仍使用所有非留出 library-source rows，包括被跳过 stratum；三个 1NN arm 的 scaler 只使用实际平衡模板。其余拆分、方法、指标、bootstrap、三联图和 sealed-confirmation 禁令完全继承 1.1。

## 12. `mainExp_TemplateMatching_2.1` 空间弧长与固定 8:2 流场拆分

2.1 是新的 major iteration，不覆盖 1.2。用户已明确：`ds` 是 RK4 的物理时间步长；`dx` 是中心 seed 到 `x±/y±/z±` 六个邻 seed 的初始距离；第三个尺度是每条 pathline 的目标累计欧氏空间弧长，不是积分步数或时间 horizon。

- 训练流场固定为 `cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400`、`deltaWing_resampled`、`deltaWing_LBM`、`f22raptor`、`channel`、`boeing747`；测试流场固定为 `tangaroa`、`smokeBuoyancy`。两个测试项各自属于 singleton physical family，train/test 无 family 交叉。不得改成 primitive 随机 8:2。
- 三个尺度轴各含 config 中显式列出的10个数值，共1000个 Cartesian tuples；每个 source timeslice 的64,000个 seeds 各只分配一个 tuple，每 tuple 恰好64个 seeds，分配与坐标、velocity、IVD、label、validity独立。
- 每条线 forward RK4 最多前进12个 source-frame intervals。累计 polyline 欧氏弧长首次跨过目标时，在最后 segment 上线性截断到精确目标，再按累计空间弧长等距采样32点。七条线必须全部达到目标；任一出域、到时限仍不足或非有限即整个 primitive invalid。
- 每个数据集选4个 source times，每个 time 加载当前帧及未来12帧；空间每轴用 `ceil(native_count/96)` stride，IVD mean 与 p95 都在完整的 loaded-strided volume 上计算，不得称为 native full-resolution IVD。
- F22 的 `x/y/z/t` coordinate variables 已确认全部 masked；2.1 只允许该 dataset 在 config 中显式使用 integer index coordinates，其他 NetCDF 不得隐式 fallback。`channel` 是 steady VTK 经冻结 Killing observer 产生的 deterministic 159-frame synthetic unsteady view，必须与实测时变流场区分。
- Train library 按 `dataset×source-time×scale×class` 分层；双类非空时才按 negative、positive 顺序各消耗一次 PCG64(15068) 随机抽样，确定性各取1个模板；任一类为空则两类都取0并审计，且该 stratum 不消耗随机数。Test 使用全部 valid primitives 的自然类别分布。匹配跨全部1000尺度做 global exact Euclidean 1NN。
- 用户要求的普通 accuracy 必须报告，但约5%的 IVD-p95 正类会使它偏向多数类，因此同表必须给 prior、balanced accuracy、Average Precision、F1、Area Under the Receiver Operating Characteristic Curve、precision、recall 与 coverage。
- 两个 test flow 的解释性三联图都固定 source ordinal `2`，不按任何性能数字选图。三栏使用同一组全部 valid query seeds：IVD-p95 等值面+240条中心pathlines、FMT exact-1NN template class assignment、FMT TP/FP/FN/TN。第二栏不得称 clustering；240条线只允许用坐标、scale tuple 和 reference class 确定性选择，禁止读取 prediction 或 metric。
- 这10个 flow 都已在 FMT 或1.2中暴露，因此2.1只能产生 exposed-development 描述，不是 sealed confirmation。首次读取 test predictions 前必须冻结 config、代码 commit、portable-window/input manifest，并先通过 `Verify_ArcLengthResampling_1.1`。

2.1 的全部精确数值、哈希、输出和运行规则以 `config/mainExp_TemplateMatching_2.1.yaml` 为准。
