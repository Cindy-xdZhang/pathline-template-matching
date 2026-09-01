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
3. scale tuple 必须按对应实验版本的数值三元组判等，不能只按名称判等：1.x 使用 `(neighbor distance, integration step, integration steps)`；2.1与3.1使用 `(neighbor distance dx, RK4 time step ds, target spatial arc length)`。3.1还必须把H48与block ID纳入完整primitive/cache身份；相同三元组不代表H12与H48 primitive相同。
4. confirmation physical family 必须在完整方法、代码 commit 与 manifest 冻结后首次读取；冻结前不得读取 raw field、query feature、valid rate、标签或指标。
5. 一旦查看 confirmation 标签或指标，该数据以后只能算已暴露 development 数据。修改方法后必须更换新 confirmation。
6. 旧 FMT 的 10 个 flow 条目、Task5 development/confirmation cache 和所有已报告 scale tuple 对本项目都属于已暴露 development 资源；旧缓存中的 `confirmation` 只是历史目录名，不是本项目的 sealed confirmation。

## 4. Primitive 与尺度

- 3D 主任务固定 7 条线：`center, x+, x−, y+, y−, z+, z−`。
- 每条线最终固定 `L=32`；积分器输出 `(x,y,z,t)`，输入 descriptor 前只保留 `(x,y,z)`，时间通道仅用于物理时间和重采样审计。
- 1.x 的相对 tuple 固定为 `(offset_grid_scale, dt_scale, integration_steps)`；2.1与3.1的相对 tuple为 `(dx_grid_scale, ds_frame_scale, target_arc_length_grid_scale)`。3.1另以`block_id`区分重复seed-index rows，并将maximum future horizon 48纳入primitive/cache身份。逐数据集必须另存实际邻居距离、物理 RK4 时间步长、实际积分步数、累计空间弧长、终止时间、重采样方法和七条线的有效状态。
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

## 13. `mainExp_TemplateMatching_3.1` 的 H48 与 2000-tuple 双 block 扩展

3.1 是新的 major iteration，不修改或覆盖2.1。改变包括：maximum future horizon从12增至48、派生窗口从13帧增至49帧、source-index公式改变、共享center安全边界按最大`dx=2.5`重建，并在同一center grid上为两个尺度block各生成一行primitive。Horizon `48`必须进入primitive/cache身份；数值相同的旧tuple也不能复用H12 cache。

- 8 train/2 test complete-family拆分、whole-loaded-volume IVD-p95标签、七线forward RK4、32点等弧长重采样、independent FMT161、Raw672、train-only Raw-PCA161、prior、global exact Euclidean 1NN、指标、bootstrap与固定三联图定义继承2.1。
- 每个49帧window的source index固定为`floor(k*(T-49)/3), k=0,1,2,3`，四个index必须唯一，因此`T>=52`。选择只依赖time-axis length。
- `scale_protocol.blocks`按`[legacy_2_1, expanded_3_1]`排序。旧block逐值保留2.1的1000个tuples与IDs 0–999；新block为10×10×10，IDs 1000–1999，dx显式线性值范围`0.125–2.50`、RK4 ds显式线性值范围`0.05–0.50`、目标空间弧长显式线性值范围`13–80`。全部数值以config中的12位小数为准，运行时不得重建`linspace`。两个block必须各含1000个unique tuples、相互无三元组交集，union恰为2000。IDs 0–999投影为2.1相同四字段后的`legacy_scale_subset_sha256`必须等于2.1 canonical rows hash `d3577011be68ee710d42f65d70ea7791428f71297471ff0468f4980fbfc558f3`；带3.1 block字段的hash必须另存，不能冒充parent equality。
- 每个source仍只有同一个endpoint-inclusive`40×40×40=64,000` center-coordinate grid，不是`40×40×80`。最大dx `2.5`决定共享安全margin。每个center在两个block各出现一行；重复`seed_index`必须用`block_id`区分。旧block使用`PCG64(15068)`并逐项保持2.1的seed-index到scale-ID mapping；每个source的legacy assignment按项目canonical-array规则计算后必须等于`21cdb937f57baf1a786a6a4622870e234074b684e5a5cda4c4271837631e0fee`。新block使用在读取test前冻结、与数据无关的独立`PCG64(35068)`。每block每scale恰有64 assigned rows，所以每source共128,000 rows，train/test/总assigned分别为4,096,000/1,024,000/5,120,000。
- “旧mapping exact”不表示旧primitive rows与2.1相同：3.1的source times、49帧window、shared center domain和H48身份均可不同。2.1 portable windows、cache、PCA、scaler、prior、library、prediction和结果均不得冒充3.1产物。
- 3.1的prior、PCA、三个feature scaler、library sampling和全部matching必须只从3.1 train population重新建立。Library仍按`dataset×source×scale×class`双类非空时正负各选1个，最大128,000 templates；test保留两个block的全部valid primitives及自然类分布。
- 混合Windows/Ibex staging必须先发布window files、后发布各dataset manifest。在train cache提交前，`TRAIN_PORTABLES_PASS.json`必须证明同一commit下8个train manifests和32个windows均已实际加载并通过size/file SHA-256；Phase B通过且test windows生成后，`ALL_PORTABLES_PASS.json`必须同样证明10个manifests和40个windows，才可提交test cache。cache sidecar与最终result manifest必须记录对应portable population marker的path、size和SHA-256。
- 3.1 evaluator可在冻结`input_manifest.json`的同一个单向自动步骤中解析immutable cache sidecars，但在manifest写完前不得报告任何sidecar值、不得用于方法或运行决策，也不得打开test NPZ arrays或计算labels、coverage、predictions和metrics。若该步骤中途失败，必须保留未完成run directory并禁止继续。该sidecar例外只服务于输入文件身份和hash冻结，不放宽任何选择或泄漏边界。
- 3.1主评测前必须按不可变顺序通过`Verify_LongArcHorizon_1.1`的两个phases。Phase A `synthetic`不读真实流场，必须把有限49帧dense velocity volume送入与production相同的`UnsteadyVectorField3D`和primitive integrator；解析式只作expected oracle。除2000-tuple union、H48/49帧身份、弧长数值与双block assignment外，还必须验证目标在H12后H48前到达为valid、恰在H48到达为valid、超过H48才到达为invalid，以及time-linear velocity确实使用第13–49帧区间。Phase A最后写`SYNTHETIC_PASS.json`；只有marker及SHA-256有效后，才可构建恰好32个train caches。Phase B `train_coverage`只能读取这32个immutable train caches及sidecars、冻结configs和Phase A marker，必须报告所有`dataset×source×block×scale` strata，要求expanded block每个新增arc level汇总后至少1个valid train primitive，且expanded block全局至少产生1个positive和1个negative selected train template；最终`verification.json`必须记录Phase A marker SHA-256并最后写独立completion marker。两phase markers必须都存在，两阶段须记录同一configs与Git commit hashes，且Phase B须记录Phase A marker SHA-256。整个Verify及train-cache build不得打开任何Tangaroa/Smoke数据或据结果删改3.1 tuple；失败后修改方法必须新建版本。
- 3.1三联图仍是required output，两个test dataset固定source ordinal `2`。每个`test dataset×scale block`必须单独出图，共4张；每图只使用该block的valid query primitive rows，三栏在图内必须使用完全相同的rows和顺序。每图240条展示中心线也必须在该block内仅按reference class与预注册几何规则选择120 positive+120 negative。禁止跨block叠画、聚合、多数投票或按性能选图；visualization manifest以`dataset+block`为唯一键。每图必须导出scene NPZ、含可编辑文字且3D marks栅格化的SVG主矢量文件、用于glyph/collision审计的PDF、360 dpi PNG预览及panel-alignment JSON；visualization manifest须为每图的5个必需导出逐文件记录relative path、export kind、size bytes与SHA-256。全局visualization manifest不自哈希，其文件SHA-256由最终result manifest记录；缺失或hash不符时不得完成主作业。
- 这10个flow和2.1 test结果均已曝光，3.1只能产生exposed-development描述。任何来自2.1 test coverage或metric的设计动机都必须如实记录；formal confirmation需要新的、从未读取的physical families。

3.1的全部显式数值、门禁、WekaFS输出和运行规则以`config/mainExp_TemplateMatching_3.1.yaml`及`config/Verify_LongArcHorizon_1.1.yaml`为准。冻结时状态为`frozen_pre_run_not_run`；随后Ibex job `50999189`在数值commit `260a07ad380d64fc300cabe8926244e92d8ba04a`上完成，运行后权威状态为`development_completed_confirmation_not_run`。性能结论、反例和文件哈希见`docs/mainExp_TemplateMatching_3.1.md`及其结构化证据；不得回写冻结config中的历史pre-run字段，也不得把exposed-development结果称为formal confirmation。

## 14. `Other_MainExp31FamilyHeldOutVisualization_1.1`的四个已暴露train flow分类图

本版本是固定3.1方法的下游解释性实验，不修改primitive、2000个尺度、IVD-p95标签、FMT161 descriptor、Euclidean distance或exact one-nearest-neighbor规则。因`cylinder3d`（Re160）、`halfcylinderRe640`、`halfcylinderRe6400`和`boeing747`都参与了3.1 library构建，禁止直接用现有96,160模板查询这四个flow。

- `holdout_half_cylinder`同时查询三个cylinder datasets，并从library中整体排除`half_cylinder` family；library只含`deltaWing_resampled,deltaWing_LBM,f22raptor,channel,boeing747`。
- `holdout_boeing_747`查询`boeing747`，并整体排除`boeing_747` family；library含其他七个train datasets。
- 每fold必须仅在其合格datasets上从`PCG64(15068)`重新开始3.1的全局模板抽样，并只用新选中library重新拟合FMT feature mean和population standard deviation；禁止过滤或复用3.1已拟合library/scaler/prediction。
- 四个datasets都固定source ordinal `2`。`legacy_2_1`与`expanded_3_1`分开查询和出图，共8张三联图；查询使用当前block的全部valid rows，但每个fold的library仍是跨全2000 scales的全局library。禁止跨block叠画、聚合或多数投票。
- 每图第一栏为whole-loaded-volume IVD-p95等值面加240条仅按reference class与几何位置选出的中心pathlines；第二栏为FMT global exact-1NN template class assignment，不得称为clustering；第三栏为同一批完整query rows的TP/FP/FN/TN。
- 每图报告coverage、accuracy、Average Precision、F1、balanced accuracy、Area Under the Receiver Operating Characteristic Curve、precision、recall及混淆计数。图和表只是`family-held-out exposed-development`；四个flow历史上已暴露，因此不是formal confirmation，也不代替跨source的主实验统计。

详细数值、fold、哈希、输出和执行规则唯一由`config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml`定义；冻结config SHA-256为`6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670`。

## 15. `Other_NegativeDistanceSpatial_1.1` 的负类单类距离机制诊断

本版本只复用 `mainExp_TemplateMatching_3.1` 与 `Other_MainExp31FamilyHeldOutVisualization_1.1` 已发布的逐 query 距离，不重积分、不重建模板库，也不改写任何父实验。两份输入 CSV 的绝对路径、允许数据集和 SHA-256 必须由唯一 config 冻结；哈希或数据集集合不符即失败。

- 基础分数固定为 FMT 最近负模板距离，越大越像涡；禁止把正模板距离重新混入该分数。
- 每个 `dataset×source×block` 单独做稳定百分位排序和 mask-normalized 40³ Gaussian；禁止跨 source、block 或 dataset 传播空间信息。
- sigma 只允许 config 中冻结的网格。无标签预测只允许高分 one-dimensional two-means 与固定 top-5%；后者使用 IVD-p95 的定义先验，不得根据该组真实标签比例改动。
- 两份输入CSV本身含label，reader解析整行后才做列投影，因此不得声称预测阶段没有物理打开label所在文件。预测逻辑的显式投影必须排除reference label，先生成并关闭不含label的预测文件；随后才允许第二次显式reference投影和评测。预测产物不得含label。
- oracle threshold 只报告排序上界，不得选择方法、写入部署预测或作为主结论。
- 所有输入已经暴露，且初步机制结果在冻结前已被查看；因此本版本只能称 `exposed-development mechanism diagnostic`。它不能证明 generalization，也不能替代下一版本 train-only nested complete-family validation。

详细 sigma、列名、输入身份、指标与不可覆盖规则唯一由 `config/Other_NegativeDistanceSpatial_1.1.yaml` 定义；冻结 SHA-256 为 `e891af14037c464a6042143625646be0d2f71c37e5e9ff30e50cc30dd553c141`。

## 16. `Verify_ScaleConditionedRetrieval_1.1` 的 train-only nested physical-family 验证

本版本只允许读取3.1的32个train caches，禁止访问Tangaroa与Smoke Buoyancy。五个train physical families依次作为outer fold；每个outer fold内，另外四个family各作一次inner validation，剩余三个family拟合全部自然负类组成的library和library-only scaler。Outer candidate按“family内`dataset×source×block`等权、再跨四个inner families等权”的F1、Average Precision、balanced accuracy、precision、recall和candidate ID顺序选择。

- Representation固定为FMT161、real-neighbor36和chirality-all35；`k`固定为`1,5,15,31`，查询只允许exact numeric same-scale negative distance。
- Supported-only distance rank按distance和center index稳定排序；unsupported行不删除，score为0并计入全部指标。
- 正sigma仅允许同`dataset×source×block`的support-mask-normalized spatial imputation；输出必须区分supported、imputed和unimputable。插值结果不得称为exact-scale k-nearest-neighbor命中。
- Fixed top-5%与`0.50–0.99`阈值共同构成3060个冻结候选；ineligible行必须判负，fixed-top不能把zero-score行通过center tie选正。
- 每个outer task必须先写入并hash含final scaler/support的`selected_candidate.json`，再首次打开outer feature members；outer prediction projection禁止打开`metadata_json`和`valid_labels`。无标签prediction NPZ与manifest关闭后，必须重新验证manifest自哈希、文件大小/SHA-256、逐数组dtype/shape/SHA-256；只有全部通过后，第二次NPZ projection才能读取label评测。
- 完整classifier因空间处理依赖完整source/block grid，必须称transductive classifier；只有FMT encoder可称per-primitive independent。
- 五个outer fit均须对全部2000尺度支持`k=31`。两个inner fold存在真实支持缺口，必须保留unsupported行和分层指标，不得删尺度、删query或跳fold。
- 达到config冻结的F1/AP/BA/precision/recall门槛只允许进入新的`mainExp_TemplateMatching_4.1`；本版本使用的八个flow都已暴露，不是formal confirmation。

精确候选、split、输入哈希、支持规则、成功条件和输出由`config/Verify_ScaleConditionedRetrieval_1.1.yaml`唯一规定；冻结SHA-256为`f5dbdae08e2e13140245a6a9fd12dba67b4eaf6a7ae1aaea8d600f89a409a6a2`。

## 17. `Other_NegativeDistanceSpatialVisualization_1.1` 的四流场当前候选三联图

本版本是 `Other_NegativeDistanceSpatial_1.1` 固定候选的下游可视化，不重新
选择 score、sigma 或 threshold。唯一候选为最近非涡 FMT 模板距离的组内稳定
rank、同 `dataset×source×block` 的 support-mask-normalized Gaussian
`sigma=1`，以及固定 top-5% positive rule。禁止使用 oracle threshold 或按图
调整任何方法参数。

- query 固定为 `cylinder3d`（Re160）、`halfcylinderRe640`、
  `halfcylinderRe6400` 与 `boeing747` 的 source ordinal `2`。每个 dataset
  分 `legacy_2_1` 与 `expanded_3_1` 单独出图，共八张；禁止跨 block 叠画、
  投票或聚合。
- 八个父 scene 必须来自
  `Other_MainExp31FamilyHeldOutVisualization_1.1` job `51029080`：查询三个
  cylinder 时完整排除 `half_cylinder` family，查询 Boeing 时完整排除
  `boeing_747` family。父 scene 的 bounds、seeds、reference、IVD mesh、240
  条 reference-only 选择的 pathlines、相机与 row identity 不得改变。
- 当前 prediction 必须来自 `Other_NegativeDistanceSpatial_1.1` job
  `51039505` 的已哈希 predictions，并按
  `dataset×source ordinal×block×center index` 与父 scene 精确连接；duplicate、
  missing、extra 或 row identity 不一致均失败。
- 三栏固定为：IVD-p95 等值面与240条中心 pathlines；固定候选对全部 valid
  rows 的 class assignment；同一完整 rows 的 TP/FP/FN/TN。第二栏不得称
  clustering。240条线为120正/120负的解释性背景，不代表自然类别比例。
- 每图必须报告 coverage、Accuracy、Average Precision、F1、balanced
  accuracy、Area Under the Receiver Operating Characteristic Curve、precision、
  recall 与混淆计数，并与父 `per_group_metrics.csv` 中同一固定候选逐组一致。
- 每图必须导出新的 immutable scene NPZ/manifest、SVG、PDF、360 dpi PNG、
  panel-alignment JSON 与 render metadata；全部三维 marks 栅格化，SVG/PDF
  文字可编辑，所有产物禁止覆盖。
- 候选与四个flow指标均已暴露，因此这些图只是
  `family-held-out exposed-development visualization`，不是 formal confirmation、
  无偏模型选择或多 source 聚合证据。

唯一配置为 `config/Other_NegativeDistanceSpatialVisualization_1.1.yaml`，冻结
SHA-256为`82b92a52690eab3883287dc71a8ac2c57a691062188b0629ae83e331c6252c5c`。

## 18. `Verify_NegativeTailCalibration_1.1` 的 fit-negative 逐尺度尾概率验证

本版本必须在读取 `Verify_ScaleConditionedRetrieval_1.1` 的任何 outer 指标前冻结。相对该父验证，唯一数值变化是把 query-group supported-distance rank 替换为 fit-negative-only scale-tail anomaly；自然负类library、global negative-only scaler、三个FMT representation、exact same-scale distance、`k`、spatial sigma、decision grid、nested family split、宏平均与停止规则全部不变。禁止同时加入PCA、逐尺度feature scaler、kinematic feature或跨尺度检索。

- 每个fit-negative row的calibration distance必须显式排除它自己，同时保留其他重复feature形成的零距离邻居；global scaler不做逐行leave-one-out重拟合。
- Tail probability固定为`(1 + count(reference >= d))/(N+1)`，分类异常分数固定为`count(reference < d)/(N+1)`；相同距离按较保守的大tail probability处理。
- Local reference只含同尺度leave-one-out距离；block-other必须排除当前尺度；global-other只在block-other为空时回退。收缩`lambda=64`，由3.1每个source/block/scale的assigned-row设计数冻结，不得由label或结果选择。
- `n_s<k`时不得用pooled tail reference伪造第k近邻距离；`n_s=k`允许exact query distance但没有local reference，固定回退block/global。Retrieval support与calibration support必须分开保存。
- Query-group rank被禁止。Tail anomaly直接进入calibration-support-mask-normalized Gaussian；完整候选因正sigma或fixed top-5%仍可能依赖query group，不能称独立逐primitive classifier。
- Final calibration artifacts与selected candidate必须在任何outer feature member打开前关闭并哈希；prediction artifact认证后才允许读取outer labels。所有失败、回退mode、support状态与哈希都必须保留。

唯一配置为`config/Verify_NegativeTailCalibration_1.1.yaml`；冻结SHA-256为`4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b`。即使达到停止规则，本版本也只使用已暴露train flows，不是formal confirmation。

在读取本版本任何 outer 结果前，额外冻结单折认证与提前停止合同。首折只允许`half_cylinder`；认证器必须绑定外部给定的40位numerical commit、精确13文件、完整inner selection证据与fresh label-free prediction，认证后才可读取label并复算指标。单折不得声称五折成功；只有任一已完成family F1低于0.50、已有两个family F1低于0.65，或将全部剩余family指标设为1后仍不可能达到某个五-family macro门槛时，才可发布`negative_tail_early_stop_certificate.v1`且`stop_version=true`。否则继续运行；成功结论必须来自五个唯一physical families齐全的`complete-five-fold`聚合。

## 19. `Verify_PerScaleNegativeMetric_1.1` 的 fit-negative 逐尺度方差度量

本版本必须在读取 `Verify_NegativeTailCalibration_1.1` 的任何 outer 指标前冻结；配置快照中的历史状态固定为 `frozen_pre_run_not_implemented`，不得为了反映后续实现进度而改写并破坏冻结 SHA-256。当前执行状态为 `completed_stopped_after_authenticated_five_fold_failure`。相对该父验证，唯一数值变化是把 global negative-only diagonal population variance 替换为 fit-negative exact-per-scale shrunk diagonal within-scale population variance；tail calibration、三个 FMT representation、exact same-scale retrieval、`k`、spatial sigma、decision grid、3060 个候选、nested family split、宏平均与停止规则全部不变。禁止扫描 lambda/metric 网格或同时加入 PCA、learned metric、kinematic feature、descriptor 修改与跨尺度检索。

- 精确尺度 `s` 的 local mean 与 variance 只用 fit-family natural negatives，`ddof=0`：`mu_s=sum(x)/n_s`，`v_s=sum((x-mu_s)^2)/n_s`。
- Block-other prior 必须对同 block 其他尺度分别减去各自 local mean，再把 within-scale squared residual 以总 row 数 pooling；禁止包含尺度均值之间的差异。只有 block-other row count 为零时才允许用两个 block 全部其他尺度、按同一 per-scale-centering 公式得到的 global-other prior。
- 固定 `lambda=64` 与 `w_s=n_s/(n_s+64)`，并先在 variance 域收缩：`v_shrunk=w_s*v_s+(1-w_s)*v_prior`。`sqrt(v_shrunk)<1e-12` 的 coordinate 固定使用 effective std `1`；禁止直接收缩 standard deviation。
- Query 与 library 必须使用同一 fit-only exact-scale mean/std。Mean 在同尺度 Euclidean difference 中理论上相消，数值机制只来自 effective std。若两级 broader prior 都空，只允许记录 local-only；若 `n_s=0`，不得用 broader row 伪造 library，该尺度 retrieval unsupported。
- `n_s<k`、`n_s=k`、`n_s>=k+1` 的 retrieval/calibration support、leave-one-out self exclusion、tail probability 方向、tail-reference `lambda=64` shrinkage与 fallback modes逐字段继承父验证；leave-one-out 不得逐行重拟合 scaler。
- Final per-scale scaler、tail calibration 与 selected candidate 必须在任何 outer feature member 打开前关闭、原子发布并认证。Scaler NPZ/manifest 必须保存完整 2000-scale 的 counts、support、mode、mean、local/prior/shrunk variance、effective std、array SHA-256，并由 selected candidate 绑定 scaler 与 calibrator manifest SHA-256；prediction 认证后才允许读取 outer labels。

唯一配置为 `config/Verify_PerScaleNegativeMetric_1.1.yaml`；冻结 SHA-256 为 `b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79`。冻结发生在任何 `Verify_NegativeTailCalibration_1.1` outer 结果可见前。clean deployment commit `e919c2e27b8c8157435d40da350866864721ac51` 的五折/聚合 jobs `51064965/51064966` 已完成认证；五-family macro F1 为 `0.5381077849`，未达到冻结成功规则并停止。该结果只属于已暴露 train flows 的 development 验证，不是 formal confirmation；完整结果、失败链与哈希以 `docs/Verify_PerScaleNegativeMetric_1.1.md` 和运行登记为准。

## 20. `Other_NegativeTailVisualization_1.1` 的四流场当前 NegativeTail 三联图

本版本只为已完成的 `Verify_NegativeTailCalibration_1.1` 生成固定 source ordinal 2 的空间分类图，不重新拟合、选择 candidate、调阈值或按结果选图。

- query 固定为 `cylinder3d` (Re160)、`halfcylinderRe640`、`halfcylinderRe6400` 与 `boeing747`；每个 dataset 对 `legacy_2_1` 和 `expanded_3_1` 分别出图，共八张。
- half-cylinder outer fold 固定为 `chirality_all35, k=15, sigma=1, top-5%`；Boeing outer fold 固定为 `real_neighbor36, k=1, sigma=1, top-5%`。两折不是同一 representation 或同一 `k`，不得为统一图面表述而改写。
- 父 scene 必须逐文件、逐数组认证 `Other_MainExp31FamilyHeldOutVisualization_1.1` job `51029080`。NegativeTail 每折必须认证精确13个文件、预测 NPZ 的18个固定数组及完整候选/校准证据闭环；inner `group_count` 分别冻结为40和56。
- 父 scene 与 prediction 必须按 dataset、source、block、center seed、assigned row、scale ID 与 block index 精确且保序连接；duplicate、missing、extra 或 reorder 均失败。每图重算指标必须与父 `outer_group_metrics.csv` 在 `1e-12` 绝对误差内一致。
- 三栏固定为：IVD-p95 与240条父中心 pathlines；FMT NegativeTail template classification；同一批 rows 的 TP/FP/FN/TN。第二栏不得称为 clustering。
- 每图必须导出 scene NPZ/manifest、SVG、PDF、360 dpi PNG、panel-alignment JSON、render metadata 和 PDF 5 pt 文字审计；八图加六个全局文件构成70个 result artifacts。下载后还必须在本地通过严格面板对齐、PyMuPDF 碰撞审计和八张原始 PNG 目视检查才能交付。
- 固定 top-5% 判决在每个 dataset/source/block 内是 transductive 的；全部流场与 fitted outer-fold classifier 已暴露，因此只是 `family-held-out exposed-development visualization`，不是 formal confirmation 或多 source 汇总证据。

唯一配置为 `config/Other_NegativeTailVisualization_1.1.yaml`；冻结 SHA-256 为 `5a82a9d1af406043066316262e5dcefb1a0d559f6d66e82da16440a2066df131`。

## 21. `Verify_EarlyOppositePairKinematics_1.1` 的 seed-time 局部运动学增强

本版本配置必须在首次读取 `Verify_PerScaleNegativeMetric_1.1` 的任何 outer
prediction、label、metric 或 aggregate 结果之前冻结。冻结时 PerScale jobs
`51063738/51063753` 已提交但 outer 结果尚未读取；配置中的
`frozen_before_first_read_of_any_per_scale_outer_result: true` 是不可改写的历史事实。
当前执行状态为 `COMPLETED_STOPPED_AFTER_AUTHENTICATED_FIVE_FOLD_FAILURE`。所有历史
合同失败与取消均保留：`51069363/51069364`为canonical JSON对象键顺序错误，
`51069778/51069782`为preparation input schema错误，`51070145/51070165`为runner result
manifest schema错误；三类失败都发生在新fold计算前，不是性能证据。完整字段动态审计后的
clean numerical commit为`2c3774dca0d81db8edd5645e63576526b9e276f7`，定向测试
`19/19 PASS`、统一回归`308/308 PASS`。首折/认证`51070299/51070310`与remaining/
聚合`51070386/51070392`全部完成，五-family macro F1为`0.6391632766`，相对PerScale
父方法`0.5381077849`提高`0.1010554917`，但低于冻结最低`0.70`；只有delta-wing与
Boeing 747两个family达到F1 0.65，macro precision `0.5914188887`也低于0.60。
`all_success_conditions_pass=false`，本版本停止，禁止结果可见后的block weight、阈值或
候选调参。完整哈希与逐family结果见
`docs/evidence/Verify_EarlyOppositePairKinematics_1.1_ibex_summary.json`。

相对 `Verify_PerScaleNegativeMetric_1.1`，唯一数值变化是把同一个固定 4D
seed-time kinematic block 无权重追加到三个父 FMT 表示，形成固定顺序的
`165D/40D/39D` 三个表示。禁止增加 kinematics-only 候选，禁止改变 per-scale
metric、`lambda=64`、tail calibration、exact same-scale retrieval、`k`、spatial
sigma、decision grid、3060 candidates、nested physical-family split、选择、成功或
提前停止规则；也禁止扫描 block weight、log、time window、DFT、mean-vorticity
correction 或任何 whole-volume IVD feature。

- 每个 valid row 的 center 固定为
  `seeds_xyz[valid_assigned_row_index]`；以
  `h=dx_grid_scale[valid_scale_id]×min(Δx,Δy,Δz)` 构造固定顺序
  `center,x+,x-,y+,y-,z+,z-`。七点必须在同一 relative seed time `t0=0`，用与
  production RK4 相同 corner/arithmetic order 的 quadrilinear sampler 从 matching
  train portable frame 0 取 float32 初始速度。禁止用 Raw 第一段或 center line
  times 估计 neighbour velocity。
- 以 float64 计算
  `G[:,x]=(u_x+−u_x−)/(2h)`、对应的 y/z columns，
  `omega=curl(G)`、`S=(G+G^T)/2`、`Omega=(G−G^T)/2`、
  `div=trace(G)` 与
  `Q=0.5(||Omega||_F²−||S||_F²)`；4D 顺序固定为
  `[||omega||₂,||S||_F,signed div,signed Q]`，finite 后 float32 序列化。禁止
  batch/flow statistics、absolute divergence、log 与 mean-vorticity subtraction。
- 不得覆盖或扩展 immutable 3.1 parent cache。必须先冻结精确 32-row、train-only
  kinematic input manifest，再构建 additive
  `pathline_template_matching.seed_time_opposite_pair_kinematics_cache.v1` sidecars。
  Sidecar exact arrays、parent/portable hashes、line/time/interpolation/dtype contract、
  identity join 与 atomic non-overwrite 规则由唯一 config 固定；sidecar metadata
  禁止 label 与 label-derived counts。Input freeze/build 禁止打开 parent
  `valid_labels/reference_labels_all/ivd_values_all/ivd_volume/metadata_json`。
- 任何真实 sidecar 之前必须用 production sampler 通过 affine、translation、
  rotation、strain、expansion、batch/chunk/order invariance、RK4-first-v1 equality、
  forbidden-member access 与 exact-identity join synthetic gate，并最后写 immutable
  PASS marker。
- 每个 outer fold 必须先用 nonouter data 完成 inner selection，再写出并 fresh
  authenticate final per-scale scaler、tail calibrator 与 selected candidate，之后才
  可打开 outer sidecar/FMT feature；outer prediction 再经 fresh recomputation 与
  authentication 后，才允许打开 parent outer labels。五折成功门槛与认证提前停止
  条件逐字段继承 PerScale。
- 只允许 8 个 3.1 train flows 及其 32 个 train shards/windows。Tangaroa 与
  SmokeBuoyancy 的 raw、portable、cache、feature、label、prediction 和 metric 全部
  禁止访问。本版本是 exposed-development direct-kinematic baseline；即使提升，也
  不得称为 FMT-only 或 pathline-history 证据，因为 IVD-p95 与 seed-time curl 有直接
  物理关系。

唯一配置为 `config/Verify_EarlyOppositePairKinematics_1.1.yaml`；冻结 SHA-256 为
`e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767`。完整 sidecar
schema、3060 candidate 合同、outer label gate、成功规则、成本边界与风险说明见
`docs/Verify_EarlyOppositePairKinematics_1.1.md`。

## 22. `Verify_RawPCANegativeMetric_1.1` 的 train-only Raw-PCA 表示对照

本版本与 `Verify_EarlyOppositePairKinematics_1.1` 一样，已在首次读取
`Verify_PerScaleNegativeMetric_1.1` 的任何 outer feature、label、prediction、metric
或 summary 前冻结；冻结时 PerScale jobs 已提交但结果未读。当前执行状态为
`STOPPED_AUTHENTICATED_FIRST_FOLD_F1_BELOW_MINIMUM`：Raw-PCA core、nested runner、aggregator 与四个
Ibex wrapper 已实现；定向测试 `26/26 PASS`，统一回归 `303/303 PASS`。numerical
commit固定为`fd0412dc134da9dba88d71d665fc2ad160e78e06`；首折 `51068864` 与独立认证
`51068901` 均已完成。认证half-cylinder F1=`0.469416`，低于冻结单family最低
`0.50`，数学早停证书为`stop_version=true`；本版本停止，不提交剩余折。该证据
反对Raw-PCA版本达到预注册成功规则，但单折不能给出five-family macro结论。

相对 PerScale 父方法，唯一数值变化是把三个可选 FMT 表示整体替换为单一固定
`raw_pca161`：3.1 cache 的 `raw_features` 是七线 `7×32×3` 坐标减中心线首点后按
C-order flatten 的 float32 672D 向量；Principal Component Analysis（PCA，主成分
分析）维数固定 161，不扫描、不 whitening，也不追加全局 standardization。禁止把
Raw-PCA 作为第四表示加入 FMT 三臂候选，否则结果会混入候选集扩大的选择效应。

- 每个 inner fit 只用对应三个 fit physical families 的全部 valid Raw rows、无视 label，
  以两遍 float64 streaming scatter 独立拟合 PCA；final fit 只用四个 nonouter families
  独立重拟合。全八流场的旧 PCA artifact、inner validation rows 与 outer rows均禁止
  进入 PCA sufficient statistics。
- 冻结 solver 为对称 scatter 的 `numpy.linalg.eigh`，降序 stable order；负特征值容差、
  clamp、首个最大绝对 loading 的 sign convention、float32 mean/components 及精确 transform
  公式都由唯一 config 固定。PCA 后再由 fit-family natural negatives 拟合父方法不变的
  exact-per-scale shrunk diagonal scaler、negative library 与 tail calibrator。
- 表示只有 `raw_pca161`，因此候选恰为
  `1 representation × 4 k × 5 sigma × 51 decisions = 1020`；nested complete-family split、
  两级等权宏平均、tie-break、outer label gate、五折成功与提前停止条件全部继承 PerScale。
- Final PCA NPZ/manifest 必须先原子发布、逐成员认证，再绑定 final scaler、calibrator、
  selected candidate 与 prediction。只有 PCA/scaler/calibrator/selection 全部关闭认证后才
  能打开 outer Raw；fresh label-free replay 完成后才允许读取 outer labels。每折固定 17 个
  文件，禁止覆盖。
- 只允许同一 32 个 3.1 train caches。Tangaroa/Smoke 的文件、manifest、cache、feature、
  label、prediction 与 metric 全部禁止访问；即使成功也只是 exposed-development 对照。

唯一配置为 `config/Verify_RawPCANegativeMetric_1.1.yaml`；冻结 SHA-256 为
`6f4718ce6d6385bd0bd5b41a7a04e74cb8f2064fee64097f162999e9eefe6440`。完整 PCA
算法、family row counts、17-file artifact/label gate、1020 candidates、资源成本和风险见
`docs/Verify_RawPCANegativeMetric_1.1.md`。

## 23. `Verify_DimensionlessDeformationFMT_1.1` 的无量纲 deformation 表示

本版本在读取 EarlyKinematics 或 Raw-PCA 的任何 outer 结果前冻结。相对 PerScale
父方法，唯一变化是把 Raw672 恢复为 `7×32×3`：中心轨迹除以其实际31段空间弧长；
六邻线相对中心的 deformation 除以六个实际初始邻距均值；随后进入不变的 independent
FMT 和父方法原有三个 coordinate subsets。变换严格逐 primitive，不读取 label、IVD、
batch/flow statistics，也不拟合参数。

纯数值核心、Raw-only nested runner、独立aggregator与四阶段Ibex wrapper均已实现并通过
本地合成门禁；runner固定4096-row重编码，aggregator强制fresh Raw→dimensionless→FMT
重放后才允许outer label，四个Slurm阶段固定到同一Rome CPU架构。实现commit为
`9a48650c219f4cada12df722d780ea383e03bb89`。首次Ibex job `51087139`在回归与backend门后、
首个`load_plan`处因runtime-rebound父身份误读而失败；失败早于output创建和任何真实cache、
fit、prediction、label或metric，认证job `51087140`未启动并取消，所以没有性能结论。
validation-only修复commit `46f02e60bb345c4e2f7f6ece6aba88cca09f1f6a`固定父身份与
transaction内loader，修复后runner SHA为`e977fa6754ad3029bfaf3e7e5f5334babac018854daec28a46dc4df11a2e01ea`，
统一回归`339/339 PASS`（2026-08-31，215.429 s）。修复后clean rerun `51088712`
通过身份门，但在首批真实nonouter Raw672的无量纲编码中发现六个实现初始邻距不满足
冻结等距合同；nonouter labels已随cache载入内存，但尚未用于inner metric。它在fit、
candidate selection、prediction、outer cache/label和真实metric之前失败并保留空partial目录，
认证`51088726`未启动后取消。当前没有性能结论。必须先做不读取label的误差分布与producer
来源诊断，禁止静默放宽冻结公式或容差；旧失败和取消记录不得删除。
唯一配置为 `config/Verify_DimensionlessDeformationFMT_1.1.yaml`，冻结 SHA-256 为
`c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18`；完整公式与风险见
`docs/Verify_DimensionlessDeformationFMT_1.1.md`。

## 24. `Other_PerScaleNegativeMetricVisualization_1.1` 的四流场当前方法三联图

本版本只报告已认证的 `Verify_PerScaleNegativeMetric_1.1` 固定 source ordinal 2
predictions，不重新拟合、选择 candidate、调阈值或按图挑结果。三个 cylinder flow 查询时
完整排除 `half_cylinder` family，Boeing 747 查询时完整排除 `boeing_747` family。

- query 固定为 `cylinder3d`（Re160）、`halfcylinderRe640`、
  `halfcylinderRe6400` 与 `boeing747`；每个 flow 对 `legacy_2_1`（原1000尺度）与
  `expanded_3_1`（新增1000尺度）分别出图，共八张。禁止跨 block 选择、投票或聚合。
- 父 scenes 固定来自 Ibex job `51029080`；predictions 固定来自 clean five-fold job
  `51064965`，其 aggregate authentication 为 job `51064966`。报告器必须验证父 scene、
  fold result、completion、prediction NPZ、候选与逐组指标的文件和内容哈希。
- scene 与 prediction 按 dataset、source ordinal/index、block、center seed、assigned row、
  scale ID 与 block index 精确保序连接；duplicate、missing、extra 或 reorder 均失败。
- 三栏固定为：IVD-p95 等值面与240条中心 pathlines；完整 valid rows 的模板二分类；
  同一 rows 的 TP/FP/FN/TN。第二栏是 classification，不得称 clustering。
- 八图保留全部 `406,177` 个 valid rows；source ordinal、pathline 选择、camera、bounds 与
  IVD mesh 均不变。交付前必须通过8/8 panel alignment、PDF 5 pt文字、PyMuPDF碰撞审计、
  PNG解码和逐图目视检查。
- 固定 top-5% 判决依赖完整 query group，因此完整分类器是 transductive。四个flow均为
  exposed development data；这些图不是 sealed confirmation，也不能作为跨 block 因果比较。

权威说明与结构化证据分别为
`docs/Other_PerScaleNegativeMetricVisualization_1.1.md` 和
`docs/evidence/Other_PerScaleNegativeMetricVisualization_1.1_local_summary.json`。

## 25. `Other_EarlyOppositePairKinematicsVisualization_1.1` 的最新方法四流场三联图

本版本只报告已认证的 `Verify_EarlyOppositePairKinematics_1.1` 固定 source ordinal 2
predictions，不重新拟合、选择 candidate、调阈值或按图挑结果。尚未运行的
`Verify_DimensionlessDeformationFMT_1.1` 没有真实 prediction，禁止冒充当前方法；旧
PerScale 图也禁止改名冒充 Early 结果。

- query 固定为 `cylinder3d`（Re160）、`halfcylinderRe640`、
  `halfcylinderRe6400` 与 `boeing747`；每个 flow 对 `legacy_2_1` 与
  `expanded_3_1` 分别出图，共八张。禁止跨 block 选择、投票或聚合。
- 父 scenes 固定来自 Ibex job `51029080`。Early predictions 固定来自同一 numerical
  commit `2c3774dca0d81db8edd5645e63576526b9e276f7` 的 half-cylinder job `51070299`
  和 Boeing task `51070386_4`，完整五折认证 job 为 `51070392`。
- half-cylinder 候选固定为 `chirality_all35_plus_seed4, k=31, sigma=0.5, top-5%`；
  Boeing 候选固定为 `real_neighbor36_plus_seed4, k=31, sigma=0.5, top-5%`。不得为
  图面统一而改写。
- 必须先认证 fold result/completion、19-array prediction manifest/NPZ、候选与逐组指标，
  再按 dataset、source ordinal/index、block、center seed、assigned row、scale ID 与 block
  index 和父 scene 精确保序连接；duplicate、missing、extra 或 reorder 均失败。
- 三栏固定为：whole-volume IVD-p95 等值面与未改变的240条中心 pathlines；完整 valid
  rows 的 EarlyOppositePair template class assignment；同一 rows 的 TP/FP/FN/TN。第二栏
  是 classification，不得称 clustering。
- 每图重算指标必须与 authenticated outer group table 在 `1e-12` 绝对误差内一致。交付前
  必须通过八图 panel alignment、PDF 5 pt文字、PyMuPDF碰撞审计、PNG解码和逐图目视检查。
- 这些图只作 `family-held-out exposed-development fixed-source` 空间解释；单一 source 无
  confidence interval，固定 top-5% 判决依赖完整 query group，且父方法未通过完整五-family
  成功规则，禁止称 sealed confirmation 或独立单primitive classifier。

唯一配置为 `config/Other_EarlyOppositePairKinematicsVisualization_1.1.yaml`；冻结 SHA-256
为 `0b5053cdd2342fcd65950b82f08b520de4c8a2717c44ad15a5d13babd0caf1c8`。

终态为 `COMPLETED_LOCAL_REPORTING_QA_PASS`。报告器由 commit
`7e2d7c7e44385f91414c6e4ec347f88e16da7466` 固定；它认证并连接全部406,177个 valid
rows，生成8张 PNG/SVG/PDF 三联图。8/8 ordered identity join 与 `1e-12` 指标重算、
8/8 panel alignment、PDF文字、PNG解码和逐图视觉检查均通过，collision audit 为0 FAIL；
图内没有独立图例，交付 caption 必须说明 panel b 和 panel c 的颜色/符号。legacy F1
Re160/Re640/Re6400/Boeing=`0.7367/0.5682/0.6981/0.8338`，expanded=
`0.6406/0.4928/0.6610/0.6873`。同一已暴露 source 2 上八行均高于旧 PerScale，只能作
描述性比较；父 Early 五-family macro F1=`0.639163<0.70` 的失败结论不变。权威结果见
`docs/Other_EarlyOppositePairKinematicsVisualization_1.1.md` 与
`docs/evidence/Other_EarlyOppositePairKinematicsVisualization_1.1_local_summary.json`。

## 26. `Other_FirstPrinciplesHeadroom_1.1` 的 score-ordering 与 decision-rule 分解

本版本是已暴露 development 结果的 posthoc 机制诊断，不改
`Verify_EarlyOppositePairKinematics_1.1` 的 FMT 表示、距离、空间平滑或
`spatial_score`。问题固定为：当前 score 的排序是否已支持 F1 接近 0.70–0.80，
剩余损失是否主要来自判决规则。

- 输入只允许五个已认证 Early outer folds：`51070299_0` 与
  `51070386_[1-4]`；必须鲜重放它们的 19-array prediction 和冻结 input/synthetic/
  sidecar manifests。父 five-family macro F1 必须在 `1e-12` 绝对误差内复现
  `0.6391632765825263`，否则失败。
- 四个预注册臂是：父方法 current prediction、只由 inner-family prevalence 定义的
  top fraction、不使用 outer label 的精确一维 two-means high cluster，以及每 outer
  group 使用 label 的 max-F1 oracle。结果必须分 legacy/expanded、family/block 与
  five-family/all-block macro 报告。
- Oracle 只是 score ordering 的诊断上限，不是 classifier，不能部署、命名新方法或
  作为达到 0.70–0.80 的证据。若 oracle 仍低于 0.70，反对继续只调 threshold/
  prevalence，下一版本必须修改 score 或表示；若 oracle 充足但 label-free 臂不足，
  才能在新的预注册 `Verify_*` 版本中研究判决规则。
- Tangaroa 与 Smoke 的 raw、portable、cache、feature、label、prediction 和 metric 全部
  禁止。这些已暴露数据不能用于 formal confirmation 或新方法的无偏选择。

最终冻结 config SHA-256 为
`a76ae95710f72a6432e4d392606fe4ca5ad4c0fb89b8d50e6d3868f546117477`；早期
`f02120bb…` 草案从未提交或运行，因缺失精确 fold/父配置路径和父 F1 复现门
而在首次真实读取前被取代。实现 commit 为
`9a48650c219f4cada12df722d780ea383e03bb89`；Ibex job `51087135`
从reporting commit `2174418a642fd4a41416a7a693b88b8f4b9ea399`完成并通过输出认证。
父current F1精确复现为`0.6391632765825263`；inner-prevalence、two-means、outer-label
oracle的all-block F1分别为`0.6352716877863082`、`0.23611906536172073`、
`0.6736418047419102`。Oracle-current gap仅`0.0344785281593839`，即使不可部署的
逐outer-group max-F1 oracle也低于0.70；因此反对继续只调threshold、prevalence或同一
score的label-free二分，下一验证必须改变score ordering或表示。Legacy oracle的
`0.7148507101403403`只属已暴露分块诊断，不能取代all-block规则或称成功。完整合同与哈希见
`docs/Other_FirstPrinciplesHeadroom_1.1.md`。

## 27. `Other_DimensionlessInputGeometryAudit_1.1` 的无标签真实输入几何审计

本诊断只回答 `Verify_DimensionlessDeformationFMT_1.1` job `51088712` 的真实 Raw672
输入为何违反冻结初始几何合同；它不改变1.1的公式或容差，也不产生分类性能结论。

- 唯一输入是 SHA-256 为 `e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`
  的32-row train-cache manifest。每个 NPZ 必须先认证完整文件 size/SHA，再且仅按顺序打开
  `raw_features, valid_scale_id, valid_center_seed_index, valid_scale_block_index,
  valid_assigned_row_index, seeds_xyz`；label、metadata、FMT、IVD、sidecar、test cache 与
  raw flow 全部禁止。
- 逐row重放 center-origin、axis-support、zero-dx、六距离相等及opposite-pair检查，并反演
  `float32(float32(seed±h)-float32(seed))` 的保守closed rounding envelope。严格正`h`
  量化为零差值仍允许反演；center非零或off-axis不能由该producer解释。
- 逐row存在共同`h`只是必要条件；同一 `dataset×source×exact scale` 的全部rows还必须有
  非空共同`h`交集。只有两级都通过的失败才记为量化可解释。
- 输出目录不可覆盖；两个CSV和自哈希summary先写，production入口在最后一次clean-commit
  门禁通过后才原子写 `RUN_COMPLETE.json`。失败、partial与否定结果都保留。

唯一配置 SHA-256 为
`c874a8d9f6abbab452c6543139073eea2ac88e3db99ea13f78e0c3d43e03f566`。完整合同见
`docs/Other_DimensionlessInputGeometryAudit_1.1.md`。Job `51092739` 已认证完成：2,967,612行中
57,446行失败，全部57,446行满足逐row和同尺度共同`h`量化解释，无不可解释、center、off-axis或
zero-`dx`失败。旧结论“原因未知”因此收窄为“全部观察失败与absolute-float32后中心化一致”；
Dimensionless 1.1仍因冻结输入合同不兼容而停止，任何修复必须新建版本。

## 28. `Verify_ClassConditionalTemplateScore_1.1` 的正负类模板一致度分数

本版本直接继承已认证的 `Verify_EarlyOppositePairKinematics_1.1`，唯一科学变化是把
negative-only anomaly改为每个fit family内正类与负类的exact-scale模板一致度对比；FMT+seed4
三种表示、共享负类逐尺度scaler、nested complete-family split、空间处理、3060候选以及成功和停止门
全部保持不变。

- 每个fit family/class/exact scale分别做第`k`近邻检索；leave-one-out第`k`距离给本地
  upper-tail reference，同family同class的其他尺度只允许作冻结calibration prior，不能进入
  retrieval library。经验一致度为`q=(1+count(reference>=distance))/(N+1)`。
- 同一family的正负retrieval与calibration四项都支持才进入共同集合`J`；分数为
  `S_f=0.5*(1+q_positive-q_negative)`，再对`J`中的families严格等权。Inner门为2/3，final门为
  3/4；这是支持门，不是family投票。Threshold严格使用`S>t`，相等判负。
- Outer feature和label保持两阶段隔离：final scaler/library/calibrator、candidate与support policy
  必须写闭并重新认证，outer prediction也必须写闭并fresh replay后，才允许打开outer labels。
- 任一真实fold前必须先通过固定的Rome CPU资源smoke：只打开f22/channel/Boeing三个fit families，
  用165D表示、`k=31`和全部observed scales构建完整模型，只查询确定性无标签synthetic rows；禁止
  metric、prediction或candidate selection。

最终冻结配置 SHA-256 为
`814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea`。本版本仍在实现和纯合成验证，
尚未打开任何本版本真实数组或产生性能结果；完整合同见
`docs/Verify_ClassConditionalTemplateScore_1.1.md`。

2026-09-01 实现审计已完成；科学字段与冻结 config 字节未变。首次 `sbatch` 在生成 job ID 前因旧
billing account `deepvortex` 不属于当前用户关联而被拒绝，未分配节点、未启动进程、未读取任何实验
数据；当前唯一关联为 `pi-hadwigm||normal`，所以在真实读取前仅修正五个 wrapper 与 runtime account
认证，artifact 将记录实际 `pi-hadwigm`，Conda 环境仍为 `deepvortex`。config SHA-256 继续是
`814f95d2…`。最终 core/runner/aggregator/resource-smoke SHA-256 依次为 `9c009376…`、`e5063887…`、
`49c80993…`、`97f02e58…`。首次实际启动job `51143571` 在427/427作业内测试后、任何数据读取前，
暴露了真实`scontrol -o`末尾空格和逻辑`cpu`请求实际解析为`batch`；当前只规范化终端空白、继续拒绝
嵌入换行，并从权威记录精确认证`batch/pi-hadwigm/rome`及环境变量一致性。调度覆盖后的42/42项
定向测试、原81/81项本版本定向测试、427/427项统一测试通过。实现现在明确排除没有 negative scaler
支持的两类 rows、允许自然缺类 family 由 joint-support 门处理、严格重构 artifacts 与发布 CSV/
certificate/report/manifest/completion，并只从 `scontrol Features=rome` 认证节点约束。Resource
smoke 的三-family/12-shard population 不是完整 final fit 的资源上界，这一证据边界必须保留。
本段仍不构成真实性能证据；`cfa369dd…` 的科学 config/core/runner/aggregator identity 保持有效，
`0e9fe3d4…` execution revision 因pre-data真实调度格式差异被基础设施提交
`30bc5a081b46972b25a0e558cbe5584e582e6410` 替代。首次真实读取只能使用该 detached checkout。

2026-09-01 mandatory resource smoke job `51144198` 已通过public authentication：`0:0`、`00:03:48`、
Slurm MaxRSS `8,066,920K`，PASS/audit SHA=`7748bbfb…/f2d578c6…`。只打开固定三fit families的12 rows，
保留half-cylinder/delta-wing member-open均0，2000/2000 scales和全部exact-path/resource门通过；该结果无
F1、prediction、candidate或性能结论，只允许提交固定half-cylinder首折。

## 29. `Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1` 的停止后 Boeing 单折诊断

`Verify_ClassConditionalTemplateScore_1.1` 的修复后 half-cylinder 首折与独立认证 jobs
`51146327/51146768` 已得到 F1=`0.404461664553<0.50` 和 `stop_version=true`；因此 Verify 剩余
四折禁止运行。用户要求的 Re160/Re640/Re6400/Boeing 当前方法三联图中，前三个 flow 已由该认证
half-cylinder fold覆盖；只允许另建本 `Other` 版本补一个 Boeing outer fold。

- 科学方法逐项继承 Verify config SHA-256 `814f95d2…dea`、core `9c009376…ef99`、runner
  `e5063887…1b48`、修复后 authenticator `77a56193…4e9c`，包括三种表示、共同负类scaler、
  family/class exact-scale fit/calibration、2/3与3/4 support、空间处理、严格`score>threshold`和
  全部3,060候选；任何数值override禁止。
- 本版本只允许 outer=`boeing_747`，fit/inner families固定为half-cylinder、delta-wing、F22和channel。
  不得读取或复用 Verify half-cylinder 的selected candidate、fit artifact、prediction、metric或support；
  Boeing必须重新nested selection、final refit、sealed prediction和outer-label gate。
- Verify stopped release `f8515858…76844` 与resource PASS `3f9197a1…57ea`只作执行授权和来源证据，
  不是数值输入。Tangaroa/SmokeBuoyancy继续禁止。
- 旧resource audit绑定的Verify config绝对路径为
  `/home/zhanx0o/pathline-template-matching-class-conditional-score/config/Verify_ClassConditionalTemplateScore_1.1.yaml`。
  因此该原路径必须保留为clean detached `58b0bc0…` producer；本Other版本的fold/auth/report只允许从
  `/home/zhanx0o/pathline-template-matching-class-conditional-boeing`运行。两个checkout不得相同，且不得用
  `/tmp` clone或alternate worktree冒充旧producer绝对路径。
- 诊断只允许固定source visualization与descriptive error analysis，不评估或输出success、stop、
  five-family macro、formal confirmation或独立test generalization。
- 认证release必须恰为`boeing_outer_summary.csv`、`boeing_diagnostic_report.json`、
  `diagnostic_manifest.json`、`DIAGNOSTIC_COMPLETE.json`四文件，并由公共接口重新认证underlying
  15-file Boeing fold。可视化reporter必须把前三个flow的Verify half release与Boeing的Other release
  保持为两个不同实验身份，只比较不变的scientific projection。

本版本配置 SHA-256 为
`6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856`。Boeing fold job
`51154451`与独立认证 job `51154654`均在exact commit
`6322d16cebe5995c8bcec2b8743e9ce0de9d8304`完成；后者重新认证父resource/停止release、底层
15-file fold、fresh fit/query/support/threshold/outer-label gate和四文件public release。
`DIAGNOSTIC_COMPLETE.json` SHA-256为
`a9bb930c540c366dd9fd9fd040bdca306cbb7a0a2fcd829fe5f307a8e85ad12c`。Boeing认证
Accuracy/AP/F1/BA/AUROC/precision/recall为
`0.933401/0.195483/0.241293/0.630077/0.862723/0.206197/0.301415`；这只支持暴露后的
Boeing描述性诊断，不改变Verify停止结论，也不得与half-cylinder结果平均。完整合同与失败尝试保留规则见
`docs/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.md`及`docs/ibex_run_registry.md`。

## 30. `Other_ClassConditionalTemplateScoreVisualization_1.1` 的双单折分类三联图

状态：**`COMPLETED_LOCAL_REPORTING_QA_PASS`**。本报告不是新的数值训练或五折实验；它只组合两个分别完成并认证的
single-fold release：Verify停止分支的`half_cylinder` release与独立Other的`boeing_747` release。
report-time禁止再次执行public fresh replay；reporter只认证冻结completion→manifest→15-file fold→13
artifact→19-array prediction链，并在打开任何NPZ member前写入完整输入manifest。

- 固定数据集按序为`cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400`、`boeing747`；前三项只来自
  Verify half-cylinder release，后一项只来自Boeing Other release。每个数据集固定source ordinal `2`。
- 每个数据集固定两个scale block：`legacy_2_1`（scale 0–999）与`expanded_3_1`（1000–1999），共
  `4×2=8`张图，不按当前prediction或metric选择场景。
- 每张图固定三联：`IVD p95 + center pathlines`、`FMT class-conditional template-score classification`、
  `TP / FP / FN / TN against IVD p95`。parent scene几何、相机、pathline与reference保持不变，只替换
  prediction及分析metadata；metrics必须从认证prediction与scene reference重新计算，并与父fold逐字段比较。
- 每图固定导出scene NPZ、scene manifest、PNG、PDF、SVG、alignment audit、render metadata共7文件；另有
  7个全局文件，因此原始不可变发布集合为`8×7+7=63`文件。禁止覆盖已存在输出目录。
- 主运行必须Ibex-first，来自push后的clean exact reporting commit；wrapper固定CPU、32 cores、128 GiB、
  Rome、12小时、无GPU，并认证Slurm allocation。production report config必须在提交前以完整绝对路径、完整
  commit和SHA-256单独冻结；不得出现占位值。
- Slurm完成只表示63文件transaction和机器检查通过。本地必须另做8/8 PNG目视检查、PDF/SVG可编辑文字、
  panel alignment、裁切/碰撞与说明文字检查；在这些QA完成前不得称为最终图件。两release不构成完整五折，
  不评估five-family success，不是formal confirmation。

完整已知输入身份、输出合同和提交门见
`docs/Other_ClassConditionalTemplateScoreVisualization_1.1.md`。冻结production config为
`config/Other_ClassConditionalTemplateScoreVisualization_1.1.yaml`，SHA-256为
`c69d4a59b4906a32f6e14e100c2fe553cc110c6c08fdb34842f20e198a504a60`。
首次Ibex job `51155277`在456/456测试与opaque input认证后，因reporter把producer固定的
`inner_family=outer_evaluation_only`误写为`outer`而在渲染前失败；没有生成图或逐图指标。修复只更正报告读取与
重算行的outer-evaluation identity并加入真实metric-CSV回归，不改上述config或任何source数值身份；失败记录见
`docs/ibex_run_registry.md`。第二次Ibex job `51155495`的reporter完成8图/63文件后，wrapper发现
`visualization_manifest.json`的自哈希在`NaN`写盘规范化为`null`前计算，因而Slurm `FAILED 1:0`；该attempt即使
存在图与完成标记也不可接受。下一revision只修JSON-safe自哈希顺序并增加真实持久化回归，不改科学config、数值
输入、prediction、metric、candidate、threshold、support、block、source ordinal、parent scene或renderer。
最小修复commit `0cf30b605e63e0b2b6866e40eb48bff114583a83`先构造实际落盘JSON-safe对象再计算
self-hash；真实`NaN→null`持久化回归、CSV空字段回归、16项定向测试及456/456全套测试通过。

第三次Ibex job `51156521`使用exact reporting commit
`5d3d49eae02b59aae11d399755cee33f3e7884e3`，终态`COMPLETED 0:0`；456/456测试、双release认证、
8组exact join与sealed metric比较、8图渲染、61-artifact/63-file复核全部通过。下载后的本地QA又完成8/8
PNG目视、PDF/SVG文字、严格panel alignment和碰撞overlay检查：collision hard FAIL为0，95个自动WARN均为
三维坐标刻度与fill/image边缘接触，逐张复核后接受。状态因此提升为可交付图件。

固定source ordinal 2的legacy/expanded F1分别为：Re160 `0.5311/0.4089`、Re640
`0.0896/0.0726`、Re6400 `0.3403/0.3247`、Boeing `0.1715/0.2024`；对应coverage分别为
`94.63%/67.73%`、`94.62%/66.35%`、`97.36%/90.48%`、`95.99%/27.50%`。约95%的有效样本为负类，
因此高Accuracy不能替代F1、Balanced Accuracy、Precision和Recall。legacy/expanded有效样本、support和补值率
不同，不构成单变量因果比较；八张图也不改变已认证half-cylinder/Boeing family F1=`0.404462/0.241293`
低于0.70的结论。完整指标、caption、哈希和QA证据见
`docs/Other_ClassConditionalTemplateScoreVisualization_1.1.md`及
`docs/evidence/Other_ClassConditionalTemplateScoreVisualization_1.1_local_summary.json`。

## 31. `Verify_SourceCenteredPairedScaleTemplate_1.1` 的source-centered局部运动学与同中心双尺度融合

本版本相对已认证Early父版本只检验两个预注册机制：把原始seed-time `||curl||`改为按
`dataset×source×block×dx level`无标签估计的`||curl-mean(curl)||`，并在最终决策前融合
同一40³ center的legacy/expanded两条尺度分数。完整合同见
`config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml`和同名实验文档；冻结config SHA-256为
`15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc`。

- 每个mean group必须使用全部6400 assigned rows，不能按pathline validity筛选；sidecar生产禁止
  打开label、IVD、valid mask、FMT、Raw或metadata。均值来自assigned interior grid，不读取
  whole-volume IVD mean。
- fold选择完成前只允许对outer sidecar做文件stat、size与whole-file SHA-256身份认证；禁止解压或
  读取outer NPZ的数组、metadata和group mean。final nonouter model/candidate认证后才能绑定这些成员。
- 三种FMT子集只替换4D seed block；负模板逐尺度scaler、same-scale kNN、negative-tail
  calibration和空间处理继承Early。whole-volume IVD p95不能推出内部中心或valid-row正类率恰为5%；
  top fraction冻结为`0.025/0.04/0.05/0.06/0.075/0.10`并只能由inner families选择。融合权重
  使用不偏向任一block的`0/0.25/0.5/0.75/1`对称网格；候选固定为
  3种表示×4个k×5个sigma×5个融合权重×6个top fraction=1800。
- 完整五个physical family继续执行nested outer/inner拆分与outer-label gate。预测单位是64,000
  unique centers，但inner选择与成功门的分类指标均使用center prediction回填后的精确parent-valid rows，与Early
  control同row比较；combined-valid unique-center另报，全部64,000 centers只作coverage分母。
  Tangaroa/Smoke全部禁止。
- 成功要求完整五折macro F1≥0.70、至少4/5 family≥0.65、任一family≥0.50、AP≥0.60、
  balanced accuracy≥0.70、precision/recall≥0.60、combined coverage≥0.90，且相对Early的
  paired source bootstrap F1差值95%下界>0。
- 新方法用outer flow自身无标签速度估计均值，因此完整classifier是transductive；任何增益必须
  归因于直接局部运动学与多尺度融合，不能冒充独立FMT几何学习成功。
- 同时固定min-`dx`与逐`dx` midrank双尺度平均两个direct top-5%诊断；它们不使用label或模板库，
  只界定表示排序上限，不能满足模板方法成功条件。

本配置在首次读取本版本任何真实source-centered feature、prediction或metric前冻结。完整五折认证
job `51160422`现已完成：primary macro F1/AP/BA/P/R=`0.679390/0.750806/0.858326/
0.648096/0.734997`，family F1 half/delta/F22/channel/Boeing=`0.636556/0.816445/0.579344/
0.572869/0.791734`；相对Early的paired bootstrap F1差为`+0.018611`，95% CI
`[0.012595,0.024880]`。旧结论“首折完成且五折门仍可能通过”只用于释放其余折；完整认证后，
macro F1<0.70且仅2/5 family达到0.65，故`stop_version=true`。direct dx-rank/min-dx macro
F1=`0.858241/0.841770`仍只是不使用模板库的直接运动学诊断，不能替代失败的模板成功门。

## 32. `Other_SourceCenteredPairedScaleTemplateVisualization_1.1` 的单一双尺度中心分类三联图

本版本是 Verify SourceCentered 方法的固定下游空间报告。在读取该 Verify 的任何新 feature、
prediction、metric 或父 scene 数组前，已冻结报告配置 SHA-256
`c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e`。固定 query 为
Re160、Re640、Re6400、Boeing，source ordinal 为2，共4张图而不是8张 block 图。

- 三个panel都必须实际绘制同一父IVD网格，并保持相机和物理边界不变。Panel A同时使用父
  `legacy_2_1` 前120条和 `expanded_3_1` 前120条路径线；两种线型只表示尺度背景，图例固定在
  axes外的figure-level顶部安全区。
- Panel B 只允许 combined-valid unique centers 上唯一的 `paired_prediction`；禁止将 legacy/
  expanded block-specific prediction 绘成两个方法、投票或按结果选块。Panel C 必须对同一升序中心
  做 TP/FP/FN/TN 分解。
- 图中 center 指标和表中主要 valid-row projection 指标必须分开。主要指标按 legacy 父 scene
  顺序再按 expanded 父 scene 顺序 exact join `valid_paired_prediction`，只报告、不绘图；两套指标
  均须在 `1e-12` 内复现 producer 行。每个valid row的prediction/score必须逐位等于相同unique
  center的paired prediction/score，producer行的family/source/arm/population/candidate身份也必须完整一致。
- aggregate/fold/parent 的完整文件身份和报告依赖 SHA 必须先写入 `input_manifest.json`，之后才可
  打开任何 NPZ member。duplicate、missing、extra、center reorder、valid-mask drift、跨 source/family
  或两个 block 重叠中心的坐标/reference差异全部失败。
- aggregate completion/manifest/report必须绑定相同aggregator commit、fold commit和完整
  source-centered evidence；数值aggregator固定为
  `a85c007ef961ce53bb40946ca3f38f033bf7a646`。生产只允许clean exact reporting commit上的
  Rome CPU Slurm wrapper与一个complete-five release，不允许本地生产渲染或未授权fold。
- 机器阶段固定输出 PNG/PDF/SVG、combined scene、render metadata 和 panel 位置证据，并停在
  `complete_pending_local_rendered_qa`。本地必须在同一clean reporting commit上认证auditor SHA，
  逐条处置固定的3个源码warning，并完成1.5 pt严格位置检查、SVG可编辑`<text>`、PDF最小5 pt
  文字检查、collision hard FAIL=0及4/4最终物理尺寸人工复核后，才可另写
  `delivery_qa_summary.json: delivery_status=PASS`；原机器 completion 不改写。
- 现有 producer 的 single-fold public release 只允许首折 half-cylinder，因此四流场报告在当前接口
  下需要 complete-five release 才能合法取得 Boeing。若 Verify 首折停止，必须另建冻结的 Boeing
  诊断实验和公开认证，禁止本报告器私自读取未授权 fold。

完整合同、运行接口和结论边界见
`docs/Other_SourceCenteredPairedScaleTemplateVisualization_1.1.md`。实现阶段只用 synthetic/opaque
fixture 完成15项核心定向测试与8项wrapper合同测试，其中完整synthetic三联图实际通过正式
collision auditor（hard fail=0）；本地标准库测试486/486通过。其后Ibex job `51162501`完成真实
四图机器事务，combined-valid center F1 Re160/Re640/Re6400/Boeing=`0.6830/0.5426/0.6663/
0.8342`。本地4/4 alignment、PDF/SVG文字和最终尺寸目视检查PASS，collision hard fail=0；
`delivery_qa_summary.json`文件SHA-256为
`ebb6b5b8545b85debd7a2a1928c7b71a1de522df0a0e998059781b3652b5aa84`。旧结论“没有实际流场图”
只适用于实现阶段；当前四图可以按暴露开发范围交付，但不是formal confirmation，也不改变完整
五折macro F1=`0.679390<0.70`的失败结论。

## 33. `Verify_SourceCenteredRankLikelihoodTemplate_1.1` 的source-rank双类似然模板

状态：**`FROZEN_PRE_RUN_NOT_RUN`**。父`Verify_SourceCenteredPairedScaleTemplate_1.1`已认证
primary macro F1=`0.6793896155`，而其不使用模板库的direct dx-rank-mean top-5%诊断为
`0.8582408452`。本版本根据这项已暴露诊断预注册一个新的表示和分数：只使用source-centered seed-time
curl的组内rank，以有标签正、负fit中心的直方图log-likelihood ratio（对数似然比）替换父版本高维、
对称的negative-distance score。上述父数字只是设计依据，不是本版本结果。

- 复用父numerical commit `a85c007ef961ce53bb40946ca3f38f033bf7a646`已经认证的32个sidecars；
  父config/input/population SHA-256固定为`15ac5b0e…60cc`、`5f7e567a…fec9`、
  `50d9d53f…97e2`。任何新数组读取前，新版本必须只以stat/size/whole-file hash认证并原子发布
  `parent_sidecar_binding.json`与`BINDING_COMPLETE.json`；禁止改写父manifest或在binding阶段解压
  sidecar member。每折final nonouter model和candidate关闭前，outer sidecar也只允许相同的opaque检查。
- 每个`dataset×source×block×dx`组必须使用全部6,400条assigned rows，包括pathline-invalid rows；
  empirical midrank固定为`(one-based average tie rank-0.5)/6400`。同一中心先以
  `z_w=w*r_legacy+(1-w)*r_expanded`融合，`w={0,.25,.5,.75,1}`。Rank归一化禁止使用validity、label或
  reference，因此是无标签的；但它使用目标source自身统计，所以完整分类器是transductive。
- 模板库只收至少一个block有效的combined-valid centers；同一中心只出现一次，label只由其
  `valid_labels`聚合且重复block必须一致。禁止为invalid center读取`reference_labels_all`。Rank总体仍是
  全部assigned rows；template/query eligibility与rank总体不能混淆。
- Primary先对每个fit family/class分别构建`B={64,128,256}`个等宽bin的自然计数，以
  `beta={0.5,2}`做加性平滑；再按`p_c(b)=mean_f p_f,c(b)`对physical families等权，最后计算
  `ell=log p_positive-log p_negative`。禁止先算family LLR再平均，也禁止按family行数加权。
- 每个fit-negative reference中心必须排除其完整`dataset×source×H48 window`的所有正、负模板后重建
  上述mean-density模型，再计算留一LLR；rank本身不重算。留一分数按所属family形成reference ECDF。
  Outer query使用full-fit histogram；对每个family的异常分数固定为
  `count(reference < query_llr)/(N+1)`，相等不计入，最后family等权。禁止半并列mid-CDF或结果可见后
  改变tie policy。
- 校准分数只在各`dataset×source`的combined-valid 40³ mask内做mask-normalized Gaussian，
  `sigma={0,.5,1}`、truncate=3；阈值`tau={.90,.925,.95,.975,.99,.995}`且严格`score>tau`。
  Primary候选恰为`5 w×3 B×2 beta×3 sigma×6 tau=540`。
- `negative_ecdf`只以fit-family natural negatives的`z_w`做family等权经验CDF control，独立从
  `5 w×3 sigma×6 tau=90`项中做inner选择，不得影响primary或满足success。Direct诊断固定
  `w=.5/top=5%`、在全部64,000 centers上产生无模板prediction，再投影到全部parent-valid rows；它也
  不能满足模板成功条件。
- 完整五个physical family的nested outer/inner拆分、source+H48不可拆、Tangaroa/Smoke禁令、两级
  inner宏平均、outer-label gate、valid-row主指标、combined-center coverage、metric和认证停止规则继承
  SourceCentered父版本。Primary成功门仍为macro F1≥.70、至少4/5 family F1≥.65、任一family≥.50、
  AP≥.60、balanced accuracy≥.70、precision/recall≥.60、coverage≥.90；另要求相对已认证父方法
  F1=`.6793896155`、在exact same valid-row identities上做5,000次paired dataset-source bootstrap，
  F1差95%区间下界严格大于0。单折和两个control都不得宣称成功。

该方法的primary representation不使用FMT或Raw pathline coordinates，只使用直接seed-time local curl。
若提高F1，结论只能是“有标签source-rank likelihood template比父版本高维对称距离更好地保留直接局部
curl的单侧排序”；禁止写成“independent FMT geometry学会涡结构”。八个flow均已暴露，本版本不是新的
formal confirmation。完整数值、父绝对路径、18-file fold合同和输出规则唯一由
`config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml`与同名实验文档定义；冻结config
SHA-256为`41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa`。
