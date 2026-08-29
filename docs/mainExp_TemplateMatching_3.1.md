# mainExp_TemplateMatching_3.1：H48 下两尺度 block 的 2000-tuple 检索

状态：**`frozen_pre_run_not_run`**。本实验的唯一冻结配置是 `config/mainExp_TemplateMatching_3.1.yaml`。尚未运行 `Verify_LongArcHorizon_1.1`，尚未提交 3.1 Slurm job，也没有 3.1 accuracy、coverage、匹配或图件结果。Config中的`performance_results_exist_at_config_freeze: false`只记录冻结时事实；正式运行后的状态必须由`result_manifest.json`与`RUN_COMPLETE.json`判定，不得把该预运行字段复制成运行时结论。

## 为什么是 3.1，而不是修改 2.1

2.1 的最大积分 horizon 是 12 个 future-frame intervals，每个派生窗口含 13 帧。3.1 将 horizon 改为 48、窗口改为 49 帧，并用新的 source-index 公式；同时把尺度集合从一个 1000-tuple block 扩展为两个 block 共 2000 tuples，每个 center seed 产生两条 primitive rows。source times、共享 center domain、primitive population、PCA fit population、library 和 query population都会改变，因此这是 major iteration `mainExp_TemplateMatching_3.1`。

2.1 的 config、40 个 cache、模板库、结果、图件和结论保持不可变。3.1 不覆盖、重命名或冒充 2.1；输出写入新的 WekaFS root：

```text
/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development
```

## 证据范围与 8:2 拆分

训练数据仍是 `cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400`、`deltaWing_resampled`、`deltaWing_LBM`、`f22raptor`、`channel`、`boeing747`；测试数据仍是 `tangaroa`、`smokeBuoyancy`。拆分单位仍是完整 physical family，两侧无 family 交叉。

这些流场以及 2.1 的 test 结果都已被读取，所以 3.1 只能产生 **exposed-development** 描述，不能称为 formal confirmation。新增尺度、H48、模板规则或任何其他方法组件都不得从 Tangaroa/Smoke 的 validity、coverage、标签、feature、prediction 或 metric 中选择。新的实验版本不能使已经曝光的 test 恢复为 sealed confirmation。

## H48、49 帧窗口与 source times

每个 source window 含当前帧和未来 48 帧，共 49 帧。对总帧数为 `T` 的均匀时间序列，四个 source indices 固定为：

```text
source_index[k] = floor(k * (T - 49) / 3),  k = 0,1,2,3
```

要求四个 index 唯一，因此 `T>=52`；只有 49、50 或 51 帧并不足以产生四个唯一 source positions。source 选择只读取 time-axis length。完整 49 帧 future window 必须留在同一 partition。

Horizon `48` 是 primitive 与 cache 身份的一部分。即使某个 3.1 tuple 的数值和 scale ID 与 2.1 相同，H48 primitive 也不等于 H12 primitive；2.1 cache 必须被拒绝，不能贴新标签后复用。

## 冻结的 2000-tuple union

三元组仍定义为 `(dx_grid_scale, ds_frame_scale, arc_length_grid_scale)`，数值按 fixed-point 12 位小数精确判等。每个 block 内顺序均为 dx outer、ds middle、arc inner，ID 公式为：

```text
scale_id = scale_id_start + ((i_dx * 10) + i_ds) * 10 + i_arc
```

| Block | Scale IDs | dx values | RK4 ds values | Target arc-length values |
|---|---:|---|---|---|
| `legacy_2_1` | 0–999 | `0.250000000000, 0.361111111111, 0.472222222222, 0.583333333333, 0.694444444444, 0.805555555556, 0.916666666667, 1.027777777778, 1.138888888889, 1.250000000000` | `0.125000000000, 0.144444444444, 0.163888888889, 0.183333333333, 0.202777777778, 0.222222222222, 0.241666666667, 0.261111111111, 0.280555555556, 0.300000000000` | `4.000000000000, 4.888888888889, 5.777777777778, 6.666666666667, 7.555555555556, 8.444444444444, 9.333333333333, 10.222222222222, 11.111111111111, 12.000000000000` |
| `expanded_3_1` | 1000–1999 | `0.125000000000, 0.388888888889, 0.652777777778, 0.916666666667, 1.180555555556, 1.444444444444, 1.708333333333, 1.972222222222, 2.236111111111, 2.500000000000` | `0.050000000000, 0.100000000000, 0.150000000000, 0.200000000000, 0.250000000000, 0.300000000000, 0.350000000000, 0.400000000000, 0.450000000000, 0.500000000000` | `13.000000000000, 20.444444444444, 27.888888888889, 35.333333333333, 42.777777777778, 50.222222222222, 57.666666666667, 65.111111111111, 72.555555555556, 80.000000000000` |

`legacy_2_1` 的 1000 个 tuple 值和 ID `0–999` 必须逐项等于 2.1；`expanded_3_1` 必须包含 1000 个新 tuple，ID 为 `1000–1999`。两个三元组集合必须无交集，union 恰为 2000。运行时禁止重新调用 `linspace`；manifest 必须序列化显式数值并分别哈希 legacy subset、expanded block 和 union。将3.1 IDs 0–999投影为2.1相同的`scale_id, dx_grid_scale, ds_frame_scale, arc_length_grid_scale`四字段行后，`legacy_scale_subset_sha256`必须严格等于2.1 canonical rows hash `d3577011be68ee710d42f65d70ea7791428f71297471ff0468f4980fbfc558f3`。包含`block_id`等3.1字段的legacy hash必须另存，不能冒充parent equality hash；2.1 scale-manifest文件SHA仍为`a407010a56540b4aecd5d81577bcca5b3812fc1e3429964ff1e46ac8707ee8de`。

## 同一 40³ centers 上的双 block assignment

每个 source time 仍只有一个 endpoint-inclusive `40×40×40=64,000` center-coordinate grid。它**不是** `40×40×80` 的 128,000-coordinate 网格。最大 `dx_grid_scale=2.5` 决定两个 block 共用的安全 interior margin，这组相同 center coordinates 同时交给两个 assignment blocks：

- `legacy_2_1` 使用 `numpy.random.Generator(PCG64(15068))`，seed-index 到 scale ID 的 mapping 必须逐项保持 2.1 算法结果；每个source投影出的legacy assignment array按项目`canonical_array_sha256`计算后必须等于`21cdb937f57baf1a786a6a4622870e234074b684e5a5cda4c4271837631e0fee`；
- `expanded_3_1` 使用独立的 `PCG64(35068)`。`35068` 在读取任何 3.1 test 数据前冻结，不由 test 结果选择。

每个 block 都执行 `assignment[permutation[j]]=scale_id_start+(j mod 1000)`，所以每个 scale 在每个 source 恰有 64 rows。每个 center seed 出现两次：一次属于 old block、一次属于 new block；两行可有相同 `seed_index`，必须由 `block_id` 区分。每个 source 共 128,000 primitive rows。

| Population | Assigned rows before validity filtering |
|---|---:|
| 每个 source time | 128,000 |
| 8 train × 4 source times | 4,096,000 |
| 2 test × 4 source times | 1,024,000 |
| 全部 40 source windows | 5,120,000 |

“legacy mapping exact”只表示同一 seed ordinal 到旧 scale ID 的映射算法和结果保持不变。由于 source indices、49帧窗口和安全 margin 都可能不同，它不表示 3.1 legacy primitive rows 与 2.1 cache 字节相同。

## Primitive、标签与方法

物理参数仍为：

```text
neighbor offset   = dx_grid_scale * minimum loaded-strided grid spacing
RK4 time step     = ds_frame_scale * source-frame interval
target arc length = arc_length_grid_scale * minimum loaded-strided grid spacing
maximum time      = 48 * source-frame interval
```

七条 forward RK4 lines、目标弧长精确截断、32点等弧长重采样、fail-closed validity、whole-loaded-volume IVD-p95 标签和 independent FMT161 均继承 2.1。七条线必须全部在 H48 内到达目标；partial primitive、替代 seed 和替代 scale 仍禁止。

四个比较臂不变：train-label prior、Raw672 exact 1NN、train-only Raw-PCA161 exact 1NN、FMT161 exact 1NN。但 3.1 的 prior、PCA、三个 scaler、library 抽样和全部匹配必须从 3.1 train population **重新建立**，不得复用 2.1 fitted artifacts 或模板选择。Library 仍按 `dataset×source×scale×class` 在双类非空 stratum 中正负各选1个，最大模板数由 64,000 增为 128,000；matching 跨两个 block 的全部 selected templates 做 global exact Euclidean 1NN。

## Verify_LongArcHorizon_1.1 前置门禁

3.1 主评测前必须按不可变顺序完成 `Verify_LongArcHorizon_1.1` 的两个 phases：

1. Phase A `synthetic` 不读取真实流场；它把49帧有限dense velocity volume送入与主实验相同的 `UnsteadyVectorField3D` 和 production integrator，解析式只作expected oracle。除验证2000 tuples、old/new IDs、union、两套 PCG64 assignments、H48/49帧身份、等弧长32点、终点截断、batch/order invariance与fail-closed外，还必须验证目标在H12后H48前到达为valid、恰在H48到达为valid、需超过H48为invalid，以及一个time-linear velocity在第13–49帧区间内的解析终点。Phase A全部证据落盘后最后写`SYNTHETIC_PASS.json`；
2. 只有Phase A marker有效并记录其SHA-256后，才可stage train windows；window files必须先发布、dataset manifest最后发布，且preflight必须实际加载并通过8个train manifests和32个windows的size/file SHA-256后最后写`TRAIN_PORTABLES_PASS.json`；只有该marker有效后才可构建恰好`8 datasets×4 sources=32`个H48双block train cache shards；这一阶段不得构建或打开test windows/caches；
3. Phase B `train_coverage`只能读取这32个immutable train caches及sidecars、冻结configs和Phase A marker；32个sidecars必须保留并一致指向同一个已认证`TRAIN_PORTABLES_PASS.json`的path/size/SHA-256。该阶段报告每个`dataset×source×block×scale`的assigned/valid/invalid/coverage与类计数，并在最终`verification.json`中记录Phase A marker SHA-256；
4. `expanded_3_1`的每一个新增arc-length level在所有train dx/ds/dataset/source汇总后至少有1个valid primitive；按冻结library规则，expanded block全局至少产生1个positive和1个negative selected train template；
5. Phase A和Phase B各自使用新的Slurm run directory及completion marker；只有两个markers都存在、两阶段记录同一configs与Git commit hashes、且Phase B记录Phase A marker SHA-256时，Verify才通过。

该诊断只是技术可行性门禁，不是性能实验。不得打开 Tangaroa/Smoke 的文件、manifest、cache、label、validity、coverage、prediction 或 metric；也不得依据诊断删除、重加权或替换 3.1 tuple。门禁失败时保留失败证据并阻止3.1主作业；任何方法修改必须另建版本。

## 指标、统计与三联图

Accuracy、Average Precision、F1、balanced accuracy、Area Under the Receiver Operating Characteristic Curve、precision、recall、coverage、source-timeslice paired bootstrap 及两 test-family 等权宏平均定义均继承2.1。除逐 source、dataset、family 和 tuple 外，3.1 必须单独报告两个 scale block 的 coverage 与指标；invalid rows只进入coverage分母。

三联图仍是 required output，固定 source ordinal `2`，不得按性能或视觉效果选图。但同一个center在old/new block中可能产生不同prediction和confusion，禁止把两个block叠画、聚合或多数投票。因此每个`test dataset × scale block`单独出一张图，共4张：`tangaroa×legacy`、`tangaroa×expanded`、`smokeBuoyancy×legacy`、`smokeBuoyancy×expanded`。每张图只包含该block的全部valid query primitive rows，三栏必须使用完全相同的rows与顺序；三栏仍为whole-loaded-volume IVD-p95+pathlines、FMT template class assignment、TP/FP/FN/TN。每张图的240条展示中心线也只在该block内按reference class预注册选择120 positive+120 negative。第二栏不是clustering，图件不替代汇总指标；visualization manifest必须以`dataset+block`作为唯一键。

每张图都必须导出scene NPZ、含可编辑文字且3D marks栅格化的SVG、同类PDF、360 dpi PNG预览及panel-alignment JSON。SVG是主矢量输出；PDF用于glyph与collision审计；PNG只作360 dpi预览。`visualization_manifest.json`必须为每张图的5个required exports逐文件记录relative path、export kind、size bytes与SHA-256；scene/render metadata可作为附加审计文件单独列出。全局`visualization_manifest.json`不自哈希，其文件SHA-256由最终`result_manifest.json`记录。任一必需文件缺失或hash不符都不得写`RUN_COMPLETE.json`。

## 当前投递状态

当前只冻结了方法与配置。正式投递前必须：

1. 从同一 clean committed revision 通过全部3.1 implementation tests；
2. 在Ibex/WekaFS完成并登记Verify Phase A `synthetic`，取得有效`SYNTHETIC_PASS.json`及其SHA-256；
3. 只生成8个train datasets×4 sources的49帧portable windows：Ibex array只处理当前在Ibex有raw的`cylinder3d, halfcylinderRe640, deltaWing_resampled`；Windows必须在同一clean Git commit和同一Phase A marker下生成`halfcylinderRe6400, deltaWing_LBM, f22raptor, channel, boeing747`，然后先上传window files、最后逐dataset发布manifest，不得对portable root使用删除式同步；只有preflight实际加载并校验8/8 train manifests、32/32 windows、config/registry/commit/size/file SHA-256后写出`TRAIN_PORTABLES_PASS.json`，才可提交32个train cache shards；
4. 只用这32个immutable train caches完成并登记Verify Phase B `train_coverage`，取得最终`verification.json`和`TRAIN_COVERAGE_PASS.json`；
5. Verify两阶段均通过后，才生成2个test datasets×4 sources的8个windows；只有preflight实际加载并校验10/10 manifests和40/40 windows后写出`ALL_PORTABLES_PASS.json`，才可生成8个test caches，至此完整主实验共40个双block cache shards；
6. evaluator可在一个单向自动的manifest冻结步骤中解析immutable cache sidecars，但在`input_manifest.json`写完前不得向操作者报告sidecar值、不得用于任何方法或运行决策，也不得打开test NPZ arrays、计算labels/coverage/predictions/metrics；若冻结步骤中途失败，该run directory保留为未完成证据且不得继续。只有config、Git commit、input/scale/assignment manifests冻结后才可进入数值评估；
7. 将每个Slurm job立即登记到`docs/ibex_run_registry.md`，失败、取消和重投均不得覆盖。

目前没有3.1性能证据，因此不得提前写“长弧长提高/降低accuracy”“2000 scales优于1000 scales”或任何formal confirmation结论。
