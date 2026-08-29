# mainExp_TemplateMatching_3.1：H48 下两尺度 block 的 2000-tuple 检索

状态：**`development_completed_confirmation_not_run`**。冻结配置仍是 `config/mainExp_TemplateMatching_3.1.yaml`；数值运行使用 Git commit `260a07ad380d64fc300cabe8926244e92d8ba04a`。Ibex job `50999189` 已完成，`result_manifest.json` 与 `RUN_COMPLETE.json` 均通过逐文件哈希验收。Config中的`performance_results_exist_at_config_freeze: false`只记录冻结时事实，不是运行后状态。结果来自已暴露的10-flow development资源，formal confirmation没有运行。

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

## 已完成运行与证据

完整流程已按冻结顺序执行：Phase A、32个train windows preflight、32个train caches、Phase B、40个全体windows preflight、8个test caches、GPU evaluator。执行副本 `50999189` 在 `gpu510-32` 的 Tesla V100-PCIE-32GB 上于 2026-08-30 00:29:49–00:46:39 +03:00 完成，`ExitCode=0:0`，102/102 tests及CUDA确定性门禁通过。排队重复副本 `50999097` 在执行副本通过启动门禁后被取消，未分配节点、运行时间为0，保留在job registry中。

| Evidence | Value |
|---|---|
| Numerical Git commit | `260a07ad380d64fc300cabe8926244e92d8ba04a` |
| Main config SHA-256 | `771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e` |
| Dataset registry SHA-256 | `5a0cdb522e2b947828e70be1109d32df75156fc0071802654e60897ccf81bfb9` |
| Successful Slurm job | `50999189`, Tesla V100 32GB, 16 CPU, 128GB, elapsed `00:16:50` |
| Input/cache manifest SHA-256 | `8c8e6c8c2fe33e9d023aae62c474069dc3a097720246671c59fb64dc6433d0ee` / `a22fbfc2ce9c8136606e193411f382588954cf9578dd5b026c453c817c7f6895` |
| Scale/library manifest SHA-256 | `5ae302683e7b2307927a120070cb35ba8f159162243c016529a8c4d824084b33` / `a21b3b7c44b2fbad3aa6ca2d15514bba79a1c410dfeeba704ed61c7f221372d4` |
| Result manifest file SHA-256 | `56c597ce70f16847d208b3ea41132e1d3804ff5baec704e46c7c5d989c536142` |
| Result manifest content SHA-256 | `b93981d8f3d139a8cf8c5e50344a6c311a52b1ca37d7a06ebf395cce94f53423` |
| Visualization manifest file/content SHA-256 | `e0a936aafd1d4bbf28644a95de29e9e5c927e9629d0649fe41645140b4d2b89f` / `d412d4d11faf04484c2295b2fd9a6195e7bdf75ed6dcb8b2cc47c10516bf06cd` |
| `RUN_COMPLETE.json` SHA-256 | `f68bc9aea80b28ff5110b5aa53504be2d2e702a9942d2efc2242507f3a193896` |
| Scheduler stdout/stderr SHA-256 | `6377a3469babb4741688ce68675e744644a13215d7dfc433348709120eaa9f9a` / `c47846812e3ea68ef54b5141a0714b68a979188483eb5066b3bc5b382dd6788f` |
| Immutable run | `/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/runs/slurm_50999189_260a07ad380d` |

远端和完整本地副本都通过59/59文件集合、57/57 manifest artifacts的size与SHA-256核验；CSV行数为主表8、source-time 32、dataset 8、family 8、block 8、tuple 8000、bootstrap 21。结构化摘要见[mainExp_TemplateMatching_3.1_ibex_summary.json](evidence/mainExp_TemplateMatching_3.1_ibex_summary.json)。

## 模板库与 query population

3.1没有追加2.1模板，而是从3.1 train population重建Principal Component Analysis、scalers、prior与library。训练有效候选为2,967,612/4,096,000；Raw-PCA使用全部2,967,612个有效train candidates拟合。最终模板库共96,160个模板，正负各48,080：legacy block为60,104，expanded block为36,056。

Test共有1,024,000个assigned rows，其中478,521个七线primitive有效，整体coverage为46.7306%。覆盖率高度依赖流场：Tangaroa为455,800/512,000=`89.0234%`，Smoke buoyancy为22,721/512,000=`4.4377%`。按block pooled coverage为legacy `52.3193%`、expanded `41.1418%`。因此任何只报pooled accuracy而不报coverage和逐流场结果的结论都不完整。

## 主结果：physical-family/source-time等权宏平均

下表是预注册主点估计；`sample_count=478,521`，coverage均为0.467306。Prior在默认判定下全部预测非涡，因此其accuracy受多数类比例影响，但F1、precision、recall均为0。

| Method | Accuracy | Average Precision | F1 | Balanced accuracy | AUROC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Label prior | 0.7429 | 0.2571 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| Raw672 exact 1NN | 0.3987 | 0.2753 | 0.3505 | 0.5070 | 0.4871 | 0.2530 | 0.8247 |
| Train-only Raw-PCA161 exact 1NN | 0.5912 | 0.4366 | 0.3998 | 0.6004 | 0.6507 | 0.2882 | 0.7952 |
| FMT161 exact 1NN | 0.6041 | 0.3621 | 0.3787 | 0.5704 | 0.5215 | 0.2805 | 0.6825 |

### Paired bootstrap：FMT减去比较方法

下面的95%区间是**方法差值**的source-timeslice paired bootstrap percentile interval，5000次、seed `25068`；它不是单个方法自己的置信区间。

| Metric | FMT − prior | FMT − Raw672 | FMT − Raw-PCA161 |
|---|---|---|---|
| Accuracy | −0.1388 [−0.1940, −0.0906] | +0.2054 [+0.1829, +0.2326] | +0.0129 [−0.0117, +0.0421] |
| Average Precision | +0.1050 [+0.0793, +0.1321] | +0.0868 [+0.0734, +0.0981] | −0.0745 [−0.0927, −0.0507] |
| F1 | +0.3787 [+0.3506, +0.4010] | +0.0282 [+0.0210, +0.0357] | −0.0211 [−0.0367, −0.0002] |
| Balanced accuracy | +0.0704 [+0.0530, +0.0832] | +0.0634 [+0.0555, +0.0722] | −0.0300 [−0.0547, +0.0038] |
| AUROC | +0.0215 [−0.0262, +0.0557] | +0.0345 [+0.0113, +0.0574] | −0.1291 [−0.1464, −0.1072] |
| Precision | +0.2805 [+0.2488, +0.3073] | +0.0275 [+0.0223, +0.0324] | −0.0077 [−0.0194, +0.0072] |
| Recall | +0.6825 [+0.6298, +0.7195] | −0.1422 [−0.1715, −0.1176] | −0.1127 [−0.1380, −0.0715] |

## 逐测试流场与尺度block结果

以下两表是pooled-query描述结果，不使用主bootstrap区间，不应冒充主估计量。

| Dataset | Method | Coverage | Accuracy | AP | F1 | Balanced accuracy | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Tangaroa | Prior | 0.8902 | 0.9505 | 0.0495 | 0.0000 | 0.5000 | 0.5000 |
| Tangaroa | Raw672 | 0.8902 | 0.3534 | 0.1680 | 0.1024 | 0.5393 | 0.6133 |
| Tangaroa | Raw-PCA161 | 0.8902 | 0.6958 | 0.3278 | 0.1775 | 0.6805 | 0.7473 |
| Tangaroa | FMT161 | 0.8902 | 0.7530 | 0.2820 | 0.1833 | 0.6616 | 0.5971 |
| Smoke buoyancy | Prior | 0.0444 | 0.5475 | 0.4525 | 0.0000 | 0.5000 | 0.5000 |
| Smoke buoyancy | Raw672 | 0.0444 | 0.4330 | 0.3636 | 0.5904 | 0.4737 | 0.3628 |
| Smoke buoyancy | Raw-PCA161 | 0.0444 | 0.4746 | 0.5276 | 0.6134 | 0.5133 | 0.5536 |
| Smoke buoyancy | FMT161 | 0.0444 | 0.4498 | 0.4251 | 0.5686 | 0.4803 | 0.4567 |

| Scale block | Method | Coverage | Accuracy | AP | F1 | Balanced accuracy | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| legacy 0–999 | Raw672 | 0.5232 | 0.1652 | 0.2318 | 0.1399 | 0.4692 | 0.6669 |
| legacy 0–999 | Raw-PCA161 | 0.5232 | 0.6619 | 0.3501 | 0.2618 | 0.6953 | 0.7714 |
| legacy 0–999 | FMT161 | 0.5232 | 0.7386 | 0.3036 | 0.2745 | 0.6784 | 0.6083 |
| expanded 1000–1999 | Raw672 | 0.4114 | 0.6013 | 0.2533 | 0.1585 | 0.6574 | 0.7000 |
| expanded 1000–1999 | Raw-PCA161 | 0.4114 | 0.7150 | 0.4691 | 0.2181 | 0.7372 | 0.8127 |
| expanded 1000–1999 | FMT161 | 0.4114 | 0.7387 | 0.3588 | 0.2168 | 0.7174 | 0.7118 |

Expanded block的pooled FMT Average Precision、balanced accuracy和AUROC高于legacy block，但F1更低；这些block结果具有不同的有效样本和类比例，没有预注册paired block置信区间，因此不能据此宣告“长弧一定改善FMT”。本实验也不是“1000模板库对2000模板库”的因果对照：三个检索臂都查询同一个global 2000-scale库，block表只按query所属block分解；相对2.1，3.1的H48、49帧source windows、center margin和拟合population也同时改变。覆盖率反例更明确：Tangaroa expanded coverage为81.7109%，Smoke expanded coverage仅0.5727%。Smoke在arc `72.5556`只有1/25,600个assigned rows有效，在arc `80`为0/25,600；Tangaroa在arc `80`仍有17,744/25,600=`69.3125%`有效。这说明H48使长弧在部分流场可执行，但固定arc 4–80不能在所有流场保持相近coverage。

## 四张解释性三联图

四张图均使用固定source ordinal `2`，不是按性能选图。每图三栏依次为：(a) whole-loaded-volume IVD-p95等值面与240条中心pathlines；(b) FMT global exact 1NN模板类别分配，红色为预测涡、蓝色为预测非涡；(c) 与IVD-p95标签的TP/FP/FN/TN分解，其中TP为红圆、TN为淡蓝圆、FP为紫色三角、FN为橙色`x`。同一图的三栏共享完全相同query rows、顺序、相机和bounds；两个block从不叠画或投票。240条展示线由与prediction/metric无关的确定性maximin规则在IVD reference正负类中各选120条，只是解释性抽样，不代表自然类比例。Legacy与expanded具有各自的valid-query总体和展示线集合，不是同一批seed的配对前后图；marker面积与透明度也不表示频率，定量解释必须使用计数表。

| Dataset × block | Source index | Valid / 64,000 | Coverage | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tangaroa × legacy | 101 | 61,656 | 96.3375% | 1,733 | 13,570 | 44,851 | 1,502 |
| Tangaroa × expanded | 101 | 51,947 | 81.1672% | 1,859 | 14,943 | 34,557 | 588 |
| Smoke buoyancy × legacy | 74 | 5,710 | 8.9219% | 1,995 | 2,727 | 506 | 482 |
| Smoke buoyancy × expanded | 74 | 369 | 0.5766% | 201 | 98 | 26 | 44 |

每图的scene NPZ、SVG、PDF、360-dpi PNG、alignment JSON及两份附加审计文件均由result manifest锚定。下载后的四份独立alignment strict audit均为PASS；PDF最小glyph为7pt。碰撞审计为0 FAIL，9–14个WARN均经overlay逐图复核为3D坐标刻度与坐标面/栅格边缘的预期接触，无标题或标签裁切。这些post-download QA文件位于独立QA目录，不属于59-file immutable run，也不冒充Slurm artifacts。冻结图适合实验报告，但不是Nature投稿版：21英寸画布缩到期刊双栏后字体过小，且PDF/SVG内三维栅格层约100dpi。若要投稿，应另建`Other_MainExp31FigureLayout_1.1`从immutable scenes重渲染，不得覆盖本run。

## 当前可支持的结论

先前状态是“方法已冻结、没有3.1性能证据”；当前状态是“完整exposed-development运行和四图已通过哈希验收”。改变原因是job `50999189`成功完成，而不是修改了冻结方法。

1. 扩展库已实际建立：2000尺度、H48、96,160个平衡模板和478,521个有效test queries均有逐文件证据。
2. FMT161相对未经PCA的Raw672，在Accuracy、Average Precision、F1、balanced accuracy、AUROC和precision上的主差值区间均高于0；recall低于Raw672。
3. FMT161并未优于强Raw-PCA161基线：Raw-PCA的Average Precision、F1、AUROC和recall差值区间优于FMT；Accuracy、balanced accuracy和precision差值区间跨0。不能把本实验写成“FMT模板匹配总体最佳”。
4. Prior的高accuracy来自多数类，F1/precision/recall为0；因此普通accuracy不能单独代表涡检索质量。
5. 长弧block能产生大量有效模板和Tangaroa queries，但在Smoke buoyancy上覆盖率坍缩。这个反例必须保留，不能从当前已暴露test反向删尺度或调horizon。
6. 本实验只评估global 2000-scale库，并非1000库与2000库的单变量因果对照；不能把2.1/3.1或legacy/expanded描述差异直接归因于模板库扩容。
7. 这些结论只适用于当前8:2拆分、2000 tuples、H48、IVD-p95标签和exact 1NN。formal confirmation未运行；未来若改方法，必须使用新版本和新的未读physical families。
