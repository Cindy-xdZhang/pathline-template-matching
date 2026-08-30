# `Other_NegativeTailVisualization_1.1`

## 状态与用途

权威运行状态：`COMPLETED_EXPOSED_VISUALIZATION`。冻结 config 中的
`frozen_pre_run_not_run` 是不回写的历史状态，完成证据以本文、运行表和
Ibex completion marker 为准。

本实验只为已经完成的 `Verify_NegativeTailCalibration_1.1` 生成固定
source ordinal 2 的空间分类图。它不重新训练、不重新选择 candidate、不调整
阈值，也不按结果选择流场、source、scale block 或展示 pathline。四个流场和
两个 fitted outer-fold classifier 都已经暴露，因此证据范围只能写成
`family-held-out exposed-development visualization`，不能称为 sealed
confirmation 或无偏 generalization 结果。

冻结 config：`config/Other_NegativeTailVisualization_1.1.yaml`，完整文件
SHA-256 为
`5a82a9d1af406043066316262e5dcefb1a0d559f6d66e82da16440a2066df131`。
loader 在解析任何字段前先验证这个完整文件哈希；任何字节变化都会关闭运行。

## 科研图结论与证据结构

核心结论为：在完整 query physical family held out 的设置中，当前 NegativeTail
FMT-template classifier 能定位 IVD-p95 涡区域，同时 false positive 和 false
negative 呈现依赖流场与 scale block 的空间结构。

结果问题为：固定 source ordinal 2 时，四个流场上的 family-held-out prediction
在哪里与 IVD-p95 一致或不一致？图型为 `image plate + quantification`。每个
`dataset × scale block` 单独出一张三联图，共8张：

- panel a：父 scene 原样复用的 IVD-p95 isosurface、全部 query seed 背景和240条
  reference-balanced center pathlines；
- panel b：同一批、同一顺序 valid rows 的 FMT NegativeTail template
  classification，明确不是 clustering；
- panel c：同一批、同一顺序 rows 的 true positive、false positive、false
  negative、true negative 空间分布。

三栏必须具有相同的物理 bounds、camera 和 evaluated seed coordinates。240条
pathline 为120条 reference positive 加120条 reference negative 的解释背景，
不代表自然 query prevalence。单个预注册 source 不报告 confidence interval。

## 固定 outer-fold candidates

8张图不是同一个 representation 或同一个 k：这是已完成 nested outer-fold
方法的一部分，不能在本可视化中改成统一 candidate。

| Query datasets | Outer fold | Frozen candidate | Decision |
|---|---|---|---|
| `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400` | `half_cylinder` | `chirality_all35`, exact same-scale negative retrieval `k=15`, Gaussian `sigma=1` | 每个 dataset/source/block 固定 top 5% |
| `boeing747` | `boeing_747` | `real_neighbor36`, exact same-scale negative retrieval `k=1`, Gaussian `sigma=1` | 每个 dataset/source/block 固定 top 5% |

完整 `candidate_id` 必须同时写入 `per_figure_metrics.csv`、`main_table.md`、scene
metadata、render metadata 和 `visualization_manifest.json`。top-5% decision 在组内
是 transductive 的；图不能解释为每个 primitive 独立使用一个固定绝对阈值。

## 固定输入与认证闭链

### 父 scene

父实验为 `Other_MainExp31FamilyHeldOutVisualization_1.1`，numerical commit
`86be29698eb689c0e269fe987a5b6d5f125a67be`，config SHA-256
`6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670`。
固定 run：

`/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6`

认证下列顶层文件及其闭链：

- `result_manifest.json` file SHA-256 `57f03ba1…d4188e`，content SHA-256
  `8cb1cbfb…0b82cf`；
- `visualization_manifest.json` file SHA-256 `90bc9e1b…d241a2`，content SHA-256
  `79ee91bf…da7474`；
- `RUN_COMPLETE.json` file SHA-256 `63325cf5…28314b`。

随后由父 result 和 visualization manifest 对8个 source ordinal 2 scene NPZ、
scene manifest 和 render metadata 做 path、size、SHA-256 双重绑定。NPZ 打开后，
child scene 除 `prediction` 与 `metadata_json` 外的每个数组必须逐数组完全相同；
child `prediction` 还必须与 exact-joined prediction 数组及 canonical array hash
完全相同。

### NegativeTail folds

父实验为 `Verify_NegativeTailCalibration_1.1`，numerical commit
`e9d4d3f11428bd2e13fc0fabf657be7c7e57db7c`，config SHA-256
`4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b`。

每个 fold 必须认证恰好13个文件：`result_manifest.json`、result 中恰好11个
artifact、`RUN_COMPLETE.json`。11个 artifact 为：

`final_tail_calibration.npz`, `final_tail_calibration_manifest.json`,
`inner_candidate_summary.csv`, `inner_fit_audits.json`,
`inner_group_metrics.csv`, `outer_group_metrics.csv`,
`outer_prediction_manifest.json`, `outer_predictions.npz`,
`outer_reference_access_audit.json`, `outer_summary.json`,
`selected_candidate.json`。

增加、缺少、改名、size 变化或 SHA-256 变化都必须失败。还必须验证：

- result、prediction manifest、selected candidate、final calibration manifest、
  outer reference audit、outer summary 和 completion marker 的 experiment、commit、
  config、outer fold 与 candidate 互相一致；
- final calibration manifest 的 `fit_families` 必须按冻结 family order 恰好等于
  全部5个 physical families 去除 outer family 后的4个，禁止 outer-family 拟合
  泄漏；final calibrator NPZ 本实验不读取或用于作图，其完整文件由 result 和
  calibration manifest 的 size/SHA-256 闭合；
- `selected_candidate.json.inner_evidence` 恰好绑定三份 inner evidence 的
  path/size/hash；`inner_selection_summary` 的 candidate 与 frozen candidate 一致，
  `inner_family_count=4`，且 `group_count` 必须与 frozen physical population 一致：
  `half_cylinder=40`、`boeing_747=56`；
- outer reference 只在 prediction file 和 manifest 完成认证后打开；
- result 中嵌入的 outer summary 与文件去除自哈希字段后的内容一致。

## Prediction archive 与 exact ordered join

在打开任何父 scene NPZ 或 NegativeTail prediction NPZ 前，runner 必须先写入
`input_manifest.json`。随后才允许打开两个 `outer_predictions.npz`。

每个 prediction archive 必须具有固定顺序的18个数组，并逐数组验证 dtype、
shape 和 canonical array SHA-256：`dataset`, `source_ordinal`, `source_index`,
`scale_id`, `center_seed_index`, `scale_block_index`, `assigned_row_index`,
`raw_negative_distance`, `tail_probability`, `tail_anomaly`, `spatial_score`,
`spatial_denominator`, `retrieval_supported`, `calibration_supported`,
`spatial_imputed`, `spatial_unimputable`, `calibration_mode`, `prediction`。

每个 `dataset/source_ordinal/scale_block` 内，`center_seed_index` 与
`assigned_row_index` 都必须分别一一唯一；仅 composite key 唯一不够。source 2
同一 dataset 的 legacy/expanded 两个 block 还必须具有相同 `source_index`。
source 2
的 parent scene 与 prediction 再按以下完整 identity 精确比较：

`dataset, source_ordinal, scale block, center_seed_index, assigned_row_index,
scale_id, scale_block_index`

两侧不允许 duplicate、missing、extra 或 row reorder。通过后直接使用
`spatial_score` 与 `prediction`，不做 projection、插值、重排或阈值重算。

## 指标门禁

每张图使用该 dataset/source/block 的全部 valid rows；64,000 assigned rows 只作
coverage 分母。重算 accuracy、Average Precision、F1、balanced accuracy、Area
Under the Receiver Operating Characteristic Curve、precision、recall 和四类
confusion counts。每个整数指标必须与父 `outer_group_metrics.csv` 完全相同，
每个浮点指标绝对误差不得超过 `1e-12`。legacy 与 expanded 的 valid population
不同，因此不能只凭两栏分数差异推断尺度方法优劣。

## 颜色、marker 与遮挡说明

为保持继承的 FMT 横向三联图布局，本版本不在图内增加 legend。独立下载结果包
仍必须通过 `main_table.md` 和 render metadata 提供下列固定 visual key：

- panel b：red 为 predicted vortex（alpha 0.92），blue 为 predicted non-vortex
  （alpha 0.24）；
- panel c：red circle 为 true positive，purple triangle 为 false positive，orange
  `x` 为 false negative，faint blue circle 为 true negative（alpha 0.035）。

较大、较不透明的错误 marker 可能按设计遮住淡色背景点；这不是 row omission。
“图内无 legend”作为本诊断布局的已知展示限制保留，不能静默改变父 renderer。

## 输出与图形 QA

每张图必须不可覆盖地输出 scene NPZ、scene manifest、SVG、PDF、360 dpi PNG、
panel alignment JSON、render metadata JSON，以及一份 final-PDF 5 pt text audit
JSON。SVG/PDF 的文字保持可编辑，三维 marks 栅格化。8份 PDF 的 `Tf` text run
必须可审计、最小字号不低于5 pt；不通过时不得写 completion marker。

Ibex `deepvortex` 当前没有 PyMuPDF，且本版本不扩大环境或临时安装依赖。因此
rendered-PDF collision audit 与8张原始 PNG 的目视检查是下载后的本地交付门禁，
必须在最终对用户报告结果前完成；其结论和报告哈希随后记录到实验文档。不能把
Ibex 内5 pt text PASS 误写成 collision PASS。

全局输出还包括 `frozen_config.yaml`, `input_manifest.json`,
`per_figure_metrics.csv`, `main_table.md`, `visualization_manifest.json`,
`environment_versions.json`, `result_manifest.json`, `RUN_COMPLETE.json`。
result manifest 记录 completion 之前全部70个 artifact 的 size 和 SHA-256，并在
写 completion 前重新验证全部输出和全部输入哈希。

## Ibex 运行

wrapper：`ibex/other_negative_tail_visualization_1.1.sh`。只允许从 clean、committed
Git revision 运行，提交时必须以 `EXPECTED_NUMERICAL_COMMIT` 明确绑定40字符
commit。wrapper 先运行共享测试和本实验12个零参数定向测试，再执行 runner 和
70-artifact post-validation。权威 run 目录：

`/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeTailVisualization_1.1/runs/slurm_51062980_cca0f589fad1`

Ibex job `51062980` 于 2026-08-31 00:14:53--00:18:16 +03:00 在
`cn604-10` 完成，exit `0:0`，elapsed `00:03:23`，16 CPU cores、64 GB，
MaxRSS `2385332K`。数值 commit 为
`cca0f589fad134c976fd94671eeda84df8845e7f`；该 committed revision 的203项共享
测试和本实验12项定向测试全部通过。作业认证53个唯一输入文件，
生成8张图、406,177个 valid query rows 和70个 result artifacts。

## 完成结果

| Flow | Scale block | Valid / 64,000 | Coverage | AP | F1 | BA | Precision | Recall | TP / FP / TN / FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Re160 | legacy | 60,560 | 94.63% | 0.7529 | 0.6858 | 0.8089 | 0.7510 | 0.6310 | 2,274 / 754 / 56,202 / 1,330 |
| Re160 | expanded | 43,347 | 67.73% | 0.6636 | 0.5720 | 0.7313 | 0.7117 | 0.4782 | 1,543 / 625 / 39,495 / 1,684 |
| Re640 | legacy | 60,555 | 94.62% | 0.5911 | 0.5342 | 0.7417 | 0.5651 | 0.5065 | 1,711 / 1,317 / 55,860 / 1,667 |
| Re640 | expanded | 42,463 | 66.35% | 0.4188 | 0.4342 | 0.6838 | 0.4816 | 0.3953 | 1,023 / 1,101 / 38,774 / 1,565 |
| Re6400 | legacy | 62,313 | 97.36% | 0.6084 | 0.5466 | 0.7577 | 0.5546 | 0.5388 | 1,728 / 1,388 / 57,718 / 1,479 |
| Re6400 | expanded | 57,906 | 90.48% | 0.5830 | 0.5088 | 0.7336 | 0.5266 | 0.4923 | 1,525 / 1,371 / 53,437 / 1,573 |
| Boeing 747 | legacy | 61,432 | 95.99% | 0.8198 | 0.7365 | 0.8660 | 0.7269 | 0.7463 | 2,233 / 839 / 57,601 / 759 |
| Boeing 747 | expanded | 17,601 | 27.50% | 0.6546 | 0.5608 | 0.8540 | 0.4529 | 0.7362 | 399 / 482 / 16,577 / 143 |

固定 source 2 上，Boeing legacy 的 F1 最高（0.7365），是唯一超过0.70的图；
Re160 legacy 为0.6858。其余六图为0.4342--0.5720。expanded block
在四个flow上的F1都低于legacy，但两block的valid population不同，特别是
Boeing expanded coverage 只有27.50%，因此不把这一固定source图解释为尺度
方法的独立因果比较。

## 产物与交付 QA

- result manifest SHA-256：`56284ba4064d88bd0c0375abd1a25d71bf17e3b8bf909711fefd72dcc29c2b83`；
  `RUN_COMPLETE.json` SHA-256：`96e1cde702c0da825efbb8ebb1951c9b6a7d385eca0d1a452eeab4af72f02fb6`。
- visualization/input manifest SHA-256：`e9e4157e2343ae5f476a1ef0f2feb2974fa09531bfb36b4d3a8a7c03466e0c37`
  / `28d6886e9dfb84209fc68a1c97c66b762c6286c82f1deb5f4a0aa4d1b607d59c`。
- stdout/stderr SHA-256：`454b28492552ed9301ec469088c1ef39b36dfbae78e5eebe457d7cd978a7c914`
  / `d3682305e805236c7dfb81f57482040808fcc78b0205f756d47ab4abcf133634`。
- 下载后重放70/70 artifact hashes；8/8 panel alignment 严格 PASS；8/8 PDF
  最小文字为7 pt；8份collision report均0 FAIL。共96个WARN全部是`x`轴标签或
  数值轴刻度与3D栅格/填充边缘接触，原始像素目视复核未发现标题、panel label
  或数据标记裁切。8/8 PNG解码和目视PASS；8/8 SVG含可编辑`<text>`。
- 本地 QA summary SHA-256：`d029b1d996e838ef1fcf07ab1294356d724af29ce3007d75efacfc4b06c8a1c6`；
  content SHA-256：`bef4351509eddf163422d3dcd946511673037b06824635436c5cde470d1d1454`。
- 本地下载：`outputs/Other_NegativeTailVisualization_1.1_job51062980_download`。

这些完成结果只支持固定source的family-held-out exposed-development空间解释；
不改变`Verify_NegativeTailCalibration_1.1`完整五折F1=0.540472、未通过停止门槛的
结论，也不是formal confirmation。
