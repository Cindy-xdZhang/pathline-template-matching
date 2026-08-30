# `Other_NegativeTailVisualization_1.1`

## 状态与用途

状态：`FROZEN_PRE_RUN_NOT_RUN`。

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
70-artifact post-validation。预期 run 目录：

`/ibex/user/zhanx0o/pathline-template-matching/Other_NegativeTailVisualization_1.1/runs/slurm_JOBID_COMMIT12`

当前没有 job ID、结果指标或图形结论；运行完成前不得把本文件状态改为 completed。
