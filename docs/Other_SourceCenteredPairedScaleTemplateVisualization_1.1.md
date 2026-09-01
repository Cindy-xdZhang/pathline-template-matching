# `Other_SourceCenteredPairedScaleTemplateVisualization_1.1`

状态：**`FROZEN_IMPLEMENTED_LOCAL_486_TESTS_PASS_NO_SOURCE_CENTERED_RESULT_READ`**。本版本是
`Verify_SourceCenteredPairedScaleTemplate_1.1` 的固定下游报告，不训练、不重新拟合、
不选择候选、不改变阈值，也不把 legacy 与 expanded 两个尺度块伪装成两个分类器。
截至实现与测试完成时，没有读取该 Verify 版本的新 feature、prediction、metric 或父 scene
数组，因此本文没有性能数值或图面结论。

冻结配置为
`config/Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml`，SHA-256 为
`c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e`。方法父配置
SHA-256 为 `15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc`；
父空间 scene 固定来自 `Other_MainExp31FamilyHeldOutVisualization_1.1` job `51029080`，
numerical commit 为 `86be29698eb689c0e269fe987a5b6d5f125a67be`，结果 manifest
SHA-256 为 `57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e`。

## 固定图与统计总体

报告固定 source ordinal 为 2，并按以下顺序生成四张图：`cylinder3d`（Re160）、
`halfcylinderRe640`（Re640）、`halfcylinderRe6400`（Re6400）和 `boeing747`。
每个 flow 只有一张三联图：

1. Panel A 保留父 scene 的 whole-volume IVD-p95 网格，并固定绘制
   `legacy_2_1` 数组前120条蓝色实线路径线与 `expanded_3_1` 数组前120条紫色虚线路径线；
2. Panel B 对两个 block 至少一个积分有效的唯一中心，只绘制一个
   `paired_prediction`；
3. Panel C 按与 Panel B 完全相同的升序 `center_seed_index`，绘制 true positive（TP）、
   false positive（FP）、false negative（FN）和 true negative（TN）。

三个 panel 都实际绘制同一个 IVD 背景，并保持相机与物理边界相同；Panel B/C 的中心顺序也
完全相同。两种路径线的图例只放在三轴以外的 figure-level 顶部安全区，不能遮挡三维刻度或
数据。Panel A 的240条路径线只说明
两个尺度块的几何背景，不是 Panel B 的抽样总体，也不代表两个分类器。

报告同时给出两套不可混称的指标：

- 图中指标使用 `combined_valid_unique_centers`，一个中心最多计一次；
- 主要方法指标使用 `all_parent_valid_rows`，按 legacy 父 scene 顺序再按 expanded 父 scene
  顺序投影 `valid_paired_prediction`。同一中心若两个 block 都有效，会在该主要总体中出现两次。
  这套指标只写表，不进入 Panel B 或 Panel C。

每个 valid row 的 `valid_paired_prediction` 和 `valid_paired_score` 还必须分别逐位等于相同中心的
唯一 `paired_prediction` 与 `paired_score`，不能只靠聚合指标偶然一致。两套指标都必须从认证
prediction 与父 reference 重新计算，并在 `1e-12` 绝对容差内逐字段复现 producer 的固定 metric
行；producer 行的 family、dataset、source、arm、population、成功资格与完整 candidate 身份也
必须一致。Accuracy（准确率）不能替代 Average Precision（平均精确率）、
F1、Balanced Accuracy（平衡准确率）、precision 或 recall；coverage 也必须与分类指标同时报告。

## 不可放宽的连接与来源检查

报告器在打开任何 NPZ 成员前完成以下事务：

- 认证 aggregate completion、aggregate manifest、公开 report/table、fold completion、fold result、
  selected candidate、prediction manifest 和全部 fold artifact 的文件大小及 SHA-256；
- aggregate completion、manifest 与 report 必须逐字节绑定相同的 `aggregator_git_commit`、
  `fold_git_commit` 和完整 `source_centered_evidence`；aggregator/fold/evidence commit 均固定为已信任
  数值 revision `a85c007ef961ce53bb40946ca3f38f033bf7a646`；
- 认证八个父 scene 的固定文件集合和已知 job/commit/config/result 身份；
- 记录冻结配置、报告器及其直接数值/渲染依赖的逐文件 SHA-256；
- 写出 `input_manifest.json` 与 `figure_contract.json`。

随后执行 fail-closed exact join。唯一中心必须恰为升序 `0..63999`；dataset、source ordinal、
source index、block index、center、assigned row 和 scale ID 必须逐项一致；重复、缺失、额外、
重排、跨 source 或跨 family 行全部失败。两个父 block 对同一中心的坐标与 IVD-p95 reference
必须逐位相同，`legacy_valid` 与 `expanded_valid` 也必须精确等于父 scene 中的中心成员关系。

每个 combined scene 使用固定16-array、`allow_pickle=False` 合同，并单独写 scene manifest。
每张图导出 360 dpi PNG、保留可编辑文字的 PDF/SVG、panel 位置 JSON 和 render metadata。
报告目录已存在时直接失败，任何旧输出都不覆盖。

## 运行接口

机器渲染必须从 push 后的 clean exact Git commit 在 Ibex 执行。生产入口是
`ibex/other_source_centered_paired_scale_template_visualization_1.1.sh`，固定1个Rome节点、32 CPU、
128 GiB、12小时、无GPU，并通过 `scontrol` 复核实际分配。提交前必须设置以下8个运行身份：

```text
EXPECTED_GIT_COMMIT
SOURCE_CENTERED_VIZ_RELEASE_ROOT
SOURCE_CENTERED_VIZ_RELEASE_COMPLETE_SHA256
SOURCE_CENTERED_VIZ_HALF_FOLD_ROOT
SOURCE_CENTERED_VIZ_HALF_RUN_COMPLETE_SHA256
SOURCE_CENTERED_VIZ_BOEING_FOLD_ROOT
SOURCE_CENTERED_VIZ_BOEING_RUN_COMPLETE_SHA256
SOURCE_CENTERED_VIZ_OUTPUT_ROOT
```

wrapper 只接受一个 `complete_five_fold_aggregate` release；所有输入 root 必须是绝对路径，三个
completion SHA-256、固定父结果 SHA-256 和冻结报告配置 SHA-256 都在测试或 NPZ 访问前认证。
它运行定向测试、wrapper 合同测试和完整测试后，才调用以下精确报告接口：

```text
python scripts/render_source_centered_paired_scale_template_visualizations.py \
  --parent-root <job-51029080-run-root> \
  --release-root <authenticated-source-centered-release-root> \
  --half-fold-root <authenticated-half-cylinder-fold-root> \
  --boeing-fold-root <authenticated-boeing-fold-root> \
  --output-root <new-immutable-output-root> \
  --expected-reporting-commit <exact-40-hex-commit>
```

当前 SourceCentered aggregator 的 `single_fold_authentication` 只允许首折
`half_cylinder`。因此，现有 producer 接口下，四流场完整报告必须由一个
`complete_five_fold_aggregate` 同时授权 half-cylinder 与 Boeing fold。若 Verify 在首折停止而没有
完整五折 release，本版本不能绕过认证补 Boeing；必须先建立新的、结果可见前冻结的 Boeing
诊断版本及公开 release。

机器阶段固定结束在 `complete_pending_local_rendered_qa`，不是可交付状态。下载完整目录后，先按
21×5 inch 最终物理尺寸逐张检查四张 PNG，并准备独立 visual-review JSON。该文件必须固定四个
dataset 顺序、逐图 PNG SHA-256、全部五项布尔检查、reviewer、UTC 时间与总体 `PASS`。如果碰撞
审计有 warning，还必须逐图写
`collision_warning_review=accepted_after_final_size_review` 和非空说明；没有 warning 时写
`not_applicable_no_warnings`。

本地后置检查命令为：

```text
python scripts/audit_source_centered_paired_scale_template_visualizations.py \
  --output-root <downloaded-machine-render-root> \
  --visual-review-json <independent-visual-review.json> \
  --nature-figure-tool-root <nature-figure-skill-scripts-directory>
```

后置脚本不打开 prediction、label、feature、父 scene 或 combined scene 的 NPZ 成员。它重新认证
机器产物，并要求本地 checkout 与机器记录的 reporting commit 完全相同且 clean；本地 auditor
本身也必须是机器 reporting dependency manifest 中的同一 SHA-256。源码预检当前固定3个warning：
无TIFF、PNG为360 dpi而非默认600 dpi、21 inch宽度不是常见期刊栏宽；它们分别由冻结的
SVG/PDF主输出、360 dpi预览和三幅3D报告画布合同处置，出现任何新增 warning 都失败。随后运行
1.5 pt 严格三面板位置检查、SVG真实`<text>`元素检查、PDF最小5 pt文字检查和 PDF
rendered-collision 检查，并绑定人工逐图复核。只有4/4位置检查、4/4 SVG/PDF文字检查、0个碰撞 hard
failure 和4/4人工复核全部通过，才会不可覆盖地写出自哈希
`delivery_qa_summary.json`，其中 `delivery_status=PASS`。原始 `RUN_COMPLETE.json` 保留机器阶段的
pending 状态，两者记录的是先后阶段，不互相覆盖。

## 代码与测试

主要实现路径为：

- `src/pathline_template_matching/source_centered_visualization.py`：两个父 block 的精确连接、
  valid-row 投影、combined scene 与单分类器三联图；
- `scripts/render_source_centered_paired_scale_template_visualizations.py`：opaque 文件认证、输入
  manifest、指标复算、四图和不可变机器结果；
- `scripts/audit_source_centered_paired_scale_template_visualizations.py`：本地最终交付检查；
- `ibex/other_source_centered_paired_scale_template_visualization_1.1.sh`：固定Rome CPU生产入口；
- `tests/test_source_centered_visualization.py`：配置、连接、重复/缺失/重排、块间坐标漂移、
  文件认证先于 NPZ 打开、人工检查绑定和后置检查范围的回归。
- `tests/test_source_centered_visualization_ibex.py`：Slurm资源、路径/SHA、clean commit、不可覆盖、
  complete-five-only与35文件机器事务的wrapper合同。

实现阶段核心定向测试为15/15 PASS，wrapper合同为8/8 PASS；其中包含一次完整 synthetic
PNG/PDF/SVG/位置证据渲染，并实际调用正式collision auditor得到hard fail=0。接入后的本地标准库
测试为486/486 PASS，相关 Python 文件均通过 `py_compile`，新wrapper通过Git Bash `bash -n`。
其余输入使用 synthetic scene
和故意无效的 opaque prediction bytes，只验证合同；它们不是实际流场结果。

## 结论边界

本报告只能作 `family-held-out exposed-development fixed-source visualization`。四个 flow 与固定
source 2 已暴露；source-centered mean 使用目标 flow 自身无标签速度，因此完整分类器属于
transductive method（推理时使用目标集合无标签统计的方法）。四张图没有置信区间，不能替代完整
five-family 统计，也不能称 sealed confirmation、无偏模型选择、聚类、独立单 primitive 分类器，
或 legacy/expanded 两个方法的因果比较。
