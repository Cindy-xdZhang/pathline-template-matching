# `Other_SourceCenteredPairedScaleTemplateVisualization_1.1`

状态：**`COMPLETED_LOCAL_REPORTING_QA_PASS`**。本版本是
`Verify_SourceCenteredPairedScaleTemplate_1.1` 的固定下游报告，不训练、不重新拟合、
不选择候选、不改变阈值，也不把 legacy 与 expanded 两个尺度块伪装成两个分类器。
上游完整五折认证和机器渲染job `51162501`均已完成；真实四图的指标、位置检查、SVG/PDF文字
检查、碰撞审计和逐图人工复核均已通过。结构化本地证据为
`docs/evidence/Other_SourceCenteredPairedScaleTemplateVisualization_1.1_local_summary.json`。

冻结配置为
`config/Other_SourceCenteredPairedScaleTemplateVisualization_1.1.yaml`，SHA-256 为
`c9c9a14b02fc3f47a4ee934ccd1091a7c7accefdbd28f569100605bf8230ca4e`。方法父配置
SHA-256 为 `15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc`；
父空间 scene 固定来自 `Other_MainExp31FamilyHeldOutVisualization_1.1` job `51029080`，
numerical commit 为 `86be29698eb689c0e269fe987a5b6d5f125a67be`，结果 manifest
SHA-256 为 `57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e`。
报告实现的exact commit为`2468222535f4c87cbd7046a88b3cd4b6dc892356`；受信任的上游数值
commit为`a85c007ef961ce53bb40946ca3f38f033bf7a646`。

## 上游五折认证结论与修订

`Verify_SourceCenteredPairedScaleTemplate_1.1` 的完整五折认证job为`51160422`。其冻结config、
输入manifest与32-sidecar population manifest的SHA-256依次为：

```text
15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc
5f7e567a2f989d18b51389814938a5d18025c4ed5247730d07df30b13458fec9
50d9d53f7dc9255d5153f0101c922975e006303b550bfb43317074080a0a97e2
```

公开release的`AGGREGATE_COMPLETE.json`、`aggregate_manifest.json`、
`aggregate_summary.json`与`outer_family_summary.csv`文件SHA-256依次为：

```text
f6599295df79f764a9e1c45a08ee62eef747f09676fab1b4c16c378f0568fbe8
e97088f7fbdb8f1edc4d27ed113341f95b1b90f948416b2fae80977a0613ba43
bfb6db3ec0db0b200ae14b37e3607a012a61d1bc96cb892d0500c461c52cc2af
39bdd096ffd8ca398f3d411280036680a097b8797659bfdb0bb372e550084934
```

completion与summary的canonical content SHA-256分别为
`640bf2f6c2f8eb95699dae186d889e4810eeab18f000a172c7ae7260edd4bfe4`和
`699b7cb20bf96f2560896cfe025bf6830d7d3960960795e6d9730d6068bf1852`；job stdout/stderr
SHA-256分别为`53afefbdb54dc3f301e30a755be1d95eec8c07b6ea1884bcfe7e8e52ca198879`和
`abb3982bb0858ad3562355934153ab76bfb77f15527655349fdca97228d6e88f`。

primary valid-row projection的五族等权macro Accuracy、Average Precision、F1、Balanced
Accuracy、Area Under the Receiver Operating Characteristic Curve、precision和recall分别为
`0.970084/0.750806/0.679390/0.858326/0.981895/0.648096/0.734997`；combined-valid
unique-center coverage为`0.956523`。half-cylinder、delta-wing、F22、channel与Boeing 747的
family F1分别为`0.636556/0.816445/0.579344/0.572869/0.791734`。相对Early parent的
paired dataset-source bootstrap F1差为`+0.018611`，5000次重采样的95%置信区间为
`[0.012595,0.024880]`。

先前首折结论“half-cylinder F1=`0.636556`，乐观五折成功门仍可能满足” → 当前五折结论
“macro F1=`0.679390<0.70`，且只有delta-wing与Boeing 747共2/5 family达到F1≥0.65，未满足
至少4/5，故`stop_version=true`” → 结论变化是因为其余四折和独立完整五折认证现已完成 →
先前结论只用于判断是否释放其余四折，范围限于首折，并未宣称完整方法成功。其余冻结门均通过，
且bootstrap改善区间下界高于0，但这不能替代两个失败的硬门。

direct source-centered dx-rank-mean与min-dx诊断的五族macro F1分别为`0.858241`和
`0.841770`。二者不使用负模板库，只检验直接source-centered局部运动学的排序信息；它们既不参与
主候选成功判定，也绝不能写成模板匹配方法达到0.70–0.80。

## 实际四图结果

Ibex job `51162501`从reporting commit
`2468222535f4c87cbd7046a88b3cd4b6dc892356`在`cn514-15-r`完成，exit `0:0`，运行
`00:06:51`，使用32 CPU、128 GiB、Rome、无GPU；486项测试通过，2项跳过。机器产物为35个文件，
本地QA后完整bundle为61个文件。`RUN_COMPLETE.json`、`result_manifest.json`、
`visualization_manifest.json`和`per_figure_metrics.csv`的文件SHA-256依次为：

```text
f93e74dc0adb9daa5023c44969cc5d671aba87c6f25adf413c5f403efde92901
2a20dbb769fe71c7113649e329fcfa96223475e344567e324f12d84e03d0754a
f07a64ebd7269456838acf40508f4dcef6c9e8f5361516b4b9ff6ac7389798b6
fdbf7b0706dd2251e91421bfeb6eb52e0c057c9bc4eec53018b55bfd7dc53097
```

以下是图中`combined_valid_unique_centers`总体；每个中心只计一次，正类是whole-volume IVD p95
reference。Accuracy受约95%负类支配，不能代替F1、Average Precision、Balanced Accuracy、
precision或recall。

| flow | centers / coverage | TP / FP / TN / FN | F1 | Average Precision | Balanced Accuracy | precision / recall |
|---|---:|---:|---:|---:|---:|---:|
| Cylinder3D Re160 | 60,560 / 94.63% | 2,105 / 455 / 56,501 / 1,499 | 0.6830 | 0.7763 | 0.7880 | 0.8223 / 0.5841 |
| Cylinder3D Re640 | 60,555 / 94.62% | 1,611 / 949 / 56,228 / 1,767 | 0.5426 | 0.6227 | 0.7302 | 0.6293 / 0.4769 |
| Cylinder3D Re6400 | 62,315 / 97.37% | 1,922 / 638 / 58,468 / 1,287 | 0.6663 | 0.7835 | 0.7941 | 0.7508 / 0.5989 |
| Boeing 747 | 61,433 / 95.99% | 2,583 / 617 / 57,823 / 410 | 0.8342 | 0.9100 | 0.9262 | 0.8072 / 0.8630 |

前三个Cylinder流场使用half-cylinder outer fold经inner-family选择的
`chirality35+source4, k=15, sigma=0.5, legacy weight=0.75, top 4%`；Boeing使用其outer fold选择的
`real-neighbor36+source4, k=31, sigma=0.5, legacy weight=1, top 5%`。这是预注册nested
family-specific selection，不是把一个相同超参数candidate横跨四个流场。

Boeing fixed source最好，F1=`0.8342`且precision/recall同时较高；Re640最弱，F1=`0.5426`，主要
问题是recall=`0.4769`且FP明显多于Re160/Re6400。Reynolds number结果不单调：Re160、Re640、
Re6400的F1依次为`0.6830/0.5426/0.6663`。这些单source图不能推翻完整五族primary macro
F1=`0.679390<0.70`的停止结论。

Panel B中红色表示预测涡流、蓝色表示预测非涡流。Panel C中红色圆点为TP、紫色三角为FP、
橙色叉号为FN、低透明度蓝色圆点为TN。Panel A中蓝色实线与紫色虚线只分别标出legacy和expanded
固定前120条pathline背景，不是两个独立分类器。

## 最终本地QA

本地checkout临时切到机器记录的exact reporting commit后执行正式auditor。4/4严格面板位置、4/4
PDF文字、4/4 SVG可编辑文字和4/4逐图最终尺寸检查均通过；PDF最小文字为7 pt，碰撞hard failure
为0。collision auditor共有51个warning，逐张叠加层确认它们只来自三维刻度文字接触栅格绘图区或
filled axes边缘，没有标题、图例、面板或数据被遮挡。`delivery_qa_summary.json`文件SHA-256为
`ebb6b5b8545b85debd7a2a1928c7b71a1de522df0a0e998059781b3652b5aa84`，canonical content
SHA-256为`9c575ff707f1c31749009f86b40b42f6a65bbd55fd16aff4432305995497ac5a`；人工复核JSON
SHA-256为`2c703814e28e4a915f188bb69291a33dabe1e55dfd5cdd209e33a3417ad12877`。

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

job `51160422`现已发布一个`complete_five_fold_aggregate`，同时授权half-cylinder与Boeing fold；
报告器绑定的aggregate completion SHA-256为
`f6599295df79f764a9e1c45a08ee62eef747f09676fab1b4c16c378f0568fbe8`，half-cylinder与Boeing
fold completion SHA-256分别为
`0cda692c60229381fad3a0e4eff278798844b0eaca9d21e52e7b4af2408cdbdd`和
`7f3a232b77d01312c6c886d051affadf5b956b84e66665f761833e27871f1a65`。生产job `51162501`已从
reporting commit `2468222535f4c87cbd7046a88b3cd4b6dc892356`完成，固定output为
`/ibex/user/zhanx0o/pathline-template-matching/Other_SourceCenteredPairedScaleTemplateVisualization_1.1/runs/report_2468222535f4_20260901_01`。
机器阶段单独仍不是可发布结果；其后完成的本地QA才构成当前交付状态。

机器阶段按合同结束在 `complete_pending_local_rendered_qa`，不是可交付状态。job `51162501`完成后，
完整目录已下载，并按21×5 inch最终物理尺寸逐张检查四张PNG，同时建立独立visual-review JSON。
该文件固定四个
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
`delivery_qa_summary.json`，其中 `delivery_status=PASS`。本次已经满足全部条件；原始
`RUN_COMPLETE.json`仍保留机器阶段的pending状态，两者记录的是先后阶段，不互相覆盖。

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
这些实现与测试固定在reporting commit
`2468222535f4c87cbd7046a88b3cd4b6dc892356`。
其余输入使用 synthetic scene
和故意无效的 opaque prediction bytes，只验证合同；它们不是实际流场结果。

## 结论边界

本报告只能作 `family-held-out exposed-development fixed-source visualization`。四个 flow 与固定
source 2 已暴露；source-centered mean 使用目标 flow 自身无标签速度，因此完整分类器属于
transductive method（推理时使用目标集合无标签统计的方法）。四张图没有置信区间，不能替代完整
five-family 统计，也不能称 sealed confirmation、无偏模型选择、聚类、独立单 primitive 分类器，
或 legacy/expanded 两个方法的因果比较。上游完整五折primary F1=`0.679390`已经按冻结门停止；
即使后续固定source图的某个流场表现较好，也不能推翻该五折失败结论。
