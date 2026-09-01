# Other_ClassConditionalTemplateScoreVisualization_1.1：双认证单折分类三联图

状态：**`COMPLETED_LOCAL_REPORTING_QA_PASS`**。Ibex job `51156521`已从exact reporting commit
`5d3d49eae02b59aae11d399755cee33f3e7884e3`完成完整wrapper认证；下载后的8张图又通过本地PDF/SVG、
alignment、碰撞overlay和目视检查。本实验只生成用户要求的Cylinder3D Re160、Re640、Re6400和Boeing 747
当前class-conditional template-score分类效果；它不恢复已停止的Verify五折，也不把两个不同实验的单折证据
伪装成完整五折或formal confirmation。

冻结production config为
`config/Other_ClassConditionalTemplateScoreVisualization_1.1.yaml`，SHA-256为
`c69d4a59b4906a32f6e14e100c2fe553cc110c6c08fdb34842f20e198a504a60`。该哈希已由本地parser测试和Ibex
逐字段只读核查共同验证。

## 运行历史

首次reporting commit `e89fedcfa18757f9b5cfdb5311214a0720b17830`的Ibex job `51155277`在
`cn514-15-r`完成456/456测试，并在打开NPZ member前认证两个release、parent scenes与report config。
随后读取已密封`outer_group_metrics.csv`时，reporter错误要求`inner_family=outer`；producer与认证aggregator
的真实固定合同是`inner_family=outer_evaluation_only`，因此job以
`half_cylinder: metric fold identity changed`停止。该错误发生在任何scene或figure写入之前；partial output只含
`frozen_config.yaml`、`input_manifest.json`和`figure_contract.json`，没有逐图指标或性能证据。

修复只新增并复用固定outer-evaluation identity，同时把synthetic 15-file fold fixture改成真实metric CSV，要求
`read_outer_group_metrics`与metric重算完成端到端比较。production config SHA、两个source numerical commit、
prediction、候选、阈值、support、block、source ordinal和parent scenes均未改变。失败job及partial output永久保留；
下一次运行必须使用新output目录。

第二次reporting commit `82c2fd117dccfa62f6225ccdf5c4acf733afb72d`的Ibex job `51155495`完成
456/456测试、两个release认证、8组prediction/scene exact join、metric重算比较与8张图渲染；reporter本体exit 0，
并写出预期63文件。随后wrapper第一次重算`visualization_manifest.json`自哈希时失败：部分subset F1为空，内存对象
包含`NaN`；`_atomic_json`落盘时按固定JSON合同把它规范化为`null`，但manifest自哈希错误地在规范化前计算。
因此落盘内容与声明哈希不一致，Slurm终态为`FAILED 1:0`。即使图件和`RUN_COMPLETE.json`存在，这一attempt也不
可发布，且未执行本地QA。下一revision只能在计算自哈希前应用同一JSON-safe规范化，并用真实`NaN→null`持久化
回归锁定；不得改变config、source prediction、逐图metric、candidate、threshold、support、scene或renderer。

该最小修复已在commit `0cf30b605e63e0b2b6866e40eb48bff114583a83`实现：先把完整visualization
manifest递归转换成实际落盘的JSON-safe对象，再计算自哈希；新增回归同时验证`NaN→null`后磁盘自哈希重算相等，
以及metric CSV仍写空字段。16项reporter定向测试、wrapper定向测试、Python编译和456/456全套测试均通过；
production config SHA仍未改变。该测试结果只支持修复revision可以提交新output attempt，不认证任何旧图件。

第三次Ibex job `51156521`从exact reporting commit
`5d3d49eae02b59aae11d399755cee33f3e7884e3`提交，于2026-09-01 19:12:33–19:18:27 +03在
`cn514-15-r`运行，Slurm终态`COMPLETED 0:0`。它完成456/456测试、两个source release与parent scene认证、
8组prediction/scene exact join、sealed metric逐字段重算、8图渲染和61-artifact/63-file wrapper复核。实际资源为
32 CPU、128 GB、Rome、无GPU；batch MaxRSS `1,646,728K`，TotalCPU `17:45.206`。权威Ibex输出为
`/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreVisualization_1.1/runs/report_5d3d49eae02b_attempt3`。
本地不可变下载目录为
`outputs/Other_ClassConditionalTemplateScoreVisualization_1.1_job51156521_download`；所有后处理QA只写入同级
`outputs/Other_ClassConditionalTemplateScoreVisualization_1.1_job51156521_qa_local`，没有修改下载集合。

## 冻结输入

### Verify half-cylinder单折release

- experiment：`Verify_ClassConditionalTemplateScore_1.1`
- numerical commit：`58b0bc0b0c7385f1b356eb343a150fcd50dad94f`
- config：`/home/zhanx0o/pathline-template-matching-class-conditional-score/config/Verify_ClassConditionalTemplateScore_1.1.yaml`
- config SHA-256：`814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea`
- release root：`/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/aggregate/slurm_51146768_58b0bc0b0c73`
- `AGGREGATE_COMPLETE.json` SHA-256：`f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844`
- fold root：`/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/runs/slurm_51146327_0_58b0bc0b0c73_outer_half_cylinder`

该release只提供`cylinder3d`、`halfcylinderRe640`和`halfcylinderRe6400` prediction。它已经认证
`stop_version=true`，因此本报告只能读取已封存证据，不能在report-time触发public fresh replay。

### Boeing独立Other单折release

- experiment：`Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1`
- numerical/authentication commit：`6322d16cebe5995c8bcec2b8743e9ce0de9d8304`
- config：`/home/zhanx0o/pathline-template-matching-class-conditional-boeing/config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml`
- config SHA-256：`6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856`
- release root：`/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5`
- `DIAGNOSTIC_COMPLETE.json` SHA-256：`a9bb930c540c366dd9fd9fd040bdca306cbb7a0a2fcd829fe5f307a8e85ad12c`
- `boeing_diagnostic_report.json` SHA-256：`dd0a001571f27e9572258f920a7f0b8065121f84274678214c0d76cb54588f55`
- `diagnostic_manifest.json` SHA-256：`9f764e5f447704f2370f39c6530d10fd1bf98bf0d254a2559fe410fb7c4f1998`
- `boeing_outer_summary.csv` SHA-256：`d4c1083bc11eed340dbbbe62d8ff5b5f9ae9bcaf341ba7ca9a7620b3d20493e8`
- fold root：`/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/runs/slurm_51154451_0_6322d16cebe5_outer_boeing_747`

该release只提供`boeing747` prediction。它没有success、stop或five-family macro语义，不是formal
confirmation，也不得与half-cylinder fold做macro平均。

### Parent scenes

固定复用`Other_MainExp31FamilyHeldOutVisualization_1.1`的source ordinal 2场景：

- root：`/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6`
- numerical commit：`86be29698eb689c0e269fe987a5b6d5f125a67be`
- config SHA-256：`6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670`
- `result_manifest.json` SHA-256：`57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e`
- `RUN_COMPLETE.json` SHA-256：`63325cf51bd9a4322ef5ba6385d00aa063b30096134679d153aa6733ac28314b`

parent scene只提供已固定的几何、相机、pathlines与IVD-p95 reference。本报告只能改变prediction和分析
metadata；其余scene arrays必须逐数组保持相同。

## 固定图件与认证顺序

数据集顺序固定为`cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400`、`boeing747`；显示名分别为
Cylinder3D Re160、Cylinder3D Re640、Cylinder3D Re6400、Boeing 747。每项固定source ordinal `2`，并各画
`legacy_2_1`（scale 0–999）和`expanded_3_1`（1000–1999），总计8张。

每张图固定三panel：

1. `IVD p95 + center pathlines`；
2. `FMT class-conditional template-score classification`；
3. `TP / FP / FN / TN against IVD p95`。

reporter必须先认证两个release completion、release manifest、各自15-file fold、13个result artifacts和
prediction NPZ整文件身份，同时认证parent result/completion与8个scene文件组。随后写入
`input_manifest.json`，再允许打开19个prediction array members或metric CSV。每组prediction必须通过
dataset/source/source-index/center/assigned-row/scale/block的exact identity join；重新计算的完整metric row
必须与sealed outer metric逐字段相同。

## 输出与质量边界

每张图导出scene NPZ、scene manifest、PNG、PDF、SVG、alignment audit和render metadata共7文件；8张图共
56文件。全局文件固定为`frozen_config.yaml`、`input_manifest.json`、`figure_contract.json`、
`per_figure_metrics.csv`、`visualization_manifest.json`、`result_manifest.json`、`RUN_COMPLETE.json`共7文件。
原始发布集合因此必须恰为63文件，且输出目录在运行前必须不存在。

主运行使用`ibex/other_class_conditional_template_score_visualization_1.1.sh`，必须从push后的clean exact
reporting commit提交到Ibex。wrapper要求CPU、32 cores、128 GiB、Rome、12小时、无GPU；`cpu`或`batch`
partition均需通过`scontrol`认证。production report config必须以绝对路径和完整SHA-256作为独立、不可变
输入；reporting commit也必须完整冻结。Ibex提交时固定使用
`/home/zhanx0o/pathline-template-matching-class-conditional-boeing/config/Other_ClassConditionalTemplateScoreVisualization_1.1.yaml`
及上述config SHA。本文不虚构尚未产生的reporting commit、job ID或output path；这些身份在实际提交后写入
`docs/ibex_run_registry.md`并回填本文。

Slurm `COMPLETED`只证明63文件transaction及自动门禁完成。下载后又逐张完成8/8 PNG目视检查、PDF/SVG
文字可编辑性、panel alignment、裁切与碰撞检查，并确认颜色和TP/FP/FN/TN说明准确；详见下文QA。任何性能
描述仍必须注明两个single-fold evidence source、不是完整五折、不是formal confirmation。

## 固定分类器与图例语义

8行均使用同一候选
`representation=chirality_all35_plus_seed4|k=5|sigma=2.0|fixed_top_fraction=0.05`：每个
dataset×source×block内按连续`spatial_score`降序排序，相同分数按center index升序，固定预测前5%为涡旋。
兼容字段`tail_anomaly`承载原始class-conditional template score；`tail_probability=1-raw score`，不是后验概率。
两block分别判断，不跨block投票，也没有置信区间。

三联图说明固定为：

1. panel a：橙色为IVD p95等值面；240条center pathline按相对积分时间使用viridis着色，紫色端为起始seed；
2. panel b：蓝色为预测非涡旋，红色为预测涡旋。这是二分类结果，不是聚类结果；
3. panel c：TN为浅蓝圆点、TP为红色圆点、FP为紫色三角、FN为橙色叉号。

图内没有重复图例，因此发布或引用图件时必须保留上述caption。

## Source ordinal 2逐图结果

`Coverage=valid/64,000`。各行真实正类只占有效样本的3.08%–7.44%，Accuracy主要受TN主导。F1 score是
Precision与Recall的调和平均；Balanced Accuracy（BA）是正负类Recall的均值；Average Precision（AP）是
Precision–Recall曲线的加权面积；Area Under the Receiver Operating Characteristic Curve（AUROC）是受试者
工作特征曲线下面积。解释时优先看F1、BA、Precision和Recall。

| Flow | Block | Valid / 64,000 | Coverage | Accuracy | AP | F1 | BA | AUROC | Precision | Recall | TP | FP | TN | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cylinder3D Re160 | legacy | 60,560 | 94.63% | 0.9486 | 0.5354 | **0.5311** | 0.7332 | 0.9704 | 0.5816 | 0.4886 | 1,761 | 1,267 | 55,689 | 1,843 |
| Cylinder3D Re160 | expanded | 43,347 | 67.73% | 0.9264 | 0.4800 | 0.4089 | 0.6576 | 0.9533 | 0.5088 | 0.3418 | 1,103 | 1,065 | 39,055 | 2,124 |
| Cylinder3D Re640 | legacy | 60,555 | 94.62% | 0.9037 | 0.1303 | 0.0896 | 0.5185 | 0.8054 | 0.0948 | 0.0850 | 287 | 2,741 | 54,436 | 3,091 |
| Cylinder3D Re640 | expanded | 42,463 | 66.35% | 0.8971 | 0.1211 | **0.0726** | 0.5085 | 0.7852 | 0.0805 | 0.0661 | 171 | 1,953 | 37,922 | 2,417 |
| Cylinder3D Re6400 | legacy | 62,313 | 97.36% | 0.9331 | 0.3277 | 0.3403 | 0.6505 | 0.9434 | 0.3453 | 0.3355 | 1,076 | 2,040 | 57,066 | 2,131 |
| Cylinder3D Re6400 | expanded | 57,906 | 90.48% | 0.9301 | 0.3225 | 0.3247 | 0.6395 | 0.9384 | 0.3360 | 0.3141 | 973 | 1,923 | 52,885 | 2,125 |
| Boeing 747 | legacy | 61,432 | 95.99% | 0.9182 | 0.1331 | 0.1715 | 0.5651 | 0.7846 | 0.1693 | 0.1738 | 520 | 2,552 | 55,888 | 2,472 |
| Boeing 747 | expanded | 17,601 | 27.50% | 0.9355 | 0.1555 | 0.2024 | 0.6112 | 0.8926 | 0.1635 | 0.2657 | 144 | 737 | 16,322 | 398 |

下载scene内的binary prediction与reference足以独立重算TP/FP/TN/FN、Coverage、Accuracy、Precision、Recall、
F1和BA，8/8均通过。63文件发布集合没有完整连续score与
support数组，因此本地没有独立重算AP、AUROC和support-subset F1；这些字段由CSV↔manifest逐字段一致、producer/
result/RUN哈希，以及Ibex reporter从sealed input进行的重算共同认证。

支持率解释如下；`imputed`是没有同时满足检索/校准支持时的空间补值，8行`unimputable=0`：

| Flow | Block | Retrieval/calibration supported | Support fraction | Spatially imputed | Imputed fraction |
|---|---|---:|---:|---:|---:|
| Re160 | legacy | 57,123 | 94.32% | 3,437 | 5.68% |
| Re160 | expanded | 10,592 | 24.44% | 32,755 | 75.56% |
| Re640 | legacy | 57,124 | 94.33% | 3,431 | 5.67% |
| Re640 | expanded | 10,590 | 24.94% | 31,873 | 75.06% |
| Re6400 | legacy | 58,769 | 94.31% | 3,544 | 5.69% |
| Re6400 | expanded | 11,513 | 19.88% | 46,393 | 80.12% |
| Boeing 747 | legacy | 59,949 | 97.59% | 1,483 | 2.41% |
| Boeing 747 | expanded | 13,867 | 78.79% | 3,734 | 21.21% |

最强逐图结果是Re160 legacy，F1=`0.5311`；最弱是Re640 expanded，F1=`0.0726`。Re640两个block的
Precision/Recall都接近零，分类几乎退化为负类基线；Re6400约为F1 `0.33`；Boeing约为`0.17–0.20`，且
expanded只覆盖27.50%的分配query。legacy与expanded的有效样本、support和补值率不同，不能把差值解释为只由
尺度扩展造成。已认证family-level half-cylinder F1=`0.404462`和Boeing F1=`0.241293`均低于项目目标
`0.70`；本次固定source图不改变该负面结论。

## 本地交付QA与哈希

- 不可变下载恰有63文件、41,120,156 bytes；`result_manifest`列出的61个artifact路径、大小和SHA-256全部一致；
  5个自哈希、RUN绑定、8个scene identity和20个继承scene arrays均通过。
- 8个scene各含`(240,32,4)`展示pathline，即240条、7,680点；总计1,920条。8个PNG均为
  7,560×1,800 RGBA、约360 dpi；8个PDF均为单页1,512×360 pt；8个SVG均可解析且各有43个文字元素。
- `nature-figure`源代码preflight为18 PASS、3个已审查WARN、0 FAIL。PDF文字审计8/8通过，最小字号
  7 pt，低于5 pt为0；panel alignment严格8/8通过，16组比较没有WARN/FAIL。
- 碰撞审计8/8可读，hard FAIL为0；text–text、text–stroke、page clipping均为0。95个WARN分为
  48个text–fill-edge和47个text–image-edge；全部只涉及三维坐标轴刻度字符，与8张overlay逐张目视复核后接受。
- 独立目视检查8/8通过：三panel、a/b/c、相机/边界、IVD、pathline、分类和四类错误标记均完整；没有空panel、
  无效轨迹、裁切或渲染破损。Boeing视图上部空白来自固定低z物理结构，不是绘制错误；全量query的密集遮挡属于
  预期现象。

关键文件SHA-256：`per_figure_metrics.csv=4cce86a2f42cd91f520a518e299d79d431da4931c895cb97a69a9259285a27f7`，
`visualization_manifest.json=7f2ea6d0a37c5db694ad015735d0989239aef0bf04958bc0e0457ad41ef22c81`，
`result_manifest.json=986d4a330cdee0ca4c92adc302e70b4fb4fdb5fe3ae74113d27f1a417476e7ee`，
`RUN_COMPLETE.json=376f731d82a8de4bd7ba1b7a79862a718a09211f49e5c3c7055aec5b5547f6cb`，
本地`delivery_qa_summary.json=2341c619d09d1811e2fb630b80b0cc955a16c39622c29c0a27921abcfd0fe45`。
结构化证据见`docs/evidence/Other_ClassConditionalTemplateScoreVisualization_1.1_local_summary.json`。
