# Other_ClassConditionalTemplateScoreVisualization_1.1：双认证单折分类三联图

状态：**`ATTEMPT_2_FAILED_AFTER_RENDER_BEFORE_WRAPPER_ACCEPTANCE_JSON_HASH_FIX_IMPLEMENTED`**。尚无通过完整
wrapper认证的结果图或报告指标；第二次尝试虽写出`RUN_COMPLETE.json`，但Slurm终态失败。本实验只生成用户要求的Cylinder3D Re160、Re640、Re6400和Boeing 747当前
class-conditional template-score分类效果；它不恢复已停止的Verify五折，也不把两个不同实验的单折证据
伪装成完整五折。

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

Slurm `COMPLETED`只证明63文件transaction及自动门禁完成。下载后仍须逐张完成8/8 PNG目视检查、PDF/SVG
文字可编辑性、panel alignment、裁切与碰撞检查，并确认颜色和TP/FP/FN/TN说明准确。在本地QA完成前状态不得
提升为最终图件；任何性能描述都必须注明两个single-fold evidence source、不是完整五折、不是formal
confirmation。
