# Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1：Boeing 单折暴露诊断合同

## 状态与目的

`Verify_ClassConditionalTemplateScore_1.1` 已在认证 half-cylinder 首折后停止：Ibex job
`51146768` fresh replay 得到 F1=`0.404461664553`，严格低于冻结的单 family 下限 `0.50`，因此
`stop_version=true`。剩余四个 Verify outer folds 禁止运行。

用户仍要求同一当前方法在 `cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400` 和
`boeing747` 上的三联图。前三个 flow 已包含在认证的 `half_cylinder` outer fold；只有 Boeing
缺少 prediction。因此本版本只补一个 `boeing_747` outer fold，并归类为 `Other` 暴露诊断。
它不是停止后的 Verify 续跑，不恢复五折，也不产生 success、stop 或 five-family macro 结论。

配置在实现和任何本诊断真实 feature、label、valid-rate、prediction 或 metric 读取前冻结为
[`config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml`](../config/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1.yaml)，
SHA-256 为
`6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856`。配置中的
`frozen_pre_run_not_implemented` 和 `current_state` 是冻结时的不可变历史记录，不得在实现或运行后回写。

本版本当前状态为 `COMPLETED_AUTHENTICATED_POST_STOP_DIAGNOSTIC`。最终数值与release commit均为
`6322d16cebe5995c8bcec2b8743e9ce0de9d8304`；fold job `51154451` 和独立 authentication job
`51154654` 均为 `COMPLETED 0:0`。`51154654` 从父 resource、父停止release、底层15文件Boeing
fold开始完整fresh replay，并逐字节重建四文件release后才发布下文的候选和指标。

失败和中间成功不得被最终job覆盖：

| job | commit | 结果与保留原因 | 当时允许的结论 |
|---|---|---|---|
| `51149373` | `417e673f4cda45a6a70d62e0ebac564316466dd3` | `FAILED 1:0`；455项测试后，`/tmp` detached Verify clone不匹配旧resource audit固化的producer config绝对路径；Boeing runner未启动且无fold目录 | 仅证明首次部署的父release认证路径错误；未读Boeing真实数组，无候选、prediction或指标 |
| `51152768` | `b9a7ba810811c6b9c0df3a7c8c2de0311da6298a` | `FAILED 1:0`；双固定根定向门通过，但full-suite入口仍导入已改名测试函数而`ImportError`；父release认证和Boeing runner均未启动 | 仅证明full-suite registry仍错误；无Boeing数值证据 |
| `51152907` | `4a2747a3829af0c22dd24a9d4dbc53f9522c1b42` | `COMPLETED 0:0`；456项测试、双固定checkout、父release门与15文件Boeing fold完成 | 只证明密封fold已计算；独立auth尚未成功，候选和指标仍不可发布 |
| `51153735` | `4a2747a3829af0c22dd24a9d4dbc53f9522c1b42` | `FAILED 1:0`；底层fold fresh replay已经通过，但release包装把深冻结后的`family_order` tuple再次交给只接受canonical JSON list的认证器，造成容器类型误拒；未创建release目录 | 反对当时的Other包装边界，不反对底层数值；仍无可发布候选或指标 |
| `51154451` | `6322d16cebe5995c8bcec2b8743e9ce0de9d8304` | `COMPLETED 0:0`；只在Other边界恢复JSON容器类型并增加回归测试后，456项测试和15文件fold完成；scaler/calibration NPZ、inner/outer CSV与prediction NPZ和`51152907`逐字节一致 | 证明容器修复未改变冻结数值，但仍须等待独立auth |
| `51154654` | `6322d16cebe5995c8bcec2b8743e9ce0de9d8304` | `COMPLETED 0:0`；456项测试、父resource/停止release、15文件fold、fresh prediction/support/label gate、四文件发布和public reauthentication全部通过 | 首次允许发布本页记录的Boeing候选、指标和固定源可视化输入 |

## 不变的科学方法

本版本完整继承 `Verify_ClassConditionalTemplateScore_1.1` 的数值合同，不允许任何 override：

| 身份 | 冻结值 |
|---|---|
| Verify config SHA-256 | `814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea` |
| class-conditional core SHA-256 | `9c009376f7cea1481f6f47a49362d54d0e78530717f480fda3e8a109f841ef99` |
| Verify runner SHA-256 | `e5063887475029320e66da1f1eb221d7988598e8918d37fbe47ee213e5ff1b48` |
| 修复后的 Verify authenticator SHA-256 | `77a561930ca85e3c1e6193a12e27b0b61bf7cc99be96889070962b8bfaf04e9c` |
| 已停止 Verify execution commit | `58b0bc0b0c7385f1b356eb343a150fcd50dad94f` |

完整继承范围是 `direct_parent`、evidence/input identity、physical-family split、三种表示、标签、共同
negative scaler、family/class exact-scale template 与 leave-one-out calibration、joint-family score、
空间变换、候选、inner selection、final refit/outer-label gate 和 metrics 共15项。

数值候选仍逐项等于原 Verify 的3,060项：三种表示，`k={1,5,15,31}`，
Gaussian sigma=`{0,0.5,1,1.5,2}`，以及每个组合下一个固定top-5%和50个严格
`score > threshold` 候选。Boeing outer 的 fit 和 inner-query families 固定为
`half_cylinder`、`delta_wing`、`f22_raptor`、`channel`；final outer joint support 仍为4个 fit
families 中至少3个。不得复用已停止 half-cylinder 折的 selected candidate、fit artifacts、prediction
或 metric；Boeing 必须重新做完整 nested selection、final refit 和 label-gated outer evaluation。

## 父停止证据与资源证据

下列证据只授权执行并记录来源，不是数值输入：

| 证据 | 固定身份 |
|---|---|
| Verify half-cylinder fold | job `51146327`，commit `58b0bc0b0c73…` |
| Verify single-fold authentication | job `51146768`，`AGGREGATE_COMPLETE.json` SHA-256 `f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844` |
| Verify resource smoke | job `51146125`，`RESOURCE_SMOKE_PASS.json` SHA-256 `3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea` |

每个 Boeing fold/auth job 都必须先调用两个公开 authenticator 重建上述 release。父停止 release 中的
候选、预测、特征、标签、指标、prevalence 和 support 不得进入 Boeing fit 或 selection。
Tangaroa 与 SmokeBuoyancy 继续全部禁止访问。

## 代码路径与运行顺序

| 角色 | 路径 |
|---|---|
| 薄配置 adapter | `scripts/run_other_class_conditional_template_score_boeing_diagnostic_1_1.py` |
| 独立 diagnostic aggregator/public authenticator | `scripts/aggregate_other_class_conditional_template_score_boeing_diagnostic_1_1.py` |
| Ibex common gate | `ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_common.sh` |
| Boeing fold wrapper | `ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_fold.sh` |
| release authentication wrapper | `ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_auth.sh` |

部署固定使用两个彼此独立、不可混用的 checkout：

| checkout 角色 | 冻结绝对路径与状态 |
|---|---|
| Verify producer | `/home/zhanx0o/pathline-template-matching-class-conditional-score`；必须 clean、detached，HEAD 恰为 `58b0bc0b0c7385f1b356eb343a150fcd50dad94f` |
| 本 Other 版本的 fold/auth/report | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing`；必须 clean，HEAD 恰为本次已 push deployment commit |

旧 resource 和 stopped single-fold 两个 public authenticator 必须从 Verify producer 的原绝对路径导入并
执行，因为旧 release 同时绑定 commit、config SHA-256、source SHA-256 和 config 绝对路径。任意 `/tmp`
clone、alternate worktree 或把两个角色放在同一 checkout 都不符合该release合同。common gate 在调用前后
都验证 producer HEAD、detached/clean状态和旧source hashes；认证结束后恢复 Other cwd/PYTHONPATH 并重跑
当前版本stage gate。认证进程还显式检查aggregator、runner、core与继承Git-identity函数全部从producer
根加载，禁止由Other checkout或环境中的同名module混入。三个 Boeing fold/auth/report wrappers 的chdir
与scheduler logs均只指向Other根。

adapter 只在互斥、可恢复的 runtime transaction 中把 Verify runner 的 experiment/config/output identity
换成本 `Other` 版本；实际 fit、selection、score、spatial transform、prediction、label gate 和15文件
transaction 仍由固定 Verify 实现执行。任何非 `boeing_747` outer family 在 Git 或数值访问前拒绝。

Ibex 请求固定为逻辑 `cpu` partition、实际认证 `batch`、account `pi-hadwigm`、Rome、32 CPU、
128 GiB、12小时、无GPU。每个 job 必须来自已 push 的 clean exact commit，并按以下顺序执行：

1. Other source/config/commit、独立 Verify producer 的 commit/clean/source身份与 Slurm allocation 认证；
2. synthetic targeted/full tests；
3. 父 resource PASS 和 stopped single-fold release fresh authentication；
4. Boeing-only fold，输出到新的不可覆盖目录；
5. 独立 authentication job fresh replay Boeing prediction/support，再打开 labels/metrics；
6. 发布四文件 diagnostic release，最后再由 public authenticator 完整重建一次。

提交后必须立即把 fold 和 auth 两个 Slurm process 分别登记到
[`docs/ibex_run_registry.md`](ibex_run_registry.md)，不能等完成后合并成一行。建议命令形态为：

提交前必须确认原 producer 根仍为 clean detached `58b0bc0…`，并从新的 Other 根提交：

```bash
cd /home/zhanx0o/pathline-template-matching-class-conditional-boeing
```

```bash
sbatch --partition=batch --export=ALL,EXPECTED_GIT_COMMIT=<clean-commit>,VERIFY_FIRST_FOLD_JOB_ID=51146327,VERIFY_FIRST_FOLD_AUTH_DIR=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/aggregate/slurm_51146768_58b0bc0b0c73,VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256=f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844,VERIFY_RESOURCE_SMOKE_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/resource_smoke/slurm_51146125_58b0bc0b0c73/RESOURCE_SMOKE_PASS.json,VERIFY_RESOURCE_SMOKE_PASS_SHA256=3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_fold.sh
```

fold 成功后以同样的冻结父 release 参数提交 auth job，并显式覆盖 wrapper 中仅作兼容声明的
partition：

```bash
sbatch --partition=batch --export=ALL,EXPECTED_GIT_COMMIT=<clean-commit>,BOEING_DIAGNOSTIC_FOLD_JOB_ID=<fold-job-id>,VERIFY_FIRST_FOLD_JOB_ID=51146327,VERIFY_FIRST_FOLD_AUTH_DIR=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/aggregate/slurm_51146768_58b0bc0b0c73,VERIFY_FIRST_FOLD_AUTH_COMPLETE_SHA256=f8515858efe531c24471a11f64f014692a5d4774146c8908f07ee4ca49476844,VERIFY_RESOURCE_SMOKE_PASS=/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1/resource_smoke/slurm_51146125_58b0bc0b0c73/RESOURCE_SMOKE_PASS.json,VERIFY_RESOURCE_SMOKE_PASS_SHA256=3f9197a19407906b0b13a2b9eaa09dbc647b166a9fe9d2ef4dc90cda532557ea ibex/other_class_conditional_template_score_boeing_diagnostic_1.1_auth.sh
```

## 四文件 release 与公共认证接口

authentication 目录必须恰好包含：

1. `boeing_outer_summary.csv`：一个 Boeing outer row，字段与已认证 Verify family row相同；
2. `boeing_diagnostic_report.json`：schema
   `pathline_template_matching.other_class_conditional_template_score_boeing_diagnostic_report.v1`；
3. `diagnostic_manifest.json`：schema
   `pathline_template_matching.other_class_conditional_template_score_boeing_diagnostic_manifest.v1`；
4. `DIAGNOSTIC_COMPLETE.json`：schema
   `pathline_template_matching.other_class_conditional_template_score_boeing_diagnostic_complete.v1`。

三个JSON逐层绑定 config、method、commit、Boeing CSV 文件名与SHA-256。manifest 的 `source_folds`
必须恰有一个 Boeing fold，并绑定15文件 transaction 中13个 result artifacts 的 size和SHA-256；
`RUN_COMPLETE.json` 与 `result_manifest.json` 另有独立 file SHA-256。四个release文件均不可覆盖。

公共接口为：

```python
authenticate_diagnostic_release(
    output_directory,
    *,
    expected_completion_sha256,
    expected_fold_commit,
    expected_config_sha256,
    expected_fold_directory,
)
```

它先 fresh-authenticate underlying Boeing fold，再逐字节重建四文件。返回恰好
`outer_family`、`fold_directory`、`fold_summary`、`selected_candidate`、`source_fold`、
`method_binding`、`release_files` 七项；`release_files` 将四个文件名映射到
`{size_bytes, sha256}`。release 不含 `stop_version`、success rule、five-family macro 或 formal
confirmation 声明。

## 可视化消费边界

四流场 reporter 必须消费两个独立且公开认证的 single-fold releases：

- `cylinder3d`、`halfcylinderRe640`、`halfcylinderRe6400` 只来自 Verify half-cylinder release；
- `boeing747` 只来自本 `Other` Boeing release。

两者只在 scientific projection（相同 representation/候选网格/fit/score/support/spatial/decision
合同）上相等；experiment、config、fold commit 和 release schema 必须保持各自身份，不能伪装成同一
五折 aggregate。图固定 source ordinal 2，并按 legacy/expanded block 各出一张。第一栏沿用认证的
固定 scene 中 IVD+pathlines；第二栏是 template classification，不称 clustering；第三栏显示
TP/FP/FN/TN。不得按 Boeing 结果调候选、阈值、block或图源。

## 已认证执行、数值与文件证据

以下数值只来自 job `51154654` 已通过public reauthentication的release，不从未认证fold日志或
partial目录摘取。权威release根为：

```text
/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5
```

底层fold根为：

```text
/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/runs/slurm_51154451_0_6322d16cebe5_outer_boeing_747
```

fold和auth均运行在`cn514-15-r`，scheduler实际partition/account为`batch/pi-hadwigm`，请求32 CPU、
128 GB、Rome、无GPU；runner的逻辑device为`cpu`。fold elapsed为`00:14:41`，auth elapsed为
`00:05:08`。

### 已认证候选与Boeing单折指标

候选由四个非Boeing fit/inner families完成nested selection后冻结，Boeing outer label未参与选择：

| 字段 | 已认证值 |
|---|---|
| 完整candidate ID | `representation=chirality_all35_plus_seed4\|k=5\|sigma=2.0\|fixed_top_fraction=0.05` |
| representation | `chirality_all35_plus_seed4` |
| nearest-neighbour count `k` | `5` |
| Gaussian sigma | `2.0` |
| decision rule | `fixed_top_fraction` |
| decision value | `0.05` |

`boeing_diagnostic_report.json`记录的完整outer population为316,358，其中positive 13,274、negative
303,084。指标为：

| 指标 | 已认证值 |
|---|---:|
| Accuracy | `0.933401444856` |
| Average Precision (AP) | `0.195482698918` |
| F1 | `0.241293471263` |
| Balanced Accuracy | `0.630076850892` |
| Area Under the Receiver Operating Characteristic Curve (AUROC) | `0.862722808175` |
| Precision | `0.206196930266` |
| Recall | `0.301414558230` |
| True Positive / False Positive / True Negative / False Negative | `3,346 / 12,477 / 290,607 / 9,928` |
| retrieval/calibration-supported | `295,442 / 316,358 = 0.881772113507` |
| spatial-imputed / spatial-unimputable | `20,916 / 0` |

最终support仍为四个fit families中至少三个共同支持；support audit的`family_order`为
`half_cylinder, delta_wing, f22_raptor, channel`，`required_joint_family_count=3`。这部分也由fresh
replay重新计算并和持久化JSON逐字段相等后才发布。

### 15文件fold transaction

`diagnostic_manifest.json`逐项绑定以下15个底层文件；前13个是result artifacts，后两个是独立
completion/result envelopes：

| 文件 | file SHA-256 |
|---|---|
| `final_per_scale_scaler.npz` | `6358f8a5f3b4b87b012bcbd44c1ed6ffaccebd7a070c3cd76cdbaca67defcd05` |
| `final_per_scale_scaler_manifest.json` | `49badf65455057584a14e57174ce583f1b2acd8a2dafd391f3e7e83ae42447e0` |
| `final_tail_calibration.npz` | `619298f3e31c56a59c01e8243d570718c8bee91b716dd19de95ce67b52cf195c` |
| `final_tail_calibration_manifest.json` | `9d179f88d088fa3da2a3bf92c6b3a671ae6caf7a23e79c24d6ad29b0f809b5ec` |
| `inner_candidate_summary.csv` | `dd19fa5b4d8debb1bc0f3b8cac2628f2dc00f182ce0f31b29984b308a063dd42` |
| `inner_fit_audits.json` | `29246ad0046d17f042501d9cff530bac9dd47f0a4a69e3400c443c9339695442` |
| `inner_group_metrics.csv` | `b4466905ad89c35e50f79ecdfc4d8710745fb8dc681e69fbe032ac6e2e661af7` |
| `outer_group_metrics.csv` | `c6f0b0f5ef24410d4ce6e96bf36e72c3816c8ac6e108339368fb52a3fc05d6c1` |
| `outer_prediction_manifest.json` | `1155e8945163b45f3198fc06f198f0661a038c61781b8ffb414dbe5369268684` |
| `outer_predictions.npz` | `a0604bde369d91f3c6a8223bebbb1602935fa62cd92fe2e2013524f20c35b059` |
| `outer_reference_access_audit.json` | `e2ea1818444d2a2164ec6cb4e98a83bc5e977f7781e04aa045cfc7802dda4718` |
| `outer_summary.json` | `fc184b3cdff73dcec37d6ce0e16646aee9d5dd5afefbd463e91ed50d819708c2` |
| `selected_candidate.json` | `9103ea2b854b640d1eb36f89ca39440de1b6ceb2956551883e884ccdf7efa0d2` |
| `RUN_COMPLETE.json` | `cb56f0e1aef8f8fa5f2c59d78ad6f25b8a9265545523603964bd4bb3086a1036` |
| `result_manifest.json` | `c4a69d988e832582e92878ac410f02ddb6a8929c886479b0958872f242b3b4d9` |

### 四文件release

authentication目录恰含下列四个不可覆盖文件：

| 文件 | 绝对路径 | file SHA-256 |
|---|---|---|
| `boeing_outer_summary.csv` | `/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5/boeing_outer_summary.csv` | `d4c1083bc11eed340dbbbe62d8ff5b5f9ae9bcaf341ba7ca9a7620b3d20493e8` |
| `boeing_diagnostic_report.json` | `/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5/boeing_diagnostic_report.json` | `dd0a001571f27e9572258f920a7f0b8065121f84274678214c0d76cb54588f55` |
| `diagnostic_manifest.json` | `/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5/diagnostic_manifest.json` | `9f764e5f447704f2370f39c6530d10fd1bf98bf0d254a2559fe410fb7c4f1998` |
| `DIAGNOSTIC_COMPLETE.json` | `/ibex/user/zhanx0o/pathline-template-matching/Other_ClassConditionalTemplateScoreBoeingDiagnostic_1.1/authentication/slurm_51154654_6322d16cebe5/DIAGNOSTIC_COMPLETE.json` | `a9bb930c540c366dd9fd9fd040bdca306cbb7a0a2fcd829fe5f307a8e85ad12c` |

三个JSON内部canonical `content_sha256`分别为report
`f906061f80756025ab273caae52ccf92fe5fbbdd21b7feb8330a86d6531219eb`、manifest
`a04ba18ca79e01ea8e8d9385f1d94bd76feaff66fe44088a3ab25c395d1efff9`、completion
`8c45f8e7e59ae9632a542cb58f44d3ae7c28fccc478ea3370b8e974b2d65c7c8`；这些内部hash不能和上表的
整文件SHA-256混名。

### Scheduler日志

| job | stdout路径与SHA-256 | stderr路径与SHA-256 |
|---|---|---|
| `51149373` | `/home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/PTMClassBoeing.51149373.out`；`8e05f557d1eb7bc79ff5f630f4ea165bbbce9ac109813dc7034e7bf5abb59dc6` | `/home/zhanx0o/pathline-template-matching-class-conditional-score/slurm_logs/PTMClassBoeing.51149373.err`；`360a8a8481d68a478b385ba8eafdc07050f228c99ccbc498d3d5169506f3f49d` |
| `51152768` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51152768.out`；`5dc2d1e8702df96e2e95ae7f12806603dbfc360c16c6940c1137185f1b9d9809` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51152768.err`；`98e673cc928d446d428755963200f92f6ffc7dfc3f8fadadd6302b50df224e6f` |
| `51152907` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51152907.out`；`3a7022bf68296da056daecf26c33764d3ad44e7c9e3be46b43ace6fa4e1962c7` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51152907.err`；`8a759aeae6eed32a7da37e0e3f605ea3defae264a31315fc31be684ae8fbc00c` |
| `51153735` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeingAuth.51153735.out`；`a99611a58e3fa9754c05f81f06defbc4bf202678096554d28408133a31d8d280` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeingAuth.51153735.err`；`1fbb72ef27b313f79de00e7cf4d86a48656325e04e354216e693412a7ce20da9` |
| `51154451` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51154451.out`；`a4387c85d702a75975850e760b87be60d3f2da355d1522789d7914b31c28827e` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeing.51154451.err`；`00f155344a952da624497ff2155169323875c8e2c90d53ad6efdc54ff409ae06` |
| `51154654` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeingAuth.51154654.out`；`b1765b9f3e2eabb8423a24c7a8f3b909e4fbe4899d9446af2a6589831f2f6e70` | `/home/zhanx0o/pathline-template-matching-class-conditional-boeing/slurm_logs/PTMClassBoeingAuth.51154654.err`；`34fd2f1156ce652db4f069aad0462c6cc56d0bddae6d5029de517b8ec03c2fd3` |

## 当前结论与边界

旧结论“Boeing性能未知，因为部署或认证尚未完成”变为当前结论：在commit
`6322d16cebe5995c8bcec2b8743e9ce0de9d8304`、config
`6112e7588efecf29cf2690b270385053d8ccd94f8e11037a6e247815afcc5856`和上述候选下，Boeing暴露诊断
得到F1=`0.241293471263`、Average Precision=`0.195482698918`、precision=`0.206196930266`、
recall=`0.301414558230`。Accuracy=`0.933401444856`不能单独解释为分类良好，因为316,358个样本中
303,084个为negative，且False Negative为9,928。结论改变的原因是`51154654`首次完成独立fresh replay
和四文件public authentication；旧结论是当时证据尚未发布的更窄状态，并非相反的性能判断。

该结果只能称为`exposed_post_stop_visualization_diagnostic`：

- 它支持固定source可视化和描述性错误分析，不是sealed confirmation或独立test generalization；
- 它不恢复已停止的`Verify_ClassConditionalTemplateScore_1.1`，不重新评估success/stop；
- 不得把Boeing和half-cylinder折平均成two-family或five-family macro，也不得用它声称四流场总体性能；
- Boeing结果已经暴露，不得据此修改representation、`k`、sigma、threshold/top fraction、support规则、block或图源；
- 本版本完成的是Boeing prediction/release输入认证；后续三联图仍须由独立reporter同时认证half-cylinder和本Boeing两个release，并单独记录渲染与目视QA。
