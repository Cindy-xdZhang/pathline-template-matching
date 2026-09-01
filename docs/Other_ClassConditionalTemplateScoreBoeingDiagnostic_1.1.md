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

首次部署 job `51149373` 在455项测试通过后、Boeing runner 启动前失败。旧 resource audit 将
Verify config 的绝对路径固化为
`/home/zhanx0o/pathline-template-matching-class-conditional-score/config/Verify_ClassConditionalTemplateScore_1.1.yaml`；
job-local detached `58b0bc0…` clone 位于 `/tmp`，虽然 commit、config SHA-256 与source bytes均正确，
仍无法满足 public authenticator 的绝对路径合同。该失败未创建 fold output，也没有读取本诊断真实
Boeing 数组、候选、prediction 或指标；因此仍没有 Boeing 性能或图像结论。

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

## 运行后必须补写的证据

真实运行完成后只能追加：exact deployment commit、fold/auth job IDs、节点与实际资源、15文件fold和
四文件release路径、完整 SHA-256、stdout/stderr路径与SHA-256、Boeing单折指标及随后可视化QA。
任何负结果、失败、取消、超时或 partial directory 都必须保留。Boeing数字只能称
`exposed_post_stop_visualization_diagnostic`，不得与 half-cylinder 折平均成 two-family、five-family
或独立 test 结论。
