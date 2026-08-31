# `Other_FirstPrinciplesHeadroom_1.1`

状态：**`COMPLETED_AUTHENTICATED_POSTHOC_DIAGNOSTIC`**。唯一配置为
`config/Other_FirstPrinciplesHeadroom_1.1.yaml`，冻结 SHA-256 为
`a76ae95710f72a6432e4d392606fe4ca5ad4c0fb89b8d50e6d3868f546117477`。
该配置冻结于本诊断首次打开任何真实 `spatial_score` 或 outer reference array 之前；
设计时只检查了已经暴露的父实验 metadata/schema 和公开 aggregate summary，没有读取或
计算本诊断的 score、label、oracle 或新指标。此前 SHA-256 为
`f02120bbf4e67a69c3ba60d1850aeb5c8b22d50477adc84f1ff4533bc9f22020` 的草案从未提交、
从未运行；部署审查发现它没有固定父配置绝对路径、fold job/task basename 和父 F1 复现门，
因此在任何真实运行前被本配置明确取代。这个边界写入配置，不能把本诊断称为无偏模型选择。

## 1. 问题

已认证的 `Verify_EarlyOppositePairKinematics_1.1` 五-family macro F1 全精度值为
`0.6391632765825263`。本诊断不再修改 FMT 表示、距离或空间分数，只回答一个更窄的问题：

> 当前 `spatial_score` 的排序是否已经足以在理想判决下达到 F1 0.70–0.80，还是排序本身
> 已经限制了上限？

这是低成本的第一性原理分解。若同一 score 的 outer-label oracle 仍低于0.70，调一个全局
threshold 或正类比例不可能解决主要问题；若 oracle 明显超过0.70，但不使用 outer label 的
判决仍低，则 calibration/decision rule 值得进入新的、预注册 nested-family 验证。Oracle
只提供上限，不能成为 classifier、部署结果或论文主结果。

## 2. 冻结父证据

只允许读取已认证 Early 五折：

| 项目 | 固定值 |
|---|---|
| Headroom reporting checkout | `/home/zhanx0o/pathline-template-matching-headroom`；不得复用父 Early checkout |
| Early numerical commit | `2c3774dca0d81db8edd5645e63576526b9e276f7` |
| Early config绝对路径 | `/home/zhanx0o/pathline-template-matching-early-kinematics/config/Verify_EarlyOppositePairKinematics_1.1.yaml` |
| Early config SHA-256 | `e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767` |
| Early runner SHA-256 | `e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796` |
| Early aggregator SHA-256 | `631909159387cba854f471b3179ff0f0cd97404905e29b74589b2b8cf71f089e` |
| 完整认证 job | `51070392` |
| folds | `51070299_0` 与 `51070386_[1-4]` |
| Early五-family macro F1 | `0.6391632765825263`；本诊断current arm必须在绝对误差`1e-12`内复现 |

生产入口要求五个 fold directory 逐个从命令行传入，并逐 family 验证下面五个精确
directory basename，不能替换为同 commit 的其他重跑：

`slurm_51070299_0_2c3774dca0d8_outer_half_cylinder`,
`slurm_51070386_1_2c3774dca0d8_outer_delta_wing`,
`slurm_51070386_2_2c3774dca0d8_outer_f22_raptor`,
`slurm_51070386_3_2c3774dca0d8_outer_channel`,
`slurm_51070386_4_2c3774dca0d8_outer_boeing_747`。

它复用 Early aggregator 的
`_authenticate_fold`：重建 scaler、calibrator、selected candidate 和 label-free outer
prediction，认证精确15个文件、19个 prediction arrays 及所有哈希，再通过父实验既有
reference gate 打开 label。当前 checkout 还必须 clean、committed，且上述 runner 与
aggregator 的源码 SHA-256 必须逐字节不变。`--early-config` 还必须等于表中旧 producer
checkout 的绝对路径，因为父 result manifest 已固定该路径。随后只把 Early evidence 的 commit 字段钉回
真实 producer commit `2c3774d…`；这不是把新诊断冒充旧 numerical code。

允许的八个 train flows 固定为：

`cylinder3d, halfcylinderRe640, halfcylinderRe6400, deltaWing_resampled,
deltaWing_LBM, f22raptor, channel, boeing747`。

`tangaroa` 与 `smokeBuoyancy` 的 raw field、portable、cache、feature、label、prediction
和 metric 全部禁止。每个数据集必须有 source ordinal `0,1,2,3`，并把
`legacy_2_1` 和 `expanded_3_1` 分开；禁止跨 block pooling、投票或选优。

## 3. 四个判决 arms

所有 arms 使用父实验已发布并 fresh replay 的同一个 `spatial_score`。Eligible row 固定为
`calibration_supported OR spatial_imputed`；其他 row 强制判负。

| Arm | outer label进入判决？ | 定义 | 解释边界 |
|---|---:|---|---|
| `current_selected_prediction` | 否 | 父 Early 已认证 boolean prediction | 当前基准；完整 classifier 仍是 source/block 级 transductive |
| `inner_prevalence_top_fraction` | 否 | 用 nonouter inner validation 估计同 block 正类比例，再稳定选择该比例的最高 score | 可在未知 outer label 时执行，但本诊断结果可见后仍须新版本 nested 验证 |
| `label_free_exact_1d_two_means` | 否 | 每组对 eligible score 做全局最优一维 two-means，高均值簇判正 | 完全 label-free；constant 或不足两个值时全负 |
| `outer_group_oracle_max_f1` | **是** | 每组枚举 distinct positive score threshold，按 outer label 取 max-F1 | **只是一组一个 threshold 的乐观排序上限，绝不可部署** |

### 3.1 Inner prevalence

只从当前 selected candidate 在 `inner_group_metrics.csv` 中的 nonouter rows 读取
`positive_count/sample_count`：

1. 在每个 inner physical family 和 block 内，对 `dataset×source` prevalence 等权平均；
2. 再对四个 inner families 等权平均；
3. legacy 与 expanded 分别得到一个 outer-fold prevalence estimate；
4. 对每个 outer `dataset×source×block`，目标数为
   `ceil(estimate × all_valid_group_rows)`，再由 eligible 且 score 严格大于0的 rows 限制；
5. 排序固定为 score 降序、`center_seed_index` 升序。

因此 family 不会因包含更多 dataset 而获得更高权重，outer label 完全不参与比例估计。

### 3.2 Exact one-dimensional two-means

复用项目已有的 deterministic exact 1D two-means：对所有 distinct-value split 计算两侧
within-cluster sum of squares，取全局最小；完全相同的 objective 取最低 sorted split。
高均值 cluster 判正，同值永不拆分。未加入 Gaussian mixture model，因为额外 mixture
initialization、variance floor 和退化处理会把这个低成本诊断变成另一套方法。

### 3.3 Oracle

Oracle 只允许在 prediction 完整认证、`input_manifest.json` 发布、reference identity
精确连接之后调用。候选 threshold 是 eligible 且 score>0 的每个 distinct score，外加
全负。Score ties 整组进入；并列 max-F1 依次取更少 predicted positives、再取更高
threshold。它逐 `dataset×source×block` 读取 outer label，所以比一个可部署的统一
threshold 更乐观。输出和 summary 都固定写 `oracle_is_deployable=false`。

## 4. Identity join 与统计

Prediction 与 fresh reference 必须按以下字段完全一致且保持顺序：

`dataset, source_ordinal, source_index, scale_block_index, scale_id,
center_seed_index, assigned_row_index`。

任何 missing、extra、duplicate、reorder 或未知 dataset 都失败。主表单位是
`dataset×source×block`；随后在一个 outer family/block 内等权平均全部 dataset/source
groups，再对五个 outer families 等权平均。Legacy/expanded 各自报告，另给十个
`family×block` 的等权总览。每层同时报告 oracle 相对 current、inner-prevalence
和 two-means 的 F1 gap；其中 `oracle_f1_minus_current_f1` 是核心 headroom 量。
在发布任何 summary 前，`current_selected_prediction` 的十个 `family×block` 等权 macro
还必须以绝对误差不超过 `1e-12` 复现父 aggregate 的全精度 F1
`0.6391632765825263`。输出固定记录 `parent_f1_reproduced`、实际 delta 和容差；失败即关闭，
不能解释 oracle headroom。

输出指标为 Accuracy、Average Precision、F1、balanced accuracy、Area Under the
Receiver Operating Characteristic Curve、precision、recall 和 confusion counts。
Average Precision 与 Area Under the Receiver Operating Characteristic Curve 始终使用同一
父 `spatial_score`，因此它们诊断排序，不因 boolean arm 改变。

## 5. 结果解释规则

| 观察 | 允许的结论 | 禁止的结论 |
|---|---|---|
| all-block 或单 block oracle macro F1 <0.70 | 该 score ordering 对 calibration-only 修复不足 | 所有未来 representation 都不可能达到0.70 |
| oracle ≥0.70，三个无 outer-label arms 都较低 | calibration 仍是合理瓶颈假说 | 把 oracle threshold 部署或称成功方法 |
| inner-prevalence 或 two-means ≥0.70 | 值得冻结为新版本并做 nested family validation | 用当前已暴露结果直接宣布0.70成功 |
| oracle ≥0.80 | 当前 ordering 有较大 post-hoc headroom | 未见流场可达到0.80 |

本版本不会选择“表现最好”的 arm，也不会修改 Early 的停止结论。任何可继续的方法必须新建
`Verify_...`，在读取其 outer results 前冻结，并保持 complete physical-family split。

## 6. 输出

不可覆盖目录固定包含：

- `frozen_config.yaml` 与 `input_manifest.json`；
- `inner_prevalence_estimates.csv`；
- `group_metrics.csv`；
- `family_block_macro_metrics.csv`；
- `aggregate_summary.json`；
- `result_manifest.json` 与最后写入的 `RUN_COMPLETE.json`。

`input_manifest.json` 绑定五折的 completion、result、全部13个 result artifacts、Early
commit/config/source hashes 和32-shard input manifest。目录一旦存在立即失败；中途失败的
partial directory 原样保留，不能覆盖后伪装成首次运行。

## 7. Ibex 运行命令

在新代码 commit 推送后，用专用 reporting checkout
`/home/zhanx0o/pathline-template-matching-headroom` 检出该 exact commit 并保持 clean；冻结父 checkout
`/home/zhanx0o/pathline-template-matching-early-kinematics` 保持原样，只通过 `--early-config`
读取其固定配置路径。两个 root 必须不同。然后提交唯一生产 wrapper：

```bash
cd /home/zhanx0o/pathline-template-matching-headroom
sbatch --export=ALL,EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)" \
  ibex/other_first_principles_headroom_1.1.sh
```

Wrapper 固定32 CPU、128 GB、10小时、AMD EPYC 7702（Rome）节点和 `deepvortex` 环境；它
不接受调用者替换证据路径。三组已核验依赖为：

| 证据 | 绝对路径 | SHA-256 |
|---|---|---|
| kinematic input | `/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/kinematic_input_manifest.json` | `1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f` |
| production synthetic PASS | `/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/preparation/slurm_51068863_fd0412dc134d/SYNTHETIC_PASS.json` | `78d0990352777e488f26bb84f3b0fc16e18845fc7cedb8a7d7fc598f32c0afe3` |
| sidecar population | `/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/kinematic_cache/train/SIDECAR_POPULATION.json` | `9f96835b9185218f40df4cc3c52bf3d80a93056681d922a30abfc5c0246f88a7` |

输出目录由 wrapper 用真实 Slurm job ID 和 reporting commit 构造，存在即失败。这一步最贵的
部分是复用五折 fresh authentication；新增诊断本身只做 CSV stream、每组一维排序和指标
计算，不重建 FMT features 或模板库。

## 8. 运行前验证

- `python -m py_compile scripts/run_other_first_principles_headroom_1_1.py tests/test_first_principles_headroom.py`：PASS；
- `python tests/test_first_principles_headroom.py`：`9/9 PASS`；
- 合成覆盖：config SHA/freeze、19-array prediction authentication、NPZ tamper、tie-aware
  oracle、label-free two-means equal-objective tie、inner-family prevalence、exact Early config/fold
  paths、fresh-auth/input-freeze/reference call order、legacy/expanded macro、父F1复现门、outer-label
  arm boundary、Ibex wrapper 与 immutable nonoverwrite；
- `bash -n ibex/other_first_principles_headroom_1.1.sh`：PASS；
- `python tests/test_all.py`：`338/338 PASS`；
- 首次真实运行前没有 headroom F1、Ibex job 或性能结论；以下第9节单独保存运行后证据。

## 9. 已认证结果

Ibex job `51087135` 从 reporting commit
`2174418a642fd4a41416a7a693b88b8f4b9ea399` 运行，于2026-08-31
13:29:39–13:35:29 +03:00在`cn514-15-r`完成（`COMPLETED 0:0`，32 CPU，
128 GB，MaxRSS `13506076K`）。Wrapper 最终报告
`headroom_output_authentication=passed` 与
`headroom_status=complete_posthoc_diagnostic_authenticated`。父 current arm 的 all-block
five-family macro F1 精确复现为`0.6391632765825263`，差值为0。

| 判决臂 / 分块 | five-family macro F1 | 解释 |
|---|---:|---|
| current，all-block | `0.6391632765825263` | 已认证父结果；不是新模型 |
| inner-prevalence，all-block | `0.6352716877863082` | label-free；没有改善 |
| exact 1D two-means，all-block | `0.23611906536172073` | label-free；明显退化 |
| outer group oracle，all-block | `0.6736418047419102` | 使用outer label的不可部署乐观上限 |
| current / oracle，legacy | `0.6980627990650958 / 0.7148507101403403` | 只作已暴露分块诊断 |
| current / oracle，expanded | `0.5802637540999565 / 0.6324328993434799` | expanded排序更弱 |

核心 headroom `oracle-current` 只有`0.0344785281593839`；all-block oracle 未达到0.70，
更未达到0.80。由于这个 oracle 已为每个 outer group 使用真实outer label选择max-F1
threshold，它比任何可部署的统一threshold更乐观。因此本诊断反对继续只修改threshold、
预测正类比例或同一score的label-free二分规则；下一验证必须改变score ordering或表示。
这不证明所有未来表示都达不到0.70，也不能把legacy oracle `0.714851`称为部署成功。

发布目录为
`/ibex/user/zhanx0o/pathline-template-matching/Other_FirstPrinciplesHeadroom_1.1/runs/slurm_51087135_2174418a642f`，
恰含第6节规定的8个文件。`RUN_COMPLETE.json`、`result_manifest.json`、
`aggregate_summary.json` 的 SHA-256 分别为
`a87dc2b45d6ad6644e57d75774c3dcdc00670297f817bd50049529418a0753e3`、
`47808fafd8bb7dccc420339c456d44c3626d7a25c79d937a2505f012092e9652`、
`b6b9b3f89ddb7a0c087a9932eb52a5738436c5a06f8f70357b52412fcba66cf7`。
