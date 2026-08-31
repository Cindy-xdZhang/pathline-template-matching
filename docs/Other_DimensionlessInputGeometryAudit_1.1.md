# `Other_DimensionlessInputGeometryAudit_1.1`

当前状态：**`COMPLETED_AUTHENTICATED_LABEL_FREE_AUDIT`**。唯一配置为
`config/Other_DimensionlessInputGeometryAudit_1.1.yaml`，冻结 SHA-256 为
`c874a8d9f6abbab452c6543139073eea2ac88e3db99ea13f78e0c3d43e03f566`。

本实验是只读、无标签的输入兼容性审计。它不修改
`Verify_DimensionlessDeformationFMT_1.1` 的公式、容差、表示、候选或成功规则，也不产生
分类性能结论。

## 1. 起因与问题

Ibex job `51088712` 在真实 nonouter Raw672 第一次无量纲重编码时触发：

```text
the six realized initial neighbor distances are unequal
```

该失败发生在 fit、候选选择、prediction、outer cache/label 和真实 metric 前，是
`pre-fit real-cache compatibility failure`，不是性能负结果。旧 job 中 nonouter labels 已随
cache 打开，但没有进入 inner metric；本审计通过成员白名单彻底排除 labels。

待检验解释为：3.1 producer 先把 absolute seed 与 `seed±dx` 写入 float32 samples，再做
float32 center subtraction，因此较大 absolute coordinate 上的小 `dx` 可产生六个不完全相等
的 Raw 初始距离。审计必须回答这个解释是否覆盖全部32个train shards中的冻结门禁失败；不能
只检查已知的 `deltaWing_LBM` 反例，也不能用结果反推并修改容差。

## 2. 冻结输入与访问边界

唯一输入 population 是 `Verify_LongArcHorizon_1.1` 的32-row train cache manifest：

```text
/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/
verification/Verify_LongArcHorizon_1.1/train_coverage/
slurm_50998592_260a07ad380d/train_cache_input_manifest.json
```

- file SHA-256：`e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`；
- size：`24009` bytes；
- rows-content SHA-256：
  `ceb6d0e3fb7a2c90fcaae98583f8d1def7ee75fa7968f38d2821ee3040ae156f`；
- parent cache commit：`260a07ad380d64fc300cabe8926244e92d8ba04a`；
- 8个train datasets，各4个source ordinals；`test_dataset_access=false`。

每个 NPZ 必须先按 manifest 的完整文件 size 与 SHA-256 认证，再且仅打开以下成员：

```text
raw_features
valid_scale_id
valid_center_seed_index
valid_scale_block_index
valid_assigned_row_index
seeds_xyz
```

禁止打开 `valid_labels`、`reference_labels_all`、`metadata_json`、`fmt_features`、全部IVD
成员、validity/line diagnostics、sidecar、portable、raw flow、test cache、prediction 或
metric。Tangaroa 与 SmokeBuoyancy 全部禁止。

## 3. 两类独立检查

### 3.1 原冻结门禁重放

Raw672 按C order恢复为 `[N,7,32,3]`，顺序固定为：

```text
center, x+, x-, y+, y-, z+, z-
```

以float64重算六个sample-0相对向量和欧氏距离，逐row重放：

- 六距离相等：`rtol=5e-5, atol=1e-7`；
- 三个 opposite-pair midpoint：相同容差；
- center sample 0精确为零；
- 六向量只有各自指定轴可非零。

审计报告上述初始几何检查以及`dx=0`失败的完整并集，但不把失败行删除、修补或传给
descriptor。严格正的真实`h`可能因absolute float32量化而表现为零差值，因此`dx=0`仍进入
rounding-envelope反演；center非零或off-axis非零则不可能由冻结producer表达式解释。

### 3.2 无metadata的float32 rounding envelope

对每个方向，已知 float64 seed `s` 和 Raw float32差值 `q`。runner反演满足

```text
q = float32(neighbor_float32 - float32(s))
neighbor_float32 = float32(s ± h)
```

的保守 closed round-to-nearest 区间，再求六方向共同正标量 `h` 的交集。该计算不读取
scale table、grid spacing 或 `metadata_json`。随后在每个 `dataset×source×scale_id` 内再次
求全部rows的共同 `h` 区间。

这里使用closed midpoint envelope，因此结论方向是保守的。逐row存在六方向共同`h`只是
必要条件；同一`dataset×source×scale_id`的全部rows还必须共享非空`h`交集。若失败row自身
无共同`h`，或其scale group没有一个可同时解释全部rows的共同`h`，则“纯float32量化解释”
被证伪；只有两级检查都通过，才能说该解释支持观察到的失败，且仍不能由此授权改写1.1容差。

## 4. 输出合同

每次运行使用新的、不可覆盖目录，只允许四个最终文件：

```text
per_shard_geometry.csv
per_scale_geometry.csv
summary.json
RUN_COMPLETE.json
```

两个CSV按冻结 dataset、source ordinal、scale ID排序。逐shard与逐scale均报告门禁失败数、
rounding-envelope可解释/不可解释数、实际`dx`范围、relative spread、绝对距离差与opposite-pair
residual；逐scale另存共同`h`区间。

`summary.json`绑定两个CSV的path、size、row count和file SHA-256，并保存去掉
`content_sha256`字段后的canonical JSON self-hash。CSV不能在自身内容中保存自身hash，否则会
形成循环；它们由summary绑定。`RUN_COMPLETE.json`最后写入，绑定summary的file SHA-256与
content SHA-256，且自身也带self-hash。Production入口在发布该最后marker前再次验证Git commit与
clean worktree；该门失败时只保留partial输出，不允许出现`RUN_COMPLETE.json`。

## 5. 结论规则

- 若任一冻结门禁失败row不满足共同`h` rounding envelope：记录
  `quantization_only_hypothesis_falsified_by_unexplained_gate_failures`。
- 若至少有一个门禁失败且全部可解释：记录
  `quantization_explanation_supported_for_all_observed_gate_failures`。
- 若没有门禁失败：记录`no_frozen_gate_failure_observed`。

无论哪种结果，都禁止写成 classifier performance、descriptor improvement、formal
confirmation，或`Verify_DimensionlessDeformationFMT_1.1`通过。若后续改变接受门禁，必须先
建立新的预注册 Verify 版本；若重建float64-before-centering cache，则还必须建立新的parent
cache/input身份。

## 6. Ibex运行

wrapper固定为`ibex/other_dimensionless_input_geometry_audit_1.1.sh`，使用32 CPU、128 GB和
`cpu_amd_epyc_7702`。提交前必须把实现提交并同步到固定checkout：

```bash
cd /home/zhanx0o/pathline-template-matching-dimensionless-deformation
git fetch origin
git checkout --detach <FULL_COMMIT>
mkdir -p slurm_logs
EXPECTED_GIT_COMMIT=<FULL_COMMIT> sbatch ibex/other_dimensionless_input_geometry_audit_1.1.sh
```

提交后必须按项目协议立即把job登记到`docs/ibex_run_registry.md`。

## 7. Ibex结果与结论

Job `51092739` 从clean detached commit
`f7ce798d57d86cb47a05d3664b2d896059682cc6`运行，2026-08-31 15:30:56--15:32:35
+03:00在`cn514-14-r`完成，exit code `0:0`，elapsed `00:01:39`，Slurm MaxRSS
`8,583,488 KiB`。Wrapper 的7项合成/认证测试与最后fresh authentication均通过。

完整32 shards包含2,967,612个valid rows与52,665个observed shard-scale groups。结果为：

| 检查 | 行数 |
|---|---:|
| 任一冻结初始几何门失败 | 57,446（1.935765%） |
| 六距离不等 | 55,686 |
| opposite-pair不闭合 | 57,238 |
| center-origin / off-axis / zero-`dx`失败 | 0 / 0 / 0 |
| 逐row rounding envelope不可行 | 0 |
| 失败且无法由逐row+同尺度共同`h`解释 | 0 |

按dataset的失败行数为：Re160 `2,194/416,006`，Re640 `2,192/412,323`，Re6400
`41,181/481,037`，`deltaWing_resampled` `0/294,504`，`deltaWing_LBM` `248/294,547`，
F22 `0/437,257`，channel `9,115/315,580`，Boeing 747 `2,516/316,358`。Re6400占全部
失败的71.69%，其自身失败率为8.560879%；这说明问题不是只有最初观察到的
`deltaWing_LBM`，也不是单一flow特例。

认证结论是
`quantization_explanation_supported_for_all_observed_gate_failures`：父cache先存absolute
float32坐标、再做float32中心化的生产顺序足以解释全部冻结初始几何失败。这个结果把此前
“原因未知”修订为“全部观察失败与该量化机制一致”，但不授权修改1.1。若继续验证无量纲表示，
必须新建预注册版本，明确选择producer-aware acceptance、从冻结scale构造逻辑初始几何，或重建
float64-before-centering cache；三者是不同方法/输入合同，不能把任一方案静默写回1.1。

四个artifact SHA-256为：`per_shard_geometry.csv` `cad91135...`，
`per_scale_geometry.csv` `9c58963a...`，`summary.json` `4e00a259...`，
`RUN_COMPLETE.json` `2585126e...`。完整结构化摘要见
`docs/evidence/Other_DimensionlessInputGeometryAudit_1.1_ibex_summary.json`。
