# `Verify_DimensionlessDeformationFMT_1.1`

当前实现状态：**`VALIDATION_FIXED_RERUN_FIRST_FOLD_SUBMITTED`**。唯一配置为
`config/Verify_DimensionlessDeformationFMT_1.1.yaml`，冻结文件 SHA-256 为
`c689b1d265bbc39327b2ed4147e8ffb22450dcd26f87b7c19ceae346c9ecfe18`。
本配置冻结于首次读取 `Verify_EarlyOppositePairKinematics_1.1` 或
`Verify_RawPCANegativeMetric_1.1` 的任何 outer 结果之前；这个时间关系和配置内的
历史状态以后不得因实现、提交或运行而改写。

配置中的历史字段仍为`status: frozen_pre_run_not_implemented`，用于保留预注册时点，
不得因当前进度改写。数值核心
`src/pathline_template_matching/dimensionless_deformation_fmt.py` 已按冻结公式实现，
SHA-256为`5fc4acb47c52c6505737e661cac7f8f503c429c5d88910992655e83cdc53a649`；
冻结合同、数值核心、production runner、独立aggregator及四阶段Ibex wrapper均已实现并
通过本地合成门禁；覆盖解析公式、单位缩放、proper rotation、batch/chunk/row permutation、
退化输入、父FMT ID/坐标索引、Raw-only 4096-row重编码、15-file/19-array事务、fresh replay、
outer-label gate、数学提前停止、混合provenance拒绝、不可覆盖发布与固定Rome CPU架构。
冻结YAML中的`status: frozen_pre_run_not_implemented`只记录首次预注册时点，仍保持不变。
实现commit为`9a48650c219f4cada12df722d780ea383e03bb89`；validation-only修复commit为
`46f02e60bb345c4e2f7f6ece6aba88cca09f1f6a`。修复后runner/aggregator SHA-256
分别为`e977fa6754ad3029bfaf3e7e5f5334babac018854daec28a46dc4df11a2e01ea`与
`ec9a10ce0885542084844815845485837b33fffa428bc85608801f382424d91d`。同一工作树的
修复后统一回归为`339/339 PASS`（2026-08-31，215.429 s）。首次Ibex job
`51087139`在338项回归与matcher backend门通过后，于首个runner `load_plan`失败；原因是
runtime rebinding后错误地用可变父模块全局量检查冻结身份。失败发生在创建output目录、
读取真实cache、拟合、prediction、outer label或metric之前；依赖认证job `51087140`
从未启动并已取消。因此仍没有任何方法性能结果，不能将该失败解释为支持或反对无量纲表示。

## 1. 研究问题与单一变化

本版本以 `Verify_PerScaleNegativeMetric_1.1` 为父方法。研究问题是：在每个
pathline primitive 内消除绝对长度单位和共同平移轨迹的幅度，是否能改善完整物理
family 留出时的迁移性能。

父方法的逐尺度负类方差度量、`lambda=64` 方差收缩、fit-negative leave-one-out
尾概率、same-scale exact retrieval、`k`、空间 Gaussian、判决网格、nested family
拆分、选择、成功和提前停止规则全部保持不变。唯一数值变化发生在 independent FMT
之前：把 Raw672 恢复为 `7×32×3` 后，逐行构造冻结的无量纲 deformation 表示。

本实验不扫描其他归一化公式，不引入 Principal Component Analysis（PCA，主成分
分析）、whitening、学习参数、descriptor 权重、query 无标签自适应或尺度重分配。

## 2. 输入与固定顺序

唯一输入 member 为 float32 `raw_features[N,672]`。按 C order 恢复为
`[N,7,32,3]`，七线顺序固定为：

```text
center, x+, x-, y+, y-, z+, z-
```

Raw672 的冻结前提是 `center[0]` 精确为零。任何 shape、dtype、line order 或该原点
合同不符都必须 fail closed。

## 3. 冻结的无量纲变换

对单个 primitive，记中心线第 `t` 个点为 `c_t`，第 `j` 条邻线为 `n_j,t`。所有
计算先转为 float64：

```text
L_c = sum(t=0..30) ||c_(t+1) - c_t||_2
d_0 = (1/6) * sum(j=1..6) ||n_(j,0) - c_0||_2

center_out[t]     = c_t / L_c
neighbor_out[j,t] = c_t / L_c + (n_j,t - c_t) / d_0
```

`L_c` 必须 finite 且严格大于零，并且只能由实际 center polyline 的31段计算；禁止
使用 config 中的目标 arc level 或 scale table。`d_0` 必须由六个实际初始相对向量
的欧氏长度取均值；禁止使用 scale ID 或 dataset grid metadata。六个长度必须在
`rtol=5e-5, atol=1e-7` 内相等，三个 opposite pairs 的 midpoint 也必须在相同容差
内回到 center。输出必须 finite，并以 float32 序列化。

这不是把全部七线统一除以一个全局尺度，而是一个预注册的混合无量纲嵌入：中心
共同轨迹以其实际弧长归一化，邻线相对中心的 deformation 以实际初始邻距归一化。
这种区分是方法定义的一部分，不能在看到结果后改成其他公式。

变换严格逐 row：禁止 batch/query statistics、flow statistics、train fit、label、IVD、
hidden clipping 和 log。单个 primitive、不同 batch、不同 chunk 或不同 row order 的
输出必须相同。

## 4. Descriptor 与 3060 个候选

无量纲输出仍为 `7×32×3`，随后进入未修改的 independent FMT；父 descriptor ID
固定为 `fmt_independent_3d_161d_sha256_25fce29499c9089e`，完整宽度为161。三个表示
只复用父方法原有 coordinate index sets：

| 表示 | 定义 |
|---|---|
| `fmt161_dimensionless_deformation` | 无量纲输出的完整161D independent FMT |
| `real_neighbor36_dimensionless_deformation` | 父方法不变的 real-neighbor 36 coordinates |
| `chirality_all35_dimensionless_deformation` | 父方法不变的 chirality 35 coordinates |

三种表示权重均为1，无可训练参数。候选数固定为：

```text
3 representations × 4 k × 5 sigma × (1 fixed top-5% + 50 thresholds)
= 3 × 4 × 5 × 51
= 3060
```

其中 `k={1,5,15,31}`，空间 sigma 为 `{0,0.5,1,1.5,2}` 个 grid cells，tail
threshold 为闭区间 `0.50,0.51,...,0.99`。候选只按完整 inner family 等权证据
选择；outer feature、label 或 metric 禁止参与选择。

## 5. 数据、输入身份与拆分

只允许读取 `mainExp_TemplateMatching_3.1` 的32个 train cache shards。冻结输入为：

- manifest：`.../Verify_LongArcHorizon_1.1/train_coverage/`
  `slurm_50998592_260a07ad380d/train_cache_input_manifest.json`；
- file SHA-256：
  `e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`；
- rows content SHA-256：
  `ceb6d0e3fb7a2c90fcaae98583f8d1def7ee75fa7968f38d2821ee3040ae156f`；
- row count：32；`test_dataset_access=false`。

五个 family 与 outer/inner 顺序都固定为
`half_cylinder, delta_wing, f22_raptor, channel, boeing_747`。拆分单位是完整 physical
family，禁止随机拆 seed。具体 dataset 组成保持：half-cylinder 三个、delta-wing
两个，其余三个 singleton family。

**Tangaroa 与 SmokeBuoyancy 的 raw、portable、cache、feature、label、prediction 和
metric 全部禁止访问。** 本版本只可能产生已暴露 train-flow development 证据，
不是 formal confirmation。

## 6. Outer 门禁与不可覆盖发布

每折必须先认证 clean exact commit、config 和冻结输入，再只用 nonouter families
拟合 transform 之后的父 per-scale metric、tail calibrator并选择候选。由于本变换
`train_fit=none`，这里的“fit”只属于不变的父 retrieval/calibration，不能新增变换统计。

在 outer label 前必须先写出并认证 label-free prediction，再从磁盘 fresh reload
transform 与 prediction 完整重放。聚合器只能先 stage label-free artifacts，fresh
replay 通过后才能打开 outer labels、result 或 outer metric artifacts，并重新计算
group、family 和 five-family 指标。

所有发布使用同目录临时文件 `fsync`，再 hard-link no-replace，最后 parent-directory
`fsync`；任何旧 artifact 禁止覆盖。

## 7. 成功与提前停止规则

成功必须来自五个唯一 outer families 全部完成并通过认证，同时满足：

- five-family macro F1 ≥ 0.70；
- 至少4个 family 的 F1 ≥ 0.65；
- 任一单 family F1 均 ≥ 0.50；
- five-family macro Average Precision ≥ 0.60；
- five-family macro balanced accuracy ≥ 0.70；
- five-family macro precision ≥ 0.60；
- five-family macro recall ≥ 0.60。

所有条件必须同时满足。提前停止只允许数学上已不可能成功时发生：任一已完成 family
F1 < 0.50；已有两个 family F1 < 0.65；或把全部剩余 family 的相关指标设为1仍无法
达到任一 macro 门槛。单折不得宣称方法成功。

## 8. 结论边界与风险

1. **混合尺度解释。** 中心轨迹和邻线 deformation 使用两个不同的实际尺度；若性能
   改变，只能归因于这个完整冻结表示，不能单独归因于某一个除法。
2. **退化 primitive。** 零中心弧长、非有限值、六邻距不等或 opposite-pair midpoint
   不闭合都必须失败，不能通过 clipping、epsilon 或 metadata 替代修复。
3. **保留的 transductive 部分。** independent FMT 和当前逐行变换是 query-independent，
   但 positive sigma 与 fixed top-5% 仍依赖完整 query group；整个 classifier 不能称为
   逐 primitive independent。
4. **证据范围。** 即使达到门槛，也只支持这8个已暴露 train flows 上的 nested-family
   development 结论；formal confirmation 需要未见 physical families。

只有从clean committed revision在Ibex运行现有runner/aggregator/wrapper，并通过五折
fresh-replay认证后，才能在实验日志新增性能结论。冻结config本身不得改写。

## 9. 首次Ibex尝试与修复

- `51087139`：failed commit `2174418a642fd4a41416a7a693b88b8f4b9ea399`，
  `FAILED 1:0`，elapsed `00:02:52`，node `cn514-15-r`，MaxRSS `677656K`；
  output目录不存在。
- stderr定位到`ValueError: parent freeze identity drifted`。原runner在
  `dimensionless_parent_runtime`中暂时重绑父模块全局量，随后子`load_plan`又读取这些
  可变全局量，造成自相矛盾；原父loader还会错误看到子config SHA，这是同一身份绑定问题。
- commit `46f02e60bb345c4e2f7f6ece6aba88cca09f1f6a`捕获稳定
  `PARENT_EXPERIMENT`，并在transaction内用闭包固定子config、父config和core哈希，返回进入
  transaction前已认证的immutable plan。生产路径不再依赖被重绑的父全局量。
- 新增production regression
  `test_production_load_plan_uses_stable_parent_identity_inside_runtime`；`py_compile`、
  `git diff --check`和`339/339`统一回归全部通过。下一次Ibex提交必须从exact修复commit，
  或保持runner、aggregator、config和core上述精确SHA不变的纯报告后代运行，并获得新的
  job ID；禁止覆盖或删除失败记录。
- 修复后专用Ibex checkout已在clean detached HEAD
  `de130540d050984e14247e0f4db4cbbf1eefd77f`复核config/core/runner/aggregator哈希与
  4/4 wrapper语法。新首折job为`51088712`，依赖它成功的独立认证job为`51088726`；
  两者未完成前仍无方法性能结论。
