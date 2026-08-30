# `Verify_EarlyOppositePairKinematics_1.1`

状态：**`FROZEN_PRE_RUN_NOT_IMPLEMENTED`**。唯一配置为
`config/Verify_EarlyOppositePairKinematics_1.1.yaml`，原始文件 SHA-256 为
`e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767`。
本配置冻结于首次读取 `Verify_PerScaleNegativeMetric_1.1` 的任何 outer
结果之前；冻结时 PerScale jobs `51063738/51063753` 已提交，但没有读取任何
outer prediction、label、metric、stdout 中的 outer 数值或 aggregate 结果。这个
历史字段以后不得因实现、提交或运行而改写。

本版本尚无 production implementation、tests、kinematic sidecar、Ibex job 或性能
结果。本文档只定义后续实现必须满足的合同，不证明方法有效。

## 1. 研究问题与唯一变化

本版本以 `Verify_PerScaleNegativeMetric_1.1` 为数值父方法。唯一变化是为父方法
的三个 FMT 表示逐行追加同一个、无参数的 4D seed-time kinematic block：

| 父表示 | 当前表示 | 宽度 |
|---|---|---:|
| `fmt161` | `fmt161_plus_seed4` | 165 |
| `real_neighbor36` | `real_neighbor36_plus_seed4` | 40 |
| `chirality_all35` | `chirality_all35_plus_seed4` | 39 |

拼接顺序固定为父 FMT coordinates 在前、4D block 在后，原始 block weight 固定
为 1。禁止增加 `seed4-only` 第四候选，也禁止扫描 block weight、log transform、
time window、kinematic Discrete Fourier Transform（DFT，离散傅里叶变换）或
mean-vorticity correction。三个表示的顺序、`k`、spatial sigma、decision grid、
candidate 数、fit-negative per-scale metric、tail calibration、nested split、选择、
成功与提前停止规则全部继承父方法。

因此本版本只有一个可归因的数值机制：**在不改变 retrieval/calibration 的条件下，
加入 seed-time 局部速度梯度的旋转不变量**。

## 2. 为什么不能直接复用现有 cache 或 FMT 44D kinematic block

当前 3.1 cache 不包含物理正确构造 early kinematics 所需的完整时间/速度证据：

1. `build_phase21_cache_slice` 只从 valid primitive 保存 center line 的
   `center_sample_time[N,32]`；见
   `src/pathline_template_matching/phase21_pipeline.py:1478-1502`。
2. `centered_xyz` 在形成 Raw672 前明确丢弃 time channel；见
   `src/pathline_template_matching/primitives.py:212-219`。所以 Raw672 可以恢复
   centered 7×32 XYZ，但不能恢复六条 neighbour line 的各自 sample times。
3. Cache exact member set 只有 `center_sample_time`，没有 7×32 line times、
   seed velocities 或 velocity gradient；见
   `src/pathline_template_matching/phase21_pipeline.py:2092-2118`。Recovery contract
   也把该数组冻结为 `(valid_count,32)`；见同文件 `2213-2265`。
4. Arc-length resampler 在每条 line、每个 crossed segment 内独立计算
   `sample_time=time+alpha*step`；见
   `src/pathline_template_matching/arc_length_primitives.py:541-566`。因此
   `x+ / x- / y+ / y- / z+ / z-` 的相同 sample index 通常不在同一物理时刻。
   `line_end_time` 和 `line_steps` 不能重建中间 32 个 times。
5. FMT Task5 旧 cache 同样只保存 Raw/FMT/seeds/scale metadata，没有 per-line
   sample times 或 initial velocities；见
   `C:/Users/xingdi/sources/FMT/Build_Task5_Multiscale_Cache.py:190-205`。其
   `task5_sample_times` 只按 old rounded integration indices 重建一个所有 line
   共享的 time vector；见
   `C:/Users/xingdi/sources/FMT/FMT_Utils/Task5FeatureRecipes_3D.py:55-68`，不适用于
   当前逐 line arc-length resampling。
6. 当前项目和 FMT 的旧 44D recipe 都执行
   `vorticity - vorticity.mean(dim=0)`；当前项目见
   `src/pathline_template_matching/fmt_descriptor.py:148-160`，FMT 见
   `C:/Users/xingdi/sources/FMT/FMT_Utils/DFT_FMT_3D.py:188-200`。这使一个 query
   的 feature 随 batch composition 改变，违反 independent-query 合同。

因此 1.1 不从 Raw pathline 的第一段估计 velocity，也不把 center times 错配给
neighbour lines。它只研究所有七个 seed 确实同步的初始时刻。

## 3. 冻结输入与禁止访问范围

只允许使用 3.1 的 32 个 train cache shards 与匹配的 32 个 train portable
windows：

- parent cache commit：`260a07ad380d64fc300cabe8926244e92d8ba04a`；
- main config SHA-256：
  `771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e`；
- parent cache schema：`pathline_template_matching.phase31_cache.v1`；
- train cache input manifest SHA-256：
  `e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393`；
- train portable population marker SHA-256：
  `489d303b4430be7eded4fe39ab87107c778e1f7db2579cb9e3bb1fdfce209341`。

必须先构建并冻结一个 32-row
`pathline_template_matching.seed_time_opposite_pair_kinematics_input.v1`
manifest，再打开任何 portable `velocity` member。每行同时绑定 parent cache 与
portable file 的 path、size、file SHA-256、canonical array SHA-256、builder commit
及 `dataset/family/source ordinal/source index`。

Input freeze 与 sidecar build 只允许从 parent cache 打开：

`seeds_xyz, valid_assigned_row_index, valid_center_seed_index,
valid_scale_block_index, valid_scale_id, center_sample_time`。

此阶段禁止打开 `valid_labels`、`reference_labels_all`、`ivd_values_all`、
`ivd_volume` 和 parent `metadata_json`。后者含 label-derived counts，不能用作
label-free sidecar provenance 的捷径。

**Tangaroa 与 SmokeBuoyancy 的 raw files、portable windows、cache arrays、features、
labels、predictions 和 metrics 在本版本全部禁止访问。** 本版本即使成功也只是
已暴露 8-train-flow development 验证，不是 formal confirmation。

## 4. 七个同步 seed velocity

对每个 parent valid assigned row：

```text
c = seeds_xyz[valid_assigned_row_index]
h = dx_grid_scale[valid_scale_id] * min(Δx, Δy, Δz)
points = [c, c+h ex, c-h ex, c+h ey, c-h ey, c+h ez, c-h ez]
```

`dx_grid_scale` 必须从 frozen 3.1 explicit scale table 读取；禁止运行时重建
`linspace`。Portable `x/y/z` 必须满足既有 finite、strictly increasing、uniform
contract。七个 points 全部在同一个 relative time `t0=0`、portable frame 0
取样；absolute source time 只存 provenance。

Velocity sampler 必须使用 production RK4 `_interp4_quadrilinear_scalar` 相同的
corner selection 和 arithmetic order。输入与存储 velocity 为 float32；越界或
nonfinite 一律失败。Center velocity 也保存，用于审计，但不进入中央差分。

## 5. 中央差分与 4D feature

令 `u_x+` 表示在 `c+h ex` 的三分量速度，其他符号同理。速度梯度 `G` 的 row
是 velocity component，column 是 spatial derivative：

```text
G[:,0] = (u_x+ - u_x-) / (2h)
G[:,1] = (u_y+ - u_y-) / (2h)
G[:,2] = (u_z+ - u_z-) / (2h)
```

随后以 float64 计算：

```text
omega = [G[2,1]-G[1,2], G[0,2]-G[2,0], G[1,0]-G[0,1]]
S     = (G + G^T) / 2
Omega = (G - G^T) / 2
div   = trace(G)
Q     = 0.5 * (||Omega||_F^2 - ||S||_F^2)
seed4 = [||omega||_2, ||S||_F, div, Q]
```

`div` 和 `Q` 保留符号。无 absolute-value divergence、log、batch/flow statistics
或 whole-volume mean-vorticity correction。所有结果必须 finite，然后以 float32
序列化。`Q` 与 curl/strain norms 在代数上相关，但本次按预注册要求保留且不调权。

## 6. Companion sidecar，而不是改写 parent cache

已有 3.1 cache 是 immutable、commit-bound evidence。现有 loader 还要求 exact
member set；直接添加 arrays 会在
`src/pathline_template_matching/phase21_pipeline.py:2116-2122` fail closed。因此
本版本必须建立不覆盖 parent 的 companion sidecar：

`pathline_template_matching.seed_time_opposite_pair_kinematics_cache.v1`。

Exact arrays 为：

| Array | dtype | shape |
|---|---|---|
| `valid_assigned_row_index` | int64 | `[Nv]` |
| `valid_center_seed_index` | int64 | `[Nv]` |
| `valid_scale_block_index` | int8 | `[Nv]` |
| `valid_scale_id` | int32 | `[Nv]` |
| `seed_velocity_xyz` | float32 | `[Nv,7,3]` |
| `seed_kinematic4` | float32 | `[Nv,4]` |
| `physical_dx_by_scale` | float64 | `[2000]` |

Sidecar 只保存 valid parent rows 且保持完全相同顺序；missing、duplicate、extra 或
reorder 均失败。`metadata_json` 不得包含 label 或 label-derived count，必须绑定
input manifest、parent/portable identities、config、clean Git commit、算法 source
SHA、composite descriptor、line/time/interpolation/dtype contract，以及全部 array
canonical hashes。Sidecar 原子发布且禁止覆盖。

这个 additive 设计不修改 frozen `fmt_descriptor.py`、`encoder.py`、
`arc_length_primitives.py` 或既有 3.1 cache schema。

## 7. Synthetic gate

任何真实 sidecar build 之前，production sampler 和 kinematic function 必须通过
无真实流场的 synthetic gate，并最后写 immutable PASS marker。最低要求为：

- 任意 affine field `v(x)=Ax+b` 在全部 frozen `dx` 上逐元素恢复 `A`；
- rigid translation、rigid rotation、pure strain、isotropic expansion 的
  curl/strain/divergence/Q oracle；
- constant proper rotation/translation 下 4D scalar invariance；
- single/batch/chunk/permutation invariance；
- 错误 opposite-pair order 被 oracle 检出；
- public sampler 与 production RK4 seed-time first `v1` 一致；
- member-access test 证明 sidecar build 未打开 label、IVD 或 parent metadata；
- exact identity join 拒绝 missing、duplicate、extra 和 reorder。

Synthetic gate 禁止读取任何真实 flow value、label 或 valid rate。

## 8. Nested family split、fit 与 3060 candidates

五个 outer families 与顺序保持：

1. `half_cylinder`；
2. `delta_wing`；
3. `f22_raptor`；
4. `channel`；
5. `boeing_747`。

每个 outer fold 的四个 inner folds 仍按 complete physical family 留出；一个 source
及其 48-frame future window 永不拆分。不得随机拆 seed。

Natural-negative library、exact-scale population variance、other-scale within-scale
residual prior、variance-domain `lambda=64` shrinkage、effective-std floor、exact
same-scale Euclidean retrieval、leave-one-out negative-tail calibration 和 support/
fallback modes逐字段继承 PerScale。Composite feature 的所有 coordinates 一起拟合
fit-negative exact-scale scaler；outer family 不进入 fit。

Candidate grid 保持：

```text
3 representations × 4 k × 5 sigma × (1 fixed top-5% + 50 thresholds) = 3060
k = {1,5,15,31}
sigma = {0,0.5,1,1.5,2}
threshold = 0.50,0.51,...,0.99
```

Inner selection 的 family/group 宏平均、primary F1 和 AP/BA/precision/recall/
candidate-ID tie breaks 不变。

## 9. Outer label gate

每个 outer fold 必须保持以下不可逆顺序：

1. 只用 nonouter sidecars、nonouter parent FMT 与 nonouter labels 完成 inner
   selection；
2. 只用 final fit families 拟合并原子写出 final per-scale scaler、tail calibrator
   和 selected candidate；
3. 从磁盘 fresh authenticate 三者，并绑定 input/sidecar population manifest、
   composite descriptor ID、fit-family set、config 与 clean commit；
4. 此后才可打开 outer sidecar feature 与 outer parent FMT feature；不得打开 outer
   label 或 parent metadata；
5. 写出 outer prediction，并 fresh recompute/authenticate feature join、scaler、
   calibration、spatial transform 和 decision；
6. 只有 prediction 闭环认证后，才允许打开 parent `valid_labels`，按 exact
   dataset/source/scale/center/block/assigned-row/order join 后计算指标；
7. 写独立 outer-reference-access audit。

Outer labels 不得选择 feature、representation、scaler、`k`、sigma、threshold 或
任何 fallback。

## 10. 成功与提前停止

成功必须来自五个唯一 outer families 齐全的 authenticated aggregation，同时满足：

- family macro F1 ≥ 0.70；
- 至少 4/5 families 的 F1 ≥ 0.65；
- minimum family F1 ≥ 0.50；
- macro Average Precision ≥ 0.60；
- macro balanced accuracy ≥ 0.70；
- macro precision ≥ 0.60；
- macro recall ≥ 0.60。

单折不得宣称成功。只有出现以下任一认证条件才可提前停止：一个已完成 family F1
严格低于 0.50；两个已完成 family F1 严格低于 0.65；或把所有剩余 family 的
相关指标都设为 1 仍无法达到任一五-family macro 门槛。失败、停止、取消和无效
run 都必须保留。

## 11. 成本估计

已认证 3.1 train population 为 32 shards、4,096,000 assigned、2,967,612 valid；
见 `docs/ibex_run_registry.md` job `50998455`。现有 train cache 总文件大小约
8.20 GB；该值来自已下载 run 的 `cache_manifest.json`。

按本配置 exact arrays，未压缩 sidecar payload 约为：

```text
2,967,612 × (21 identity bytes + 84 velocity bytes + 16 feature bytes)
= 359,081,052 bytes = 0.334 GiB
```

`physical_dx_by_scale` 与 metadata 只增加很小开销。实际 NPZ 大小依压缩率而定，
预计约 0.15–0.35 GB；这是估计，不是运行证据。若错误地保存全部 7×32 line
times，仅 train valid rows 未压缩就需 2.476 GiB，所以不属于本最小版本。

Sidecar build 只解压每个现有 portable window 的 frame 0、做 trilinear sampling
和小矩阵运算，不重新进行 H48 integration。建议仍用 32-task CPU array、每 task
16 CPU/32–64 GB、15–30 min walltime request；全部 tasks 获得资源后预计数分钟内
完成，但这是依据 3.1 cache jobs 最长 81 s 的工程估计。五折 retrieval 的宽度只
比父方法多 4，因此预计接近 NegativeTail/PerScale 的 5–13 min/fold 量级；实际
elapsed、CPU-hours、MaxRSS 必须由 Slurm 登记替代本估计。

## 12. 结论边界与主要风险

1. **直接标签代理风险。** IVD-p95 label 本身来自 seed-time curl deviation；若
   local curl-related block 提高 F1，只能结论为“Eulerian seed-time local
   kinematics 对该 label 有用”，不能称 FMT geometry 或 pathline history 学会涡结构。
2. **Observer 风险。** 未减 whole-volume mean vorticity 的 curl magnitude 与 Q
   对 time-dependent rotating observer 不完全 objective；`channel` 又是 Killing
   observer 产生的 synthetic unsteady view。1.1 明确禁止 correction，避免同时改
   第二个机制或直接复制 label construction。
3. **尺度数值风险。** `dx=0.125–2.5 hmin` 会分别出现 float32 cancellation 和
   finite-difference truncation/smoothing；只允许报告逐尺度诊断，不能据 outer
   结果删除或重权尺度。
4. **Q redundancy。** 标准 Q 可由 curl/strain norms 推得；保留它会改变距离权重，
   但这是本次预注册的固定选择，不得结果后移除。
5. **Transductive 决策。** Positive sigma 与 fixed top-5% 仍依赖完整 query group；
   即使 descriptor 独立，也不能称整个 classifier 是逐 primitive independent。

只有实际实现、synthetic gate、32-sidecar population authentication、五折 Ibex
运行及 aggregate authentication 全部完成后，才能把本状态改写到运行日志和实验
版本表；冻结 config 本身不得改写。
