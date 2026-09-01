# Verify_ClassConditionalTemplateScore_1.1：运行前冻结合同

## 当前状态与证据边界

本实验是 **exposed-development-only（已暴露开发集上的验证）**，不是独立确认实验。配置已经冻结为
[`config/Verify_ClassConditionalTemplateScore_1.1.yaml`](../config/Verify_ClassConditionalTemplateScore_1.1.yaml)，
当前最终字节级 SHA-256 是
`814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea`，并在本文末尾重复。最初未提交草案 SHA-256
`19be180258e791aab8a14705294d80030feb543eccc3bad76148fac4c7de5cbd` 错把同一 family、同一 class
的跨尺度校准先验禁用，因而错误地把所有 `n=k` 情形判为 calibration unsupported。该草案尚未提交、
没有 runner，也没有读取任何本版本真实数组；现已在首次真实读取前按原预注册设计修正并保留此记录。

冻结时状态是 `frozen_pre_run_not_implemented`：尚未为本版本打开任何真实 feature、label、valid-rate、
prediction 或 metric 数组，尚未运行 resource smoke、任一 outer fold 或聚合，也没有本版本性能结果。
这里只使用了已公开的父实验 metadata、manifest 身份和公开 summary 来定义合同；因此本文不能支持
“优于父版本”“达到 0.70”或任何新性能结论。以后开始实现或运行时，不得回写这段历史状态。

## 直接父版本与唯一方法变化

直接父版本只能是 `Verify_EarlyOppositePairKinematics_1.1`，不能经由 PerScale、dimensionless、
headroom 或 oracle 实验间接继承。冻结身份如下：

| 项目 | 冻结值 |
|---|---|
| 父 numerical commit | `2c3774dca0d81db8edd5645e63576526b9e276f7` |
| 父配置 SHA-256 | `e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767` |
| 父 runner SHA-256 | `e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796` |
| 父 aggregator SHA-256 | `631909159387cba854f471b3179ff0f0cd97404905e29b74589b2b8cf71f089e` |
| 父五折认证 job | `51070392` |
| 父公开 family-macro F1 | `0.6391632765825263`，仅作历史背景，不进入本版本选择或打分 |

本版本复用父版本已认证的三种 FMT+seed4 表征、数据身份、exact-scale negative scaler 算术、
family nested split、空间平滑网格和成功/停止门。唯一科学方法变化是：把父版本的
negative-only tail anomaly 改成 **family/class exact-scale conformity contrast**。父 tail-calibration、
selected candidate、outer prediction 和 metric 都不是本版本输入，必须重新 fit、选择和预测。

三种表征及顺序保持不变：

1. `fmt161_plus_seed4`，165 维；
2. `real_neighbor36_plus_seed4`，40 维；
3. `chirality_all35_plus_seed4`，39 维。

不得重算或改变表征、追加新特征、学习距离权重、平衡 class/family、抽样 template，或把
legacy scales（0--999）与 expanded scales（1000--1999）混成一个 scale library。

## 共同负类标准化

每一个 inner fit 或 final refit 都只用**当前 fit families 的全部 natural negative rows**重新拟合一套
共同 scaler。这套 scaler 同时作用于每个 family 的 positive library、negative library 和 query；禁止
positive-fitted、class-specific、family-specific 或 query/outer-fitted scaler。

算术逐 exact numeric scale 冻结为父 Early 规则：float64 拟合和存储，标准化后用 float32 计算距离；
local mean 是该 scale 的所有 fit-negative rows 算术平均，local variance 是 `ddof=0` population
variance。other-scale prior 先分别减去每个 other scale 自己的 local mean；优先同一 block，只有
block-other 为空才用跨 block global-other。以 `n_s/(n_s+64)` 在 variance domain shrink，开平方后
对严格小于 `1e-12` 或为零的标准差使用 1.0。没有 local negative row 的 exact scale 不建立 retrieval
或 calibration 支持，不能用 broader/global library 补齐。这是缺少 exact-scale scaler 与 retrieval 的
前置失败；它不禁止在 scaler 和 exact-scale retrieval 已支持时使用同 family、同 class 的跨尺度
calibration reference prior。

## Family/class exact-scale k-nearest-neighbour conformity

这里的 k-nearest neighbours（kNN）是：在共同 negative scaler 标准化后的空间中，按精确 Euclidean
distance 取第 `k` 个最近样本。对每个 representation、`k ∈ {1,5,15,31}`、fit physical family、
class（negative 或 positive）和 exact numeric scale，library 是该 family/class/scale 的**全部**自然
标注行；retrieval library 不得跨 family、class 或 scale 池化。跨尺度只允许用于下述 calibration
reference prior，而且必须保持同一 physical family 和同一 class。

对 class `c`：

- query distance `d_c` 是到对应 family/class/exact-scale library 的第 `k` 小距离；retrieval 至少需要
  `k` 行；
- local calibration reference 是该 library 每一行经过 leave-one-out 得到的第 `k` 小距离。只把当前行的
  self-distance 设为正无穷，其他 duplicate（包括零距离）全部保留；local reference 本身因此至少需要
  `k+1` 行，但这不是整体 calibration support 的最低行数；
- 对任何一组 reference distances `r_i`，经验 upper-tail conformity 为
  `q_c = (1 + count(r_i >= d_c)) / (N_ref + 1)`。相等距离计入 `>=`，所以相等时采用较大的、保守的
  conformity；
- `n<k` 时 exact-scale retrieval 不支持；`n=k` 时 retrieval 有效但 local LOO reference 为空；
- calibration 只允许在同一 physical family、同一 class 内使用父方法冻结的尺度先验。优先使用同一
  block 的其他尺度 LOO references；若为空才使用另一 block。local reference 存在时以
  `N_local/(N_local+64)` 将 local conformity 与该 prior conformity 加权；没有 prior 时只用 local；
  local 为空时可只用 prior。若 local 与 prior 都为空，则 calibration 不支持；
- 这里跨尺度的只有 calibration reference prior，exact-scale retrieval library 始终不跨尺度；禁止
  跨 family 或跨 class 的 reference pooling。

Calibration 必须逐项复现父 `negative_tail_calibration` 的六种 exact mode。对每一个 reference multiset
都单独使用上面的 plus-one、`>=` upper-tail 公式，不能先拼接 local 与 prior 再算一次：

- `local_block_shrink`：local 和同 block 其他尺度 reference 都非空；后者严格是
  `block multiset - local multiset`，结果为 `w*q(local)+(1-w)*q(block_other)`；
- `local_global_shrink`：local 非空、同 block 其他尺度为空、另一 block 非空；结果为
  `w*q(local)+(1-w)*q(other_block)`；
- `local_only`：local 非空且两个 prior 都为空，直接使用 `q(local)`；
- `block_fallback`：local 为空但当前 block reference 非空，使用 `q(current_block)`；这正是 `n=k`
  可以通过同 block 其他尺度获得 calibration support 的情况；
- `global_fallback`：local 与当前 block reference 都为空但另一 block 非空，使用 `q(other_block)`；
- `none`：三者全空，`q` 未定义且 calibration unsupported。

其中 `w=N_local/(N_local+64)`。因为只有两个固定 scale blocks，`global_other` 就是另一 block；这些
跨尺度 shrink/fallback conformity 不能称为 exact conformal p-value 或 posterior probability。

一个 fit family `f` 只有在同一 query/representation/exact-scale/`k` 下，positive 和 negative 两类的
retrieval 与 calibration **四项都支持**时，才属于 jointly-supported family 集合 `J`。两类 exact-scale
都至少需要 `k` 行；当某类只有 `k` 行时，还必须有同一 family、同一 class 的其他尺度 LOO prior。

单 family 分数和最终分数冻结为：

```text
S_f = 0.5 * (1 + q_positive - q_negative)
S   = mean_{f in J}(S_f)
    = 0.5 * (1 + mean_J(q_positive) - mean_J(q_negative))
```

`J` 中 family 严格等权；不得按 row count、reference count、dataset count、distance 或 prevalence
加权。`S` 不是 `q_positive/q_negative` 比值，也不是 family 判正票数。

- Inner query 有 3 个 fit families，必须 `|J| >= 2`（2-of-3 joint-support gate）。
- Final outer query 有 4 个 fit families，必须 `|J| >= 3`（3-of-4 joint-support gate）。

这两个数是**支持门**，不是“2 票判正”或“3 票判正”。未通过门的 raw row 标为 unsupported 且 raw
score 为 0；只有后述 mask-normalized spatial imputation 的分母非零时，邻域信息才能补入。

## 空间变换和 3,060 个固定候选

空间 group 固定为 `(dataset, source_ordinal, scale_block)`，每组网格为 `40×40×40`，不允许跨
dataset、source 或 block 平滑。Gaussian sigma 为 `{0,0.5,1,1.5,2}` grid indices，truncate 为 3。
当 sigma 大于 0 时：

```text
spatial_score = G(S * joint_support_mask) / G(joint_support_mask)
```

分母为零的行 score 为 0 并显式判 negative；sigma 为 0 时只保留直接通过 joint-support gate 的行。

每个 representation/`k`/sigma 有 51 个固定 decision：一个 fixed top-5% 和 50 个 threshold
`0.50, 0.51, ..., 0.99`。top-5% 的目标数是 `ceil(0.05 * all valid group rows)`，再以 eligible
positive-score row 数封顶；排序 tie 固定为 score 降序、center index 升序。Threshold 的科学语义
严格是 **`class_conditional_template_score > threshold`**，等于 threshold 必须判 negative。

为复用 Early 已认证 runner 的 serialized schema，artifact/candidate ID 内允许保留兼容字段名
`calibrated_tail_anomaly_threshold`；在本版本它只表示上述 class-conditional score 的严格 `>`
decision，**不再表示 negative-tail anomaly**。Candidate ID 必须同时编码 representation、`k`、sigma、
decision type，以及 threshold 值和 strict comparator，避免把两种语义混淆。

总候选数在任何真实结果读取前固定为：

```text
3 representations × 4 k × 5 sigma × (1 top-5% + 50 thresholds) = 3,060
```

Average precision 和 area under the receiver operating characteristic curve（AUROC）使用连续 spatial
score；F1、balanced accuracy、precision、recall、accuracy 和 confusion counts 使用冻结 decision。

## 完整 family split 和选择

Physical family 与 dataset 的映射固定为：

- `half_cylinder`: `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400`；
- `delta_wing`: `deltaWing_resampled`, `deltaWing_LBM`；
- `f22_raptor`: `f22raptor`；
- `channel`: `channel`；
- `boeing_747`: `boeing747`。

Outer 和 inner 顺序都固定为
`half_cylinder, delta_wing, f22_raptor, channel, boeing_747`。每次 outer fold 的拆分如下：

| Outer family（完全不可用于选择） | 四个 nonouter families；每次轮流留一个作 inner query，另外三个 fit |
|---|---|
| `half_cylinder` | `delta_wing`, `f22_raptor`, `channel`, `boeing_747` |
| `delta_wing` | `half_cylinder`, `f22_raptor`, `channel`, `boeing_747` |
| `f22_raptor` | `half_cylinder`, `delta_wing`, `channel`, `boeing_747` |
| `channel` | `half_cylinder`, `delta_wing`, `f22_raptor`, `boeing_747` |
| `boeing_747` | `half_cylinder`, `delta_wing`, `f22_raptor`, `channel` |

一个 family 必须完整离开，source 加 48 个 future frames 也是不可拆单位；禁止随机 spatial seed split。
每个 inner query 都用剩余三个 families 重新 fit shared negative scaler、family/class libraries 和
leave-one-out calibrators。先在每个 inner family 内对 dataset×source×block group 等权平均，再对四个
inner families 等权平均。

候选选择顺序固定为最高 family-macro F1；若完全相同，依次比较最高 average precision、balanced
accuracy、precision、recall，最后取 lexicographically smallest candidate ID。Outer feature、label、
metric、prevalence 或 support 不得进入选择。

## Final refit、outer label gate 与调用顺序

选定候选后，必须在四个 nonouter families 上重新 fit共同 negative scaler 和全部 family/class/
exact-scale library 与 leave-one-out calibrator。下面四类 final artifacts 必须先原子写入、关闭并重新
认证：

1. `final_shared_negative_exact_scale_scaler`；
2. `final_family_class_exact_scale_template_and_LOO_calibration_bundle`；
3. `selected_candidate`；
4. `final_support_policy`（包括 3-of-4 gate）。

冻结调用顺序是：

1. 验证 clean commit、当前配置、父源码、input manifest、32-sidecar population 和 exact family set；
2. 只在 nonouter data 上完成四个 inner fits、candidate selection 和 final refit；
3. 写完并 fresh-authenticate 上述 final artifacts；
4. 此时才允许打开 outer sidecar 的 identity 与三种 representation，仍不得打开 label、父 prediction
   或父 metric；
5. 完整生成、关闭并 fresh-authenticate 本版本 outer prediction；
6. fresh replay authentication 通过后，才允许打开 parent `valid_labels`，按
   dataset/source/scale/center/block/assigned-row/order 精确连接并计算 outer metric。

任何缺失、重复、额外、乱序 identity，或任何提前 outer-label member access 都必须 hard fail。不得
复用父 prediction 或 metric。所有输出使用 hard-link-without-replace 原子发布，不得覆盖既有目录。

## Mandatory resource smoke

真实 fold 前必须先运行一个只验证资源和 exact code path 的 smoke；它不是性能实验。固定保留且完全
不打开 `half_cylinder` outer family 和 `delta_wing` inner family，只打开 fit families
`f22_raptor`, `channel`, `boeing_747` 的父 sidecar feature/identity 与 fit labels。Smoke 以最宽的
`fmt161_plus_seed4`、最大 `k=31` 和三个 fit families 中出现的全部 exact scales 构建完整共同 scaler、
family/class libraries 与 leave-one-out references，再用 deterministic label-free synthetic queries
走查询 path。

Ibex 请求固定为 CPU Rome node、`deepvortex` account、32 CPUs、128 GB、4 小时、无 GPU。只有在
exit code 为 0、peak memory 严格小于 128 GB、elapsed 不超过 4 小时、数组全部 finite、self-exclusion/
duplicate/support audit 全通过、没有打开 forbidden 或 reserved family，并且 PASS marker 最后原子写入
时才通过。Smoke 禁止输出 accuracy、F1、average precision、AUROC、inner/outer prediction、selected
candidate 或性能判断。

若 smoke 因资源失败，只能修改不改变 exact arithmetic 和认证输出的 chunking、streaming 或 storage
plumbing；任何数值或方法变化都必须新建实验版本，并在再次读取真实数据前冻结。

## 运行门、成功门与停止门

Resource smoke 前必须有 clean committed revision、当前 config/source hashes、共同 scaler、
leave-one-out conformity、duplicate/tie、joint support、严格 `>` 和完整 3,060 enumeration 的 synthetic
tests，以及 outer member-access tests。Smoke 通过后才可提交第一折；第一折完成后必须先通过 fresh
replay、artifact call-order 和 outer-label gate 认证，并评估停止证书，才能提交其余四折。

五折成功需要同时满足：family-macro F1 ≥ 0.70；至少 4/5 family F1 ≥ 0.65；最低单 family F1 ≥
0.50；family-macro average precision ≥ 0.60、balanced accuracy ≥ 0.70、precision ≥ 0.60、recall ≥
0.60。必须有五个唯一 outer families；单折不得声称成功。

认证停止条件保持与父 Early 完全相同：任一完成 family F1 严格低于 0.50；已有两个完成 family 的
F1 严格低于 0.65；或把所有剩余 family 的每项 metric 都设为 1 后仍不可能达到任一五-family macro
门。停止也必须保存不可变 certificate，不能删除失败、取消、超时或 superseded 输出。

## 输出与当前结论

预定根目录是
`/ibex/user/zhanx0o/pathline-template-matching/Verify_ClassConditionalTemplateScore_1.1`，所有 Slurm
process 必须立即登记，且只能从已 push 的 clean Git commit 在 Ibex checkout 运行。

截至本合同冻结时，本版本没有真实运行、没有数值结果，也没有“class-conditional template score
有效或无效”的结论。Tangaroa 与 SmokeBuoyancy 的 raw、portable、cache、feature、label、prediction
和 metric 全部禁止访问；即使本 exposed-development 实验以后通过，也不能称为 formal confirmation。

## 配置身份复核

本次运行前审查后的最终配置 SHA-256：
`814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea`。

## 2026-09-01 实现与首次真实读取前审计

冻结配置没有改动。本节只记录配置冻结后的实现、合成验证和部署门；截至本节写入时，本版本仍未打开
任何真实 feature、label、valid-rate、prediction 或 metric。

| 组件 | 路径 | SHA-256 |
|---|---|---|
| class-conditional 数值核心 | `src/pathline_template_matching/class_conditional_template_score.py` | `9c009376f7cea1481f6f47a49362d54d0e78530717f480fda3e8a109f841ef99` |
| fold runner | `scripts/run_verify_class_conditional_template_score_1_1.py` | `e5063887475029320e66da1f1eb221d7988598e8918d37fbe47ee213e5ff1b48` |
| fresh-replay aggregator | `scripts/aggregate_verify_class_conditional_template_score_1_1.py` | `49c80993c9704a46f7aa8aa4dd4a0ed7d08599b02edc50d12f3354e036f924cc` |
| mandatory resource smoke | `scripts/run_verify_class_conditional_template_score_resource_smoke_1_1.py` | `dc42f9ee07f85470034cf543a204461edd2cf98328c6af0b58dd17994599c106` |

提交前最终快照通过 81/81 项 ClassConditional 定向测试、427/427 项项目统一测试、全部相关 Python
文件编译、六个 Bash wrapper 的 `bash -n` 和 `git diff --check`。其中 aggregator 的 12 项测试包含
fresh numerical replay、outer-label gate、严格 JSON 类型、canonical CSV、certificate/report/manifest/
completion 全重构，以及自洽改写发布文件仍必须拒绝的回归。

实现审查发现并在任何本版本真实数组读取前修正了以下问题。它们都是冻结公式的忠实实现或认证/
部署修复，不是新的科学机制：

| 原实现草案 | 当前实现 | 修正原因 |
|---|---|---|
| 没有 pooled negative scaler 的 scale 仍可能把 positive rows 放进其他尺度 calibration prior | 该 scale 的正负 class rows 都从可变换 library/calibration 中排除 | 配置已规定没有本地 negative scaler 时 exact-scale transform 与 retrieval 不支持 |
| 任一 fit family 缺少一个 class 就让整个 fit 失败 | 保留自然不平衡 population；缺失 class 只让该 family 在相应 query 上不能进入 joint-support 集合 | 冻结 2/3、3/4 门定义的是共同支持，不要求每个 family/scale 人为补齐两类 |
| 反序列化可能接受没有 scaler 支持的 class rows | artifact reconstruction 对这种 row 直接失败 | 防止被改写 artifact 绕过拟合期的 support 规则 |
| 发布认证可先信任相互一致但可重写的 outer summary、artifact map 和 stop boolean | public release 重新绑定冻结 Early 证据并调用完整 `_authenticate_fold`；F1、support、13个 artifacts、CSV 与停止证书都由 fresh fold 重构 | wrapper 当场计算 completion SHA，不是独立信任锚；只有 fresh replay 才能证明 source fold 未被改写 |
| JSON 的普通相等允许 `true==1`、`31==31.0` | method binding、candidate、evidence、support、summary、release 全部递归比较精确 JSON 类型和字段集合 | 防止数值别名掩盖 schema 漂移 |
| aggregate 临时文件可能落到共享 `/tmp`；新 clone 没有 Slurm 日志目录 | 每个 job 导出 job-local `TMPDIR/TMP/TEMP` 并验证 Python `tempfile`；Git 跟踪 `slurm_logs/.gitkeep` | 避免大 NPZ 暂存和 `#SBATCH -o/-e` 在脚本正文执行前失败 |
| 可选的 `SLURM_JOB_CONSTRAINTS` 被当作 Rome 证据 | 只接受 `scontrol show job -o` 的 authoritative `Features=rome`，missing、非 Rome 和 `zen3|rome` 均拒绝 | 普通 batch 环境不保证提供该变量，且 OR 表达式不能证明实际冻结请求 |
| remaining folds 只传递依赖第一折认证 | 每个 remaining task 还直接重新认证 resource-smoke PASS | 防止调度依赖被错误配置时绕过资源门 |

Resource smoke 固定只打开 `f22_raptor/channel/boeing_747` 的 12 个 source shards；它覆盖最宽 165D
表示、`k=31`、三个 fit families 的全部 natural scales 和完整 library/calibration/query 路径，但不是最多
28 shards 的 production final-fit 人口上界。因此 smoke PASS 只能支持代码路径和本次 128 GB/4 h 请求，
不能单独证明完整 fold 的峰值资源上界；真实 fold 仍按冻结的 128 GB/12 h 请求并保留实际 MaxRSS/elapsed。

本节所在 implementation/numerical deployment commit 已固定为
`cfa369dd35ab1b3dd89232b74ead7f3b3c937b40`。本次只改文档的后续提交不属于 numerical checkout；
首次真实读取必须在 Ibex detached checkout 该精确 commit，并逐文件复核上述 SHA。任何 SHA、
`slurm_logs/.gitkeep` tree identity 或 clean-worktree 门不通过都禁止提交 resource smoke。
