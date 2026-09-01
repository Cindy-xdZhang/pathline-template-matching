# `Verify_SourceCenteredRankLikelihoodTemplate_1.1`

状态：**方法与候选网格已冻结，尚未产生本版本任何真实 prediction 或 metric**。

唯一数值合同为
`config/Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml`，冻结 SHA-256 为
`41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa`。任何本版本真实
prediction 或 metric 可见后，均不得修改 rank 公式、模板总体、直方图、校准、空间处理、候选网格、
拆分、标签门或成功条件。

## 研究问题与已有证据

直接父版本 `Verify_SourceCenteredPairedScaleTemplate_1.1` 的完整五族认证结果是 macro
F1=`0.6793896154555039`。同一父版本中，不使用模板库的 `dx-rank-mean top-5%` 诊断达到
macro F1=`0.8582408452`。这两个数字来自父版本 exact numerical commit
`a85c007ef961ce53bb40946ca3f38f033bf7a646` 与 aggregate job `51160422`，不是本版本的新结果。

两者的差距说明：source-centered curl 本身已有较强的单调排序信息，但父方法把它与38–164个其他坐标
共同标准化后做对称 Euclidean negative-template distance，可能稀释这个一维信号；同时 IVD-p95 正类是
centered curl 的高尾事件，对称“离负模板远”分数并不区分高于负类还是低于负类。本版本因此同时改变
**表示**和**模板分数**，而不是继续调父分数的 top fraction。

本版本的问题严格限定为：按 source 内尺度组转换成经验 rank 后，以有标签正、负中心模板估计的
class likelihood ratio，能否把上述直接排序上限转化为合法的 family-held-out 模板分类，并稳定超过父方法。

## 复用旧 sidecar，但先建立新版本绑定

本版本不重新采样速度，也不改写旧 sidecar。它只复用父版本已经独立认证的32个 source-centered
sidecar：

| 父证据 | 固定身份 |
|---|---|
| 父 config | SHA-256 `15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc` |
| 父 numerical commit | `a85c007ef961ce53bb40946ca3f38f033bf7a646` |
| input manifest | `/ibex/user/zhanx0o/pathline-template-matching/Verify_SourceCenteredPairedScaleTemplate_1.1/preparation/slurm_51158660_a85c007ef961/source_centered_input_manifest.json`；SHA-256 `5f7e567a2f989d18b51389814938a5d18025c4ed5247730d07df30b13458fec9` |
| sidecar population | `/ibex/user/zhanx0o/pathline-template-matching/Verify_SourceCenteredPairedScaleTemplate_1.1/source_centered_cache/train/SIDECAR_POPULATION.json`；SHA-256 `50d9d53f7dc9255d5153f0101c922975e006303b550bfb43317074080a0a97e2` |
| population规模 | 32 sidecars；4,096,000 assigned rows；2,967,612 parent-valid projection rows |

任何新方法数组读取前，独立 binding job 必须重新认证父 experiment/config/commit、两个 manifest 的
path/size/file SHA-256/self-hash、32个population entries以及每个sidecar的path/size/whole-file
SHA-256，再原子写入本版本的 `parent_sidecar_binding.json` 和 `BINDING_COMPLETE.json`。该步骤禁止
解压任何sidecar member，也禁止读取label、reference或IVD。旧manifest保持逐字节不变；新binding是
consumer证据，不是对父证据的覆盖。

每个outer fold选定final model之前，outer sidecar仍只允许做stat、size和whole-file SHA-256检查。
只有nonouter inner selection、final model、calibrator和两个selected candidates全部关闭并fresh replay后，
才允许打开outer sidecar成员并计算rank。

## 一维 source-centered rank 表示

唯一数值输入是 `source_centered_seed4[:,0]`，即
`||curl-mean_group(curl)||₂`。FMT coordinates、Raw pathline coordinates、strain、divergence与Q都不进入
本版本的分数。

每个 `dataset × source ordinal × scale block × dx level` 必须恰有6,400条assigned rows，来源为
`10 ds × 10 arc × 64 centers`。rank总体包括pathline-invalid rows；validity与label不得参与rank归一化。
对并列值使用稳定mergesort和平均秩。若组大小为`N=6400`，则

```text
r = (one-based average tie rank - 0.5) / N
```

因此`0<r<1`。每个40³中心都有legacy和expanded两个rank。先按候选权重融合：

```text
z_w = w * r_legacy + (1 - w) * r_expanded
w in {0, 0.25, 0.5, 0.75, 1}
```

即使某一block的pathline invalid，它的seed-time assigned rank仍可用于另一个block有效的combined-valid
中心。模板库和主query总体只收`legacy valid OR expanded valid`的中心；两边都invalid的中心固定判负。
同一中心只作为一个模板，禁止因两个valid block而重复加权。其label只由该中心的`valid_labels`聚合；
若两个block均valid，其label必须一致。禁止为建库读取invalid center的`reference_labels_all`。

这里的rank由目标source自身全部assigned rows决定，所以完整分类器是
**target-unlabeled transductive classifier**，不是独立逐primitive descriptor。

## 主方法：双类、family等权的直方图似然比

Log-likelihood ratio（LLR，对数似然比）在本版本中定义为“该rank落入正类模板分布的对数概率减去
落入负类模板分布的对数概率”；分数越大越像正类。`dual`只表示同时使用正、负两类模板，**不是**
分别对两个scale block计算LLR；两个block已先由`w`融合为`z_w`。

对每个fit physical family `f`、class `c∈{0,1}`和等宽bin `b`，使用全部自然combined-valid中心计数
`n_fcb`。不抽样、不平衡类别。候选bin数`B∈{64,128,256}`，边界严格为`j/B`，bin index为
`min(B-1,floor(B*z_w))`。使用`beta∈{0.5,2.0}`做加性平滑：

```text
p_f,c(b) = (n_f,c,b + beta) / (n_f,c + B * beta)
p_c(b)   = mean over fit families of p_f,c(b)
ell(b)   = log p_1(b) - log p_0(b)
```

family等权发生在class probability层：先分别归一化每个family/class histogram，再平均
`p_f,c`，最后计算一个LLR。禁止先计算各family LLR再平均，也禁止按family行数加权。每个fit family
必须同时存在正、负combined-valid模板，否则该fit失败，不能静默丢family。

## 按完整 source 留一的负类校准

为了避免同一source的空间相关中心给自身定义“正常负类”分布，主方法使用完整source留一校准。对每个
fit-negative中心：

1. 排除与它相同 `dataset × source ordinal × complete H48 window` 的全部正、负模板中心；
2. 用未改变的per-source ranks重新构建受影响的family/class histograms、family等权class probability
   和LLR；rank本身不因留一而重算；
3. 计算该negative中心的留一LLR，并按它所属physical family加入reference集合`R_f`。

Outer query使用完整nonouter fit histograms，不排除任何fit source，得到`ell_query`。Empirical
cumulative distribution function（ECDF，经验累积分布函数）把它相对每个fit family的留一负类reference
转换为单侧异常分数：

```text
a_f(ell_query) = count(r in R_f where r < ell_query) / (|R_f| + 1)
a(ell_query)   = mean over fit families of a_f(ell_query)
```

严格`<`使相等reference更保守；所有fit-family reference集合都必须非空。这里不做shrinkage或family
row-count weighting。

## 空间处理与严格阈值

在每个 `dataset × source ordinal` 的40³网格上，对combined-valid mask独立计算：

```text
G(a * mask) / G(mask)
```

`sigma∈{0,0.5,1.0}`，Gaussian truncate固定3.0；`sigma=0`保持原分数。禁止跨dataset、source、fold
传播。预测阈值固定为
`tau∈{0.900,0.925,0.950,0.975,0.990,0.995}`，仅当`score > tau`时判正；相等判负。
threshold不得根据outer prevalence、label或结果修改。

## 两个对照及候选数量

Negative ECDF control只使用每个fit family的自然negative combined-valid `z_w`作为template reference；
query分数为各family `count(reference < z_w)/(N+1)`的等权平均。它不使用positive templates、bin或beta，
但使用相同空间sigma和strict tau。它独立做inner selection，只回答正类histogram与LLR是否必要，不能改变
主候选，也不能满足主成功条件。

Direct rank diagnostic固定`w=0.5`，在每个source的全部64,000中心上选rank最高5%，并按center index
稳定破并列。它不使用模板库、fit label或outer label生成prediction；最终只把prediction投影到全部
parent-valid rows后作可比诊断。64,000中心总体的指标只能在最终label阶段另报。无论数值多高，它都不能
满足模板方法成功条件。

| arm | 冻结轴 | 候选数 | 角色 |
|---|---|---:|---|
| `dual_histogram_llr` | `5 w × 3 B × 2 beta × 3 sigma × 6 tau` | 540 | primary；唯一可满足success的arm |
| `negative_ecdf` | `5 w × 3 sigma × 6 tau` | 90 | 独立inner-selected control |
| `direct_rank_mean_top5` | `w=0.5, top=0.05` | 1个固定诊断 | 非模板上限诊断 |
| 父 `SourceCenteredPairedScale` | 已认证固定prediction | 不选择 | exact-row control与bootstrap parent |

Primary和negative control分别按inner-family macro F1选择；tie-break依次为Average Precision、balanced
accuracy、precision、recall和lexicographically smallest candidate ID。Control不得进入primary的540项
排序，outer结果不得选择任何candidate。

## Nested physical-family 拆分与标签门

五个outer family顺序固定为half-cylinder、delta-wing、F22、channel、Boeing 747。每折留出一个完整
family，其余四个轮流作inner query，剩余三个family拟合rank template。source及其完整H48 future window
不可拆；禁止随机空间seed拆分。Tangaroa和SmokeBuoyancy的raw、portable、cache、sidecar、feature、label、
prediction与metric全部禁止。

Inner选择在每个inner family内先等权平均`dataset × source`的全部parent-valid row指标，再对inner
families等权。Final模型只用四个nonouter families重拟合。每折必须按以下顺序执行：

1. 认证全局parent-sidecar binding，只打开nonouter sidecar与nonouter fit/inner labels；
2. 写闭并fresh replay `inner_group_metrics.csv`、`inner_candidate_summary.csv`、fit audit、final histogram
   model、LOO-negative calibrator、negative-ECDF control和同时含primary/control的selected candidate；
3. 此时才打开outer sidecar成员，使用全部assigned rows构造outer rank，并写闭
   `outer_rank_binding.json`；
4. 生成不含label的outer prediction，关闭manifest后从final artifacts和outer sidecar fresh replay；
5. 新prediction关闭后可重放父prediction，但仍不得打开reference；
6. 最后只打开outer `valid_labels`，按combined-valid center聚合并投影回全部parent-valid rows，写metric与
   reference-access audit。`reference_labels_all`始终禁止。

每折固定18个发布文件；完整列表在config的`outer_label_gate.required_fold_files`中。任何缺失、额外、
覆盖、hash/self-hash错误、outer提前member-open、label提前访问、center/block identity不一致或父control
row join不一致都必须失败并保留partial目录。

## 评价、父方法比较与成功条件

主分类总体为center prediction回填后的全部parent-valid rows，与父方法按
`dataset, source, block, center, scale, assigned-row`精确连接。Combined-valid unique-center指标另报；
全部64,000 centers只作coverage分母。必须报告Accuracy、Average Precision、F1、balanced accuracy、
Area Under the Receiver Operating Characteristic Curve、precision、recall和混淆计数。

不确定性固定为以`dataset × source`为配对单位的5,000次bootstrap，seed=`17068`，目标是
`dual_histogram_llr F1 - authenticated SourceCenteredPairedScale F1`在完全相同valid-row identities上的差。

完整五折primary必须同时满足：

- macro F1≥0.70；F1≥0.75是stretch目标；
- 至少4/5 family F1≥0.65，且任一family F1≥0.50；
- macro Average Precision≥0.60、balanced accuracy≥0.70、precision≥0.60、recall≥0.60；
- combined-valid unique-center coverage≥0.90；
- 相对父方法F1差的paired bootstrap 95%区间下界严格大于0。

任一已认证family primary F1<0.50、已有两个family primary F1<0.65，或把全部剩余family指标设为1仍
无法满足任一完整五折门时，版本必须认证停止。单折、negative control或direct diagnostic不得宣称成功。

## 证据边界

本方法使用目标source的无标签rank statistics和combined-valid空间mask，因此是transductive。它的primary
表示只有seed-time source-centered curl，并没有使用FMT pathline geometry。若本版本提高F1，允许的结论
只有：“有标签的source-rank likelihood template比父版本高维对称距离更好地保存了直接局部curl的单侧
排序信息。”禁止写成“independent FMT geometry学会了涡结构”或“pathline history解释了提升”。

八个flow均已在旧实验暴露；本版本是exposed-development family-held-out验证，不是新physical family上的
formal confirmation。所有失败、取消、超时、无效、负结果和被替代结果必须保留并逐项登记。

## Ibex执行

除本地synthetic/unit tests外，全部真实运行必须由clean、pushed exact Git commit在
`glogin.ibex.kaust.edu.sa`的Rome CPU节点执行，account=`pi-hadwigm`、partition=`batch`、无GPU。
输出根固定为
`/ibex/user/zhanx0o/pathline-template-matching/Verify_SourceCenteredRankLikelihoodTemplate_1.1`，禁止覆盖。
每个binding、fold、authentication或aggregate Slurm job提交后必须立即登记到
`docs/ibex_run_registry.md`。当前尚未提交本版本真实job，因此没有可报告的新候选、prediction、metric或
性能结论。
