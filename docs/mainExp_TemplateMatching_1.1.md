# mainExp_TemplateMatching_1.1：3D FMT 特征库最近邻基线

状态：**基线方法参数已预注册；cache-backed development 协议已冻结但尚未运行；formal confirmation 仍被禁止，尚无性能结果**。基础方法配置为 `config/mainExp_TemplateMatching_1.1.yaml`，development 运行配置为 `config/mainExp_TemplateMatching_1.1_development.yaml`。

## 研究问题

当 query 来自建库未出现的 physical flow family 时，逐 primitive 独立的 161 维 training-free FMT 特征，是否比同协议 Raw 对照更适合用 exact one-nearest-neighbor（1NN，一最近邻）识别 whole-loaded-volume IVD p95 涡区域？

## 已预注册的方法参数

- Primitive：3D 7 线、每线 32 点，顺序固定为 `center,x+,x-,y+,y-,z+,z-`；积分器保留 `7×32×4=(x,y,z,t)`，descriptor view 固定为前三通道 `7×32×3=(x,y,z)`；rounded integration index 重采样。
- Descriptor：`fmt_independent_3d_161d_sha256_25fce29499c9089e`；6 frequencies、Gram、chirality、sorted-neighbor slots、`neighbor_weight=1`、`neighbor_scale=1`。该数值约定与旧 Task5 cache 一致。
- Library：每个 `flow × source time × scale` 取 `m=min(512,n_positive,n_negative)` 个正类和负类；空类别使该 stratum 失败。候选按 source ordinal、seed index 稳定排序，采样 seed 为 `15068`。
- Normalization：逐 feature mean/std，只在当前 fold 的 library 上拟合。
- Matcher：exact Euclidean 1NN。Query 必须声明相同 descriptor ID，否则代码拒绝检索。
- Score：`nearest-negative distance − nearest-positive distance`；score `>0` 判涡，完全等距固定判非涡。
- 1.1 不设 reject threshold，不搜索 k，不做 distance weighting。

## 开发拆分与尺度角色

旧 FMT Task5 的 10 个数据条目及两套历史 phase 已全部暴露，在本项目中都只是 development evidence。每折留出一个完整 physical family 做 query，其余 family 建库。

- `library` 的 18 个 tuple：旧 development cache 的前 4 个 source times。非留出 family 建库；留出 family 用相同 tuple 报告“未见 family、已见尺度”。
- `descriptor_selection_only` 的 6 个 tuple：旧 development cache 的后 2 个 source times，只能用于 `Verify_...`，不进入本主实验指标。
- `unseen_scale_evaluation` 的 9 个 tuple：旧目录名为 confirmation 的 4 个 source times；在本项目中只报告“未见 family、未见尺度”的 development 结果。

每折 query 使用留出 family 在对应 evaluation scale set 的全部 valid primitives，保持自然类别比例，不按标签平衡或下采样。必须报告 assigned、valid、invalid 和自然正负类数量；若以后限量，必须采用与标签独立的冻结规则并升级版本。

正式 confirmation 必须使用未被 FMT 或本项目读取的新 physical families。先登记 ID、文件 SHA-256、冻结 commit 和 manifest，再首次读取 raw field；冻结前不得查看 query feature、有效率、标签或指标。

## 主比较

1. 常数 prior：从平衡抽样前的 eligible library candidate 标签比例估计；不得使用 query 标签。score 为该常数，比例大于 0.5 才判涡，等于 0.5 判非涡。
2. 672 维 centered Raw：`(xyz - center_seed_xyz)`，按 `line,time,xyz` 的 C-order flatten；library-only standardization + exact 1NN。
3. 161 维 Raw Principal Component Analysis（PCA，主成分分析）：PCA mean/components 只由当前 fold library 的 672 维 centered Raw 用确定性 full Singular Value Decomposition（SVD，奇异值分解）拟合；query 只 transform；再做 library-only standardization 和 exact 1NN。
4. 161 维 FMT：冻结 descriptor、library-only standardization 和 exact 1NN。

Raw-PCA 控制维数效应；若没有它，最多只能写“161D FMT 优于当前 672D Raw baseline”，不能归因于 FMT 几何描述。

## 指标与结论边界

主要指标为 Average Precision 和 F1；另报 Area Under the Receiver Operating Characteristic Curve、precision、recall、balanced accuracy。必须分开报告 seen-scale 与 unseen-scale 的逐 flow、dataset macro、physical-family macro、逐 tuple，并以 source timeslice 为配对 bootstrap 单位给出 95% confidence interval。

本 development 运行把 `dataset macro` 固定为10个逐-flow pooled-query 指标的算术平均；`physical-family macro` 固定为先在每个 family 内对 source-timeslice 指标做宏平均、再对7个 family 做宏平均。成对 bootstrap 重采样同一个 source-timeslice block，因此其点估计和置信区间与主表的 physical-family macro 是同一个估计量；另在 `per_family.csv` 保留 pooled-query family 指标，二者不得混写。

只有 development 与 sealed confirmation 的 family-macro F1、Average Precision 都高于两项 Raw 对照，才能描述点估计改善。是否还要求 paired 95% confidence interval 的差值下界大于 0，尚待用户在首次 sealed-confirmation performance job 前决定；决定前 development 结果只能作描述性报告。任一 flow 或 scale 反例必须列出。

结论不得推广到连续任意尺度、不同重采样、2D primitive、六邻居独立距离或不同标签。

## Cache-backed development 运行冻结

本版本的 development phase 只读取已暴露旧 Task5 cache，不从 raw field 重算 primitive，也不重算或替换主指标标签。三联图可从 raw field 重建 IVD-p95，但只用于 fail-closed 一致性审计和等值面绘制：

- 旧 development ordinals `0–3`：非留出 family 建库，留出 family 作 seen-scale query；
- 旧 development ordinals `4–5`：禁止进入主指标；
- 旧 historical confirmation ordinals `0–3`：只作 exposed-development unseen-scale query；
- 主指标标签来源固定为 cache `reference`；raw IVD 审计不替换该标签；
- 七个 physical family leave-one-out；四臂固定为 constant prior、Raw672 exhaustive 1NN、library-only Raw-PCA161 exhaustive 1NN、FMT161 exhaustive 1NN；
- bootstrap 固定 seed `25068`、5000次，按 physical family 分层，并以 source timeslice 成对重采样；95% 区间使用 percentile interval，NumPy percentile method 固定为 `linear`；
- 结果状态固定为 `descriptive_only_pending_user_ci_decision`；
- sealed confirmation access 固定为 `forbidden`；
- 输出根目录固定为 `outputs/mainExp_TemplateMatching_1.1_development`。

Development job 只允许在 legacy-cache validator/input manifest、development leave-one-family-out evaluator、Raw-PCA、审计计数、bootstrap 和绘图流程实现并测试后提交。Seed-time IVD 插值不是本 cache-backed phase 的前置条件，因为主指标标签不重算；它仍是 raw-field 重新建库和 formal confirmation 的前置条件。

## Development 表格与三联图

机器结果必须至少输出 input manifest、audit counts、逐 query/timeslice/flow/family/scale 表、bootstrap difference 和主表；seen-scale 与 unseen-scale 必须分开。当前没有任何结果文件或数值。

每个数据集分别生成 seen-scale 与 unseen-scale 图，固定 source ordinal `2` 并覆盖全部 scale tuple，不按任何指标选图。三栏固定为：

1. IVD-p95 reference + cached center pathlines；raw field 可访问时叠加 whole-loaded-volume IVD-p95 isosurface，否则明确写 `seed-reference fallback`；
2. FMT exact-1NN vortex/non-vortex class assignment，禁止写 KMeans；
3. FMT TP/FP/FN/TN，TN 可淡化。

每张图使用 deterministic stratified maximin、seed `15068` 选择240条中心线，三栏共享相同 seed IDs、camera 和 physical bounds。Raw-PCA 必须进入表格，可另做独立比较图，但不进入该三联图。

## Formal confirmation 前缺口

已完成：精简 3D RK4 integrator、7-line 构造、rounded-index 重采样及解析流测试；development 部署协议已经冻结但尚未执行。

仍必须完成并单独验证：

1. seed-time IVD 插值与面向新 raw field 的正式端到端 builder；
2. sealed confirmation family 清单、文件 hash、冻结 commit、manifest 与 first-read gate；
3. 用户在首次 sealed performance job 前决定置信区间结论规则。

这些缺口完成前，config 状态不得改为 full frozen，也不得提交 formal confirmation job。Cache-backed development job 完成后仍只能登记为 development evidence，不能把整个 `mainExp_TemplateMatching_1.1` 标成已完成。
