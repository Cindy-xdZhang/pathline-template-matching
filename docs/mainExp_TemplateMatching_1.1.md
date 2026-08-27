# mainExp_TemplateMatching_1.1：3D FMT 特征库最近邻基线

状态：**基线方法参数已预注册；完整主实验尚未冻结，尚无性能结果**。配置为 `config/mainExp_TemplateMatching_1.1.yaml`。

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

只有 development 与 sealed confirmation 的 family-macro F1、Average Precision 都高于两项 Raw 对照，才能描述点估计改善。是否还要求 paired 95% confidence interval 的差值下界大于 0，尚待用户在首个性能 job 前决定；决定前结果只能作描述性报告。任一 flow 或 scale 反例必须列出。

结论不得推广到连续任意尺度、不同重采样、2D primitive、六邻居独立距离或不同标签。

## 完整冻结前缺口

已完成：精简 3D RK4 integrator、7-line 构造、rounded-index 重采样及解析流测试。

仍必须完成并单独验证：

1. seed-time IVD 插值、正式 library builder、cache schema 和 manifest；
2. leave-one-family-out evaluator、Raw-PCA 与 timeslice bootstrap；
3. `scale × class` assigned/valid/selected/invalid 审计；
4. sealed confirmation family 清单与 first-read gate；
5. 用户决定置信区间结论规则。

这些缺口完成前，config 状态不得改为 full frozen，也不得提交性能主实验。
