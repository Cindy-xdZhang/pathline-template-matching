# Pathline template matching：第一性原理诊断

本文记录为什么当前跨物理族模板匹配停在 F1 约0.54，以及后续实验如何直接检验
真正的瓶颈。所有性能数字都来自已完成、可认证的运行；尚未运行的方法不写成结论。

## 1. 先统一比较口径

FMT Task1 3.3 的十个数据条目平均 F1 为 `0.606360`，但这不能直接与当前五个
outer physical families 的等权宏平均比较。把 Task1 限制到当前允许使用的八个 train
flows，并按当前相同的五个完整物理族等权汇总，结果为：

| Physical family | Task1 FMT + KMeans F1 | NegativeTail F1 | PerScale F1 |
|---|---:|---:|---:|
| `half_cylinder` | 0.537455 | 0.537691 | 0.542668 |
| `delta_wing` | 0.764954 | 0.770469 | 0.781242 |
| `f22_raptor` | 0.314110 | 0.513438 | 0.503504 |
| `channel` | 0.252632 | 0.253965 | 0.244366 |
| `boeing_747` | 0.840281 | 0.626794 | 0.618757 |
| **五family等权宏平均** | **0.541887** | **0.540472** | **0.538108** |

Task1 源数据是 FMT 的
`outputs/mainExp_Task1_3D_3.3_reference_{old8,new2}/paper_table.csv`；NegativeTail
与 PerScale 分别由 Ibex `51059491` 与 `51064966` 认证。因此，在相同 family 集合
和相同汇总单位下，当前模板方法不是比 Task1 低很多，而是几乎相同。变化发生在
family 之间：模板方法显著提高 F-22，却损失 Boeing；channel 在三种方法中都约0.25。

这项口径校正不降低项目目标。目标仍是五family宏平均 F1 至少0.70，并且至少4/5
family达到0.65、任何family不低于0.50；当前三种方法都没有达到。

## 2. Task1 与“未见流场模板匹配”不是同一个任务

| Task1 3.3 | 当前模板验证 |
|---|---|
| 在目标 flow 自身的 development features 上拟合 StandardScaler、PCA 和 KMeans | scaler、PCA或模板必须只来自其他完整物理族 |
| 用目标 family 标签选择 FMT block/PCA，并用目标 flow 标签冻结 cluster 语义 | outer family 标签不能选择 representation、距离、`k`、空间尺度或阈值 |
| 单一 pathline 尺度 | 同时覆盖2000个 `dx×RK4 ds×空间弧长` 尺度 |
| 多数 flow 使用依赖当前 batch mean 的旧 kinematic block | 当前 FMT 表示要求每个 primitive 独立、不能用 query label 或 batch label statistics |
| 留出的是同一 flow 的新时间片，完整 future windows 还可能重叠 | 留出完整 physical family 及其全部 source windows |

因此 Task1 证明的是“在已见物理族内，FMT feature 可被二分聚类”，不是“一个跨物理
族的绝对特征空间已经存在”。模板匹配需要后者，困难明显更大。

## 3. 已经被证据排除的解释

1. **不是正模板抽样太少这一项。** 平衡正负 exact-1NN 的 family-held-out F1 很低；
   改成使用全部自然负类的 anomaly score 后提高到约0.54，但仍失败。
2. **不是不同尺度的 score 不能比较这一项。** NegativeTail 用 fit-negative 的逐尺度
   leave-one-out 尾概率校准后，support 为100%，macro F1仍为0.540472。
3. **不是全局 diagonal variance 应改成逐尺度 variance 这一项。** PerScale 只改变该
   权重，完整五折 F1 变为0.538108，反而降低0.002364。
4. **不是简单空间平滑或阈值搜索不足。** 当前每个 inner fold 已完整搜索3060个冻结
   candidates；channel 的 Average Precision 只有0.179057。阈值只能在既有排序上取点，
   不能修复这样的排序失败。
5. **不是尺度检索缺少支持。** PerScale final folds 的 retrieval/calibration support
   都为100%，imputation与unimputable均为0。

## 4. 当前最可能的结构性错误

### 4.1 把“相同 scale ID”误当成“跨流场相同物理表示”

Scale ID 固定的是 `dx/hmin`、`RK4 ds/dt` 与 `arc length/hmin` 的组合，但当前 FMT
coordinate amplitudes仍保留物理 `hmin`、速度与时间单位。不同 flow 的同一个 scale ID
并不保证其中心位移、邻线变形或 Fourier amplitude 同分布。Task1 在目标 flow 内拟合
scaler/PCA，会吸收这种单位与流场偏移；跨family模板库不能。

### 4.2 目标是局部速度梯度统计，表示仍混有大量平流与尺度能量

IVD 标签是 seed time 的
`||curl(v)-spatial_mean(curl(v))||`。当前 independent FMT 主要编码中心线方向变化和
邻线相对运动的 Fourier Gram/chirality；它不是 velocity-gradient estimator，也没有
显式把位移除以目标弧长、把邻线变形除以 physical dx。高维 Euclidean distance因而可
由与 IVD 无关的流场、速度和尺度能量主导。

### 4.3 “离所有负模板远”不是普适的涡定义

Negative-only anomaly 假设涡 primitive 在其他family的非涡分布之外。delta-wing符合
这个假设，channel不符合。只要 held-out flow 的普通剪切、边界层或背景输运本身发生
domain shift，负类距离就会很大；反过来，涡的几何也可能靠近另一family的非涡模板。
尾概率和逐尺度方差都不能改变这个语义错误。

### 4.4 2000个尺度增加了覆盖范围，也增加了 nuisance variation

当前每个 center 在 legacy/expanded block 各分配一个尺度，而不是对同一 center 观察
全部2000尺度并融合。增加尺度并不会自动产生多尺度投票；它主要扩大训练与查询分布。
若 descriptor 不先无量纲化，更多尺度反而扩大跨family分布差。

## 5. 冻结的下一轮检验顺序

1. `Verify_EarlyOppositePairKinematics_1.1`：从同步 seed-time 七点速度直接估计
   curl/strain/divergence/Q，检验“缺少目标相关局部导数”这一原因。
2. `Verify_RawPCANegativeMetric_1.1`：每个 nested fit 独立用无标签 Raw672 拟合 PCA161，
   检验“现有 FMT 压缩丢失判别信息”这一原因。
3. 若两者仍失败，建立 `Verify_DimensionlessDeformationFMT_1.1`：中心轨迹除以实际目标
   弧长，六条邻线相对中心的形变除以实际 physical dx，再进行独立 FMT；检验单位与
   尺度 nuisance。该版本必须在读取 Early/Raw-PCA outer 结果前冻结。
4. 若无量纲化仍失败，使用 query-unlabeled 自适应：在每个新 flow 内拟合二分结构，
   只用训练库决定两个 cluster 的物理语义。这样保留 Task1 的 flow-adaptation 优点，
   但 outer label 仍不能参与拟合、选择或 cluster orientation。
5. 最后才考虑有监督的 physics-anchored metric learning；必须保持完整family留出，
   不能退回随机拆 seed 来制造高分。

直接从 velocity 重新计算完整 IVD 并取 p95 可以作为采样/插值的诊断上界，但它与标签
定义近乎同义，不能冒充模板匹配改进或用于宣称本项目达到目标。

## 6. 当前稳定结论

当前失败不是某个 `k`、threshold 或 diagonal variance 公式选错，而是跨flow表示和
“负类异常=涡”的假设没有普适成立。后续实验应优先改变输入表示或适应机制；继续扩大
同一3060候选网格没有第一性原理依据。
