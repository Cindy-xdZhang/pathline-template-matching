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

代码级复核还得到三个容易被“encoder无训练参数”掩盖的事实：

- Task1的canonical pipeline仍会拟合目标dataset自己的StandardScaler、可选PCA和KMeans；
  目标family标签选择feature/PCA，另外两个目标时间片的标签决定cluster 0/1语义。
- half-cylinder正式配置使用`fmt_all+kin4/PCA8`，Boeing使用`kin2/PCA2`；`kin4`
  在每个cache record内部显式计算`vorticity-vorticity.mean(dim=0)`，不是独立模板描述子。
- 历史Task1固定`dx=0.5 hmin`、`RK4 dt=0.25 source_dt`及48步积分时间，再抽取32个
  同步时间序列点。当前2000-scale primitive固定空间弧长并让七条线分别做等弧长重采样，
  因而不是同一个输入任务。Task1配置曾写`neighbor_weight=0.5`，但canonical cache和结果
  实际使用1.0；只读重算Re640与归档逐指标一致。

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

### 4.5 七条等弧长线的相同sample index不是共同物理时间

当前每条pathline分别积分到自身目标空间弧长，再分别等弧长重采样。邻线和中心线的第`j`
个点一般对应不同物理时间，但FMT仍逐index相减；cache又只保存中心线的32个时间，无法从现有
Raw672恢复七线同步的flow-map deformation。这个问题对长弧尤其明显。Seed-time七点velocity
sidecar在`t=0`同步，因此Early的`+0.101055`增益同时支持“同步局部导数很重要”；它不能证明
异步的完整pathline-history表示正确。

### 4.6 最新class-conditional分支改变了问题，却没有形成有效类别分数

该分支计算每个family/class内部的kNN upper-tail conformity，再取
`0.5*(1+q_positive-q_negative)`；它既不是最近正负模板类别，也不是后验概率。对两类都典型或
都异常的query都可能接近0.5。认证half-cylinder首折F1=`0.404462`后已按规则停止；Boeing仅作
停止后诊断，F1=`0.241293`。它不是当前最佳方法，不能用来代表整个模板思路。当前完成五折的
最佳结果仍是EarlyOppositePair的F1=`0.639163`。

### 4.7 IVD p95不等于query中心或valid rows固定5%正类

之前把whole-volume体素上定义的IVD p95直接解释成“每个64,000中心query组应预测top 5%”。
这是错误的：内部40³中心网格、至少一个block有效的中心以及center prediction回填后的valid rows
是三个不同population，任何一个都不保证正类比例恰为5%。固定预测数会在排序正确时仍人为限制
precision或recall。当前结论改为：在outer label不可见的前提下，预冻结
`0.025/0.04/0.05/0.06/0.075/0.10`并只由inner physical families选择；成功指标与候选选择
都使用精确parent-valid rows，unique-center指标只作secondary。原top-5%结果仍是历史父版本的
有效定义，但不能再作为p95必然推出的先验。

## 5. 当前冻结的下一轮检验

Early、Raw-PCA和初版Dimensionless已经运行：Early五折F1=`0.639163`；Raw-PCA首折
F1=`0.469416`后停止；Dimensionless在任何metric前发现absolute-float32输入几何与冻结合同
不兼容，57,446个失败后来全部被量化误差解释。因此当前优先级已更新为：

1. `Verify_SourceCenteredPairedScaleTemplate_1.1`：对全部assigned seed rows按
   `source×block×dx`无标签估计mean curl，把Early的raw `||curl||`替换成
   `||curl-mean(curl)||`；然后在同一个40³中心上融合legacy/expanded两条尺度分数。融合
   使用`0/0.25/0.5/0.75/1`完整对称权重，决策率使用上述6值网格，两者都只能由inner
   families选择，不能预设legacy更可靠或读取outer prevalence。
2. 同一版本冻结min-`dx`和逐`dx` midrank双尺度平均两个direct top-5%诊断。若direct高而
   模板分数低，表示已经足够、瓶颈在高维FMT距离或negative-only anomaly语义；若direct也低，
   才说明seed-time局部导数的分辨率不足。
3. 若source-centered仍失败，下一项应重新积分一个小型、七线共同物理时间的deformation
   cache，并做`邻居固定方向/排序池化 × seed/full-history梯度`的2×2分解；不能从现有Raw672
   冒充同步kinematic history。
4. 之后才比较class-balanced正负距离margin、目标flow无标签KMeans或有监督physics-anchored
   metric learning；全部必须保持完整family留出，不能退回随机拆seed制造高分。

直接从 velocity 重新计算完整 IVD 并取 p95 可以作为采样/插值的诊断上界，但它与标签
定义近乎同义，不能冒充模板匹配改进或用于宣称本项目达到目标。

## 6. 当前稳定结论

当前失败不是某个 `k`、threshold 或 diagonal variance 公式选错，而是跨flow表示和
“负类异常=涡”的假设没有普适成立。后续实验应优先改变输入表示或适应机制；继续扩大
同一3060候选网格没有第一性原理依据。
