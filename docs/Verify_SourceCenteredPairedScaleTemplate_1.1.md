# `Verify_SourceCenteredPairedScaleTemplate_1.1`

状态：**方法已冻结，尚未读取本版本任何真实 source-centered feature、prediction 或 metric**。

## 研究问题

已认证的 `Verify_EarlyOppositePairKinematics_1.1` 在五个完整 physical family 上得到
macro F1=`0.6391632766`。它比 PerScale 父版本提高约0.101，但仍未达到0.70。
本版本检验两个由第一性原理得到、且必须同时区分的缺陷：

1. Early 追加的是原始 `||curl||`，而 IVD 标签和 FMT Task1 的旧 `kin2/kin4`
   使用的是 `||curl-mean(curl)||`；
2. 3.1 在同一个40×40×40中心网格上生成 legacy 与 expanded 两条尺度证据，旧方法却把
   两个 block 当成互不相干的分类样本。

完整数值合同唯一存于
`config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml`，冻结SHA-256为
`15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc`。任何真实结果可见后都不得
修改均值分组、表示、候选、融合权重、top-fraction网格或成功门。

## 为什么 Task1 不是原先理解的“纯 FMT、无训练二分”

FMT Task1 的 encoder 没有可训练参数，但完整 evaluator 会在目标 dataset 自己的开发时间片上
重新拟合 StandardScaler、可选 Principal Component Analysis（PCA，主成分分析）和
KMeans（二分 K 均值聚类）。目标 family 标签还参与 feature/PCA 选择，并在另外两个开发时间片上
决定 cluster 0/1 哪一个表示涡。其高分 `kin2/kin4` 会在每个时间片把涡量减去该批 primitive
的平均涡量。因此 Task1 的数字不是固定模板库在未见 physical family 上的可比基线。

在相同五-family等权口径下，Task1+KMeans F1=`0.541887`，低于 Early 的`0.639163`。
Task1的Re160/Re640/Re6400/Boeing分别约为`0.596/0.569/0.533/0.841`，也没有证明所有
流场达到0.7–0.8。本版本仍保留完整 family-held-out 规则。

## 无标签 source-centered sidecar

每个 `dataset × source ordinal` 有128,000条 assigned rows：同一64,000中心网格分别进入
legacy和expanded block。对每条assigned row，在portable frame 0同步采样
`center,x+,x-,y+,y-,z+,z-`七点速度并做中央差分，得到速度梯度和三分量curl。

均值分组严格为：

```text
dataset × source ordinal × scale block × dx level
```

每组必须恰有`10 ds × 10 arc × 64 centers = 6400`条assigned rows。均值使用全部assigned
rows，不能按pathline validity过滤。这样每个dx level都在同一均匀空间样本上估计自身的数值均值，
不会让长弧失败率或valid coverage隐含进入均值。

新4D block为：

```text
[ ||curl_i - mean_group(curl)||_2,
  ||strain_i||_F,
  signed divergence_i,
  0.25*||curl_i - mean_group(curl)||_2^2 - 0.5*||strain_i||_F^2 ]
```

sidecar生产路径不得打开`valid_labels`、`reference_labels_all`、`ivd_values_all`、
`ivd_volume`、`valid_mask`或metadata。这里的均值来自内部assigned center grid，并不是读取标签
构造时的whole-volume mean。由于outer flow自身的无标签速度用于估计均值，完整方法必须称为
**target-unlabeled transductive classifier**；不能再称每个primitive完全独立。

## 模板分数与尺度融合

三个表示只把Early旧4D替换为新4D：

| 表示 | 宽度 |
|---|---:|
| `fmt161_plus_source_centered_seed4` | 165 |
| `real_neighbor36_plus_source_centered_seed4` | 40 |
| `chirality_all35_plus_source_centered_seed4` | 39 |

负模板库、fit-only逐尺度scaler、exact-same-scale k-nearest-neighbour、负模板leave-one-out
tail anomaly以及40³ mask-normalized Gaussian空间处理都继承Early已验证实现。IVD p95是在
whole-volume体素总体上定义，不能推出40³内部中心、combined-valid centers或回填valid rows的
正类率必为5%。因此决策率冻结为`{0.025,0.04,0.05,0.06,0.075,0.10}`，只能由inner
physical families选择；禁止读取outer prevalence或标签，也不扫描0.50–0.99分数阈值。

每个中心先得到legacy与expanded的空间分数。两个都可用时按
`w*legacy+(1-w)*expanded`融合；只有一个可用就使用该分数；都不可用则分数0、判负。
`w∈{0,0.25,0.50,0.75,1.00}`只允许由inner families选择。该对称网格不预设
legacy一定优于expanded；`w=1`时expanded-only centers仍使用expanded分数。候选总数为
`3 representations × 4 k × 5 sigma × 5 weights × 6 top fractions = 1800`。

预测单位是每个source的64,000个unique centers，再把同一prediction回填到相应valid block rows，
同时报告unique-center、valid-row、legacy、expanded、both-valid、legacy-only、expanded-only和neither-valid。
为保持与Early F1=`0.639163`完全可比，成功门中的F1/AP等分类指标使用回填后的全部parent
valid rows；inner候选选择也使用每个`dataset×source`的全部parent-valid rows，并在inner family内
先等权聚合这些组。parent control与新方法必须按dataset、source、block、center、scale和assigned-row精确连接。
Unique-center分类指标只在至少一个block有效的center上另报，neither-valid不伪装成预测负类；全部
64,000 centers仍用于combined coverage分母。

另冻结两个不参与候选选择的直接运动学诊断：其一在每个center的两条assigned证据中选physical
`dx`较小者的centered-curl norm；其二先在每个`block×dx`的6400 rows内做经验midrank，再平均
同一center的legacy/expanded rank。二者都固定取top 5%，都不读label，但也都不使用模板库。
它们只回答“新表示本身有没有足够排序信息”：即使F1超过0.70，也不能替代主模板方法的成功规则。

## 拆分、标签门与成功条件

五个outer family固定为half-cylinder、delta-wing、F22、channel和Boeing 747。每折留出一个完整
family，其余四个轮流作inner；source与完整H48 future window不可拆。Tangaroa和SmokeBuoyancy的
raw、portable、cache、feature、label与metric全部禁止。

每折顺序必须是：

1. nonouter scaler、负模板库、calibrator和selected candidate写闭并认证；
2. 才打开预先密封的outer无标签sidecar，并写入、重放fold-local source-mean绑定artifact；
3. 写闭unique-center及row projection prediction并fresh replay；
4. 最后才打开outer reference评测。

32个全局无标签sidecar可以在fold之前一次性生产和seal，和Early旧sidecar population相同；但每个
fold在final nonouter model与candidate写闭前不得解压或读取outer sidecar的任何NPZ数组、metadata或
group mean。启动时允许的outer访问只有文件stat、size和whole-file SHA-256，用于认证密封population，
不产生任何数值特征。之后还必须
把所用outer mean复制/绑定到fold-local artifact并认证，再生成prediction。

完整五折主方法必须同时满足：macro F1≥0.70、至少4/5 family F1≥0.65、任一family
F1≥0.50、AP≥0.60、balanced accuracy≥0.70、precision和recall均≥0.60、unique-center
combined coverage≥0.90，并且相对Early的paired source bootstrap F1差值95%区间下界>0。
F1≥0.75是stretch目标，不把0.80事后改成硬门。

## 证据边界

若本版本提高F1，结论只能是“source-centered局部运动学和同中心多尺度融合改善了负模板异常分数”。
因为新block直接利用curl deviation，不能写成“独立FMT几何已经学会涡旋”。若分数仍低于0.70，
就说明缺失的Task1 batch statistic不是唯一瓶颈，下一步才需要验证同步时间采样的flow-map deformation，
而不是继续调同一分数的阈值。
