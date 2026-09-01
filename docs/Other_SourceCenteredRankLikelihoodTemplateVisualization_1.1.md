# `Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1`

状态：**`FROZEN_PRE_RUN_WAITING_FOR_AUTHENTICATED_RANK_LIKELIHOOD_RELEASE`**。

本版本是`Verify_SourceCenteredRankLikelihoodTemplate_1.1`的固定下游空间报告。它不训练、
不重新拟合、不选择候选、不改阈值，也不依据结果在primary、negative control或direct diagnostic
之间切换。唯一允许绘制的分类arm是`dual_histogram_llr`。冻结配置为
`config/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml`，SHA-256为
`a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3`。

配置是在未读取本方法任何真实fold prediction、label、metric或NPZ成员的条件下冻结的。当前没有
RankLikelihood真实图或指标可报告；numerical commit、aggregate completion SHA和两个fold completion
SHA必须等待上游真实认证运行完成后，由生产wrapper作为显式运行时输入提供，不能预先填造。

## 固定四个场景

四张图全部固定`source_ordinal=2`，不按性能选择：

| 数据集 | 显示名 | outer family | source index |
|---|---|---|---:|
| `cylinder3d` | Cylinder3D Re160 | `half_cylinder` | 68 |
| `halfcylinderRe640` | Cylinder3D Re640 | `half_cylinder` | 18 |
| `halfcylinderRe6400` | Cylinder3D Re6400 | `half_cylinder` | 68 |
| `boeing747` | Boeing 747 | `boeing_747` | 100 |

前三张图必须来自同一个认证的`half_cylinder` outer fold；Boeing图必须来自同一个认证的
`boeing_747` outer fold。两个fold又必须同时被一个`complete_five_fold_aggregate` release绑定。
Partial或single-fold release不能授权本报告。如果上游认证停止规则阻止Boeing fold生成，本版本不能
私自越过停止规则绘制Boeing；应另立并冻结post-stop diagnostic。

## 三联图

每个flow只生成一张三联图，不把legacy和expanded画成两个分类器：

1. Panel A复用`Other_MainExp31FamilyHeldOutVisualization_1.1`固定父scene的whole-volume
   IVD-p95 mesh，并依次画`legacy_2_1`前120条蓝色实线和`expanded_3_1`前120条紫色虚线。
2. Panel B只在`unique_legacy_valid OR unique_expanded_valid`的唯一中心上画
   `unique_primary_prediction`；标题固定为
   `Source-rank likelihood template classification`，禁止误标为FMT。
3. Panel C对完全相同、同顺序的中心画TP、FP、FN、TN。

三个panel必须实际绘制同一个IVD mesh，使用同一相机、bounds和升序center顺序。Panel A的240条
pathline只提供两个尺度块的几何背景，不是分类样本。Primary有效行指标另用
`valid_primary_score/valid_primary_prediction`按完整
`dataset/source/block/center/assigned-row/scale`身份投影并写表，不进入Panel B/C。

## 输入认证与标签边界

方法config SHA固定为
`41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa`。
报告器必须先认证：

- complete-five aggregate的4个精确文件、self-hash、schema、config、fold commit及五family顺序；
- `half_cylinder`与`boeing_747`各自恰好18个fold文件；
- fold result对16个非result/completion artifact的size与SHA绑定；
- selected primary确为冻结540项中的`dual_histogram_llr`，control确为冻结90项中的
  `negative_ecdf`，且selection与prediction关闭时outer label尚未打开；
- aggregate、fold和父sidecar binding证据具有相同config、binding hashes与历史证据；
- 固定父scene的8个`dataset×block`文件及已知parent commit/config/result SHA；
- 冻结报告配置和全部直接报告依赖文件。

上述opaque身份必须先写入自哈希`input_manifest.json`与`figure_contract.json`，之后才允许首次打开
两个`outer_predictions.npz`或八个父scene NPZ成员。报告器不打开fold sidecar、`valid_labels`、
`reference_labels_all`、FMT features或Raw features；TP/FP/FN/TN只使用认证父scene已经冻结的
IVD-p95 reference。

唯一中心必须恰为升序`0..63999`。Dataset、source ordinal/index、block、center、assigned row和
scale ID必须精确连接；duplicate、missing、extra、reorder、valid-mask drift、跨source/family、两个
block重叠中心的坐标/reference不一致，或valid-row score/prediction不等于对应center primary值，均失败。
中心和valid-row指标都要重算，并在绝对容差`1e-12`内复现producer的
`dual_histogram_llr`行及candidate身份。`negative_ecdf`和`direct_rank_mean_top5`的完整valid-row指标
也必须重算并写入表格，但不得进入任何panel或触发arm切换。

## 输出与QA

机器阶段从clean、pushed exact reporting commit在Ibex Rome CPU节点执行：1 node、32 CPU、
128 GiB、12小时、无GPU。生产入口为
`ibex/other_source_centered_rank_likelihood_template_visualization_1.1.sh`。它要求显式提供：

```text
EXPECTED_GIT_COMMIT
RANK_LIKELIHOOD_VIZ_METHOD_COMMIT
RANK_LIKELIHOOD_VIZ_RELEASE_ROOT
RANK_LIKELIHOOD_VIZ_RELEASE_COMPLETE_SHA256
RANK_LIKELIHOOD_VIZ_HALF_FOLD_ROOT
RANK_LIKELIHOOD_VIZ_HALF_RUN_COMPLETE_SHA256
RANK_LIKELIHOOD_VIZ_BOEING_FOLD_ROOT
RANK_LIKELIHOOD_VIZ_BOEING_RUN_COMPLETE_SHA256
RANK_LIKELIHOOD_VIZ_OUTPUT_ROOT
```

每图写combined-scene NPZ与manifest、360 dpi PNG、editable-text PDF/SVG、panel alignment和render
metadata；全局写config副本、input/figure/visualization/result/completion self-hashed manifests和指标CSV。
所有文件先在同一目录完成临时写入和`fsync`，再原子发布；目标已存在时失败且绝不替换。机器状态固定停在
`complete_pending_local_rendered_qa`，此时不可交付。

如果方法fold commit与reporting commit不同，wrapper和reporter还必须逐文件证明13个方法解释源在两个
commit中的Git blob完全相同；这些blob身份写入输入、结果和完成证据，本地QA再独立复算。这样旧fold不会
被后来改动的candidate、schema或dtype代码重新解释。

本地auditor只检查机器bundle、clean exact reporting checkout、auditor自哈希、1.5 pt严格panel位置、
SVG真实`<text>`、PDF最小5 pt、collision hard failure为0，以及四张7560×1800 PNG的21×5 inch
人工复核。它不含`np.load`，不会重新打开任何预测、label或scene数组。全部通过后才另写不可覆盖的
`delivery_qa_summary.json: delivery_status=PASS`，原机器completion保持不变。

## 结论边界

RankLikelihood primary只使用source-centered seed-time curl rank，不使用FMT pathline geometry或Raw
pathline coordinates；Panel A pathline不能解释Panel B分类分数。本方法还使用目标source自身的无标签
rank总体和valid mask，因此是transductive。四个flow都属于已暴露development数据，单source图没有
置信区间，不能称formal confirmation、聚类、independent per-primitive classifier，或替代完整五family
统计。
