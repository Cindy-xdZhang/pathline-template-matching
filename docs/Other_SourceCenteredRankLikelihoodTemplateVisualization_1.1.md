# `Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1`

状态：**`COMPLETED_LOCAL_RENDERED_QA_PASS`**。

本版本是`Verify_SourceCenteredRankLikelihoodTemplate_1.1`的固定下游空间报告。它不训练、
不重新拟合、不选择候选、不改阈值，也不依据结果在primary、negative control或direct diagnostic
之间切换。唯一允许绘制的分类arm是`dual_histogram_llr`。冻结配置为
`config/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml`，SHA-256为
`a464761eb8df3ebf43d55b6f05eee2e90302be770b43f3e5e75a5944f13ff9a3`。

配置是在未读取本方法任何真实fold prediction、label、metric或NPZ成员的条件下冻结的。
认证数值commit为`8db286f07da0ad484a595f85be5c4577957e032b`，报告commit为
`1abd0a09adf34bcdf3f993e47e39ec6cefd1e618`。完整五折release、half-cylinder fold与
Boeing fold completion SHA-256分别为
`6a59b22f2bed5a66d382cb71da14aa6753873a46c89d0290fe286972c958ac71`、
`18413f08b1975d47c1b61e6b304bf823a78de9a024303d95f526a4cf13f4abe3`与
`d3c70cd4d3d5fee223a5a1ff4c0f4ffb4f7c3c2a71735eafb7e6657a6cbf1a5e`。

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

## Ibex运行与失败保留

首次job `51165219`在2026-09-02 00:58:58–01:02:03 +03:00运行，完整488/488回归
（2项跳过）后，在打开任何prediction、label或scene NPZ成员以及创建output之前，因
`aggregate source evidence identity changed`关闭失败。原因是首次reporter将尚未绑定的
`load_plan().source_evidence=None`与真实aggregate历史证据直接比较；修复只改了证据封套认证和
回归测试，没有改primary prediction、candidate、threshold或冻结report config。该job没有
output或图，不产生可视化结论；stdout/stderr SHA-256为
`045433f7644d1e84e3a7acb4d040c57ac01e680769dfffefa605696dc50a34e5`/
`24ef9ef3a172f05f368bcf63ede64365128308c31f238d2b7aa459a995187d3d`。

修复后job `51167090`在2026-09-02 01:16:19 +03:00提交，01:16:23开始，01:22:42结束，
状态`COMPLETED 0:0`，elapsed=`00:06:19`。它在`cn514-13-r`使用32 CPU、128 GiB、
Rome节点、无GPU；batch MaxRSS=`2,370,032K`，TotalCPU=`15:55.963`，488/488回归通过
（2项跳过）。stdout/stderr SHA-256为
`7151b0af2d10468dbbb337fded4b80132c6ab810a5449ee83530616a1d432752`/
`a80c61df14b559e46a0d58aa0eabf2ccd90e07837cb9b4751410270782924ff3`。不可变Ibex output为
`/ibex/user/zhanx0o/pathline-template-matching/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1/runs/report_1abd0a09adf3_20260902_02`。

## 四个固定scene的中心分类结果

表内全部是Panel B/C实际绘制的combined-valid unique centers，每个中心只计一次：

| scene | 中心数 / coverage | TP / FP / TN / FN | Accuracy | AP | F1 | BA | AUROC | Precision / Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Re160 | 60,560 / 94.6250% | 2,978 / 0 / 56,956 / 626 | 0.989663 | 0.970477 | 0.904892 | 0.913152 | 0.998459 | 1.000000 / 0.826304 |
| Re640 | 60,555 / 94.6172% | 2,827 / 123 / 57,054 / 551 | 0.988870 | 0.943060 | 0.893489 | 0.917367 | 0.996128 | 0.958305 / 0.836886 |
| Re6400 | 62,315 / 97.3672% | 2,705 / 285 / 58,821 / 504 | 0.987339 | 0.919959 | 0.872721 | 0.919060 | 0.995663 | 0.904682 / 0.842942 |
| Boeing 747 | 61,433 / 95.9891% | 2,437 / 74 / 58,366 / 556 | 0.989745 | 0.942748 | 0.885538 | 0.906483 | 0.996500 | 0.970530 / 0.814233 |

四图合计244,863个combined-valid centers；primary完整valid-row投影另有406,177行。Re160的
固定source F1最高，Re6400最低，三个Reynolds number的F1为
`0.904892/0.893489/0.872721`，这只是固定source描述，不能解释为稳定的Reynolds number因果规律。
正类只占各图中心的约4.87%–5.95%，因此Accuracy主要由TN支配，解读应以F1、AP、BA、
precision和recall为主。

Panel B只绘制primary `dual_histogram_llr`。`negative_ecdf`和`direct_rank_mean_top5`只在
`per_figure_metrics.csv`中对all-parent-valid rows重算，不进入任何panel、不触发arm switching、不是
primary success证据。同理，Panel A的legacy/expanded pathline只是IVD背景上的尺度几何语境，
不是Panel B的分类输入或分数解释。

## 最终本地rendered QA与证据哈希

机器阶段原子发布恰好35个文件；本地exact reporting commit、clean checkout的auditor新增26个
审计文件，最终bundle恰好61个文件。4/4 alignment、PDF文字、SVG可编辑文字和21×5 inch
目视复核全部PASS；PDF最小文字为7 pt，collision hard failure=0。51个warning经逐图最终尺寸
复核，均是三维坐标刻度或x轴文字与栅格化pane/data edge接触，没有标题、图例、panel或数据被遮挡。

主要文件SHA-256为：`RUN_COMPLETE.json`
`8edefa024db642237332d2079b70dd6c4c7260b68b658af98e6e3cdbaf9cdcfe`；
`result_manifest.json` `2123933ee472f6248a6dd175c2e21f59683a509712aca66799f4bbc0f12e1f2d`；
`visualization_manifest.json` `523b874d85f79ad833398235c26fe6ee297e839d3832dce58079c79452513196`；
`per_figure_metrics.csv` `889c88b9278d358cf8ebc2a15ba7ae9c18f505886d0b63796cb83c19f7cd5772`；
`input_manifest.json` `1120a589a0f578294438c07c20b60bc488ccad11871f8768820cb7afc69d9598`；
`figure_contract.json` `2de7580fd5f9089798ebd096c5dea4710b851ef3551c049f336642f1c5addaeb`；
`delivery_qa_summary.json` `8bda4c6a6d46c63c6b10208d5836a4c37709061ea3849d9f3eeddeeff5efaa3b`；
visual review `c3c181146ea5525cc3b9c30350d7295bcc2c9c3136b221dcd0d0dff2226495c4`。
全35+26个bundle文件的完整哈希映射与输入/日志哈希见
`docs/evidence/Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1_local_summary.json`。

## 结论边界

RankLikelihood primary只使用source-centered seed-time curl rank，不使用FMT pathline geometry或Raw
pathline coordinates；Panel A pathline不能解释Panel B分类分数。本方法还使用目标source自身的无标签
rank总体和valid mask，因此是transductive。四个flow都属于已暴露development数据，单source图没有
置信区间，不能称formal confirmation、聚类、FMT几何分类、independent per-primitive classifier，或替代完整五family
统计。结论从“只有冻结协议、尚无真实图”修订为“四张固定source暴露开发图已通过机器事务和本地
rendered QA，可作描述性交付”；变化原因是job `51167090`与61文件QA首次完成，原结论只适用于
预运行阶段，不是一个性能判断。
