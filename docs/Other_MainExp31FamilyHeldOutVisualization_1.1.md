# Other_MainExp31FamilyHeldOutVisualization_1.1：四个已暴露训练流场的完整物理族留出分类图

状态：**`family_held_out_exposed_development_completed`**。唯一配置为 `config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml`。本实验只回答用户要求的空间分类可视化问题，不修改 `mainExp_TemplateMatching_3.1` 的 descriptor、primitive、尺度、标签或主结果。

完成证据：Ibex job `51029080` 在 numerical commit `86be29698eb689c0e269fe987a5b6d5f125a67be`、config SHA-256 `6fec35d2f64a3b593a74e8b35674137b1665ce169491e3546384142514b46670` 上完成，状态 `COMPLETED 0:0`。作业运行于 `gpu102-02` 的一张 NVIDIA A100-SXM4-80GB，开始于 2026-08-30 14:18:35 +03:00，结束于 14:28:04，elapsed `00:09:29`；105/105 tests 和 deterministic CUDA matcher gate 均通过。

## 为什么不能直接使用当前 96,160 模板库

`cylinder3d`（Re160）、`halfcylinderRe640`、`halfcylinderRe6400` 和 `boeing747` 都参加了 3.1 的模板库构建。若直接查询当前库，最近模板可能来自 query 自身、同一 source，或同一 physical family；这种图只能称为库内重建，不能解释跨流场泛化。

因此本版本为每个目标完整排除其 physical family，并从仍合格的 3.1 train caches 重新运行平衡模板抽样、重新拟合 FMT feature mean/std，再进行 global exact one-nearest-neighbor（精确一近邻）分类。禁止简单过滤当前模板库，因为 3.1 使用一个按固定遍历顺序消耗的全局 PCG64 随机数生成器；移除一个 family 后，后续选中模板的身份也必须重新计算。

## 冻结 folds

| Fold | Query datasets | 完整排除 family | Library datasets |
|---|---|---|---|
| `holdout_half_cylinder` | `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400` | `half_cylinder` | `deltaWing_resampled`, `deltaWing_LBM`, `f22raptor`, `channel`, `boeing747` |
| `holdout_boeing_747` | `boeing747` | `boeing_747` | `cylinder3d`, `halfcylinderRe640`, `halfcylinderRe6400`, `deltaWing_resampled`, `deltaWing_LBM`, `f22raptor`, `channel` |

两个 fold 都独立从 `PCG64(15068)` 重新开始。Library 遍历顺序、`dataset×source×scale×class` 双类非空时各选一个模板、空类时两类都选零、selected-library-only 标准化、Euclidean distance、二分类 score 和 tie rule 全部继承 3.1。目标 family 的 cache 不进入该 fold 的 library 或标准化。

## 固定 query 与图件合同

四个 query dataset 都固定 source ordinal `2`，不按 accuracy、错误数量或视觉效果选择。每个 dataset 的 `legacy_2_1`（scale IDs 0–999）与 `expanded_3_1`（1000–1999）分开出图，共8张；两个 block 的 query 不叠画、不投票。每个 block 都查询该 fold 的 global 2000-scale library。

每张三联图依次为：

1. whole-loaded-volume IVD-p95 等值面，以及只按 reference class 和几何位置选出的120条正类、120条负类中心 pathlines；
2. 全部 valid query rows 的 FMT exact-1NN 涡/非涡模板类别分配；这一栏不是 clustering；
3. 同一批完整 query rows、同一顺序、同一相机和 bounds 下的 TP/FP/FN/TN。

240条 pathlines 是平衡的解释性背景，不代表自然类别比例。Accuracy、Average Precision、F1、balanced accuracy、Area Under the Receiver Operating Characteristic Curve、precision、recall 和 TP/FP/TN/FN 都从每张图的全部 valid rows 计算；invalid rows只进入coverage分母。

## Figure contract

- 核心结论：固定3.1 FMT exact-1NN方法可在完整目标物理族不进入库时，对四个指定的已暴露流场产生可定位、可审计的空间分类与错误分布。
- Results-level问题：当完整目标 physical family 被排除后，FMT assignment 在哪里与 IVD-p95 一致或不一致？
- 图件类型：`image plate + quantification`。
- 面板作用：a提供物理背景；b显示主要分类输出；c用reference验证并定位错误。
- 输出定位：21×5英寸实验报告图，不冒充Nature投稿版；三维 marks 栅格化，SVG/PDF文字可编辑，PNG为360 dpi预览。
- 可推翻结论的风险：目标family泄漏进library/scaler、按效果选source、三栏rows不一致、跨block投票、coverage不报、把240条平衡展示线当自然分布。

## 输入与证据边界

输入只允许使用数值commit `260a07ad380d64fc300cabe8926244e92d8ba04a` 生成并已哈希验证的32个3.1 train cache shards/sidecars。Re160和Re640原始NetCDF在Ibex可读；Re6400和Boeing 747在Ibex没有注册原始路径，但其49帧portable windows、IVD volume、pathline geometry与FMT features已由parent cache manifest锚定，足以重现本实验。不得读取任何历史confirmation source pack。

Runner必须先在新run目录写入冻结config和32个sidecar/cache的路径、大小、SHA-256、builder commit及portable marker身份，之后才允许打开NPZ arrays。每个fold还必须输出library、preprocessing、逐query match metadata、逐图metrics、scene/render manifests和最终result manifest。所有文件不可覆盖；`RUN_COMPLETE.json` 最后写。

## 结论边界

即使当前fold排除了完整目标family，这四个flow仍在早期项目和3.1训练阶段被读取过。因此结果只能称为 **family-held-out exposed-development**，不能称为formal confirmation；八张图也不替代跨source、跨family的聚合性能实验。

## 完成后的 library 与 preprocessing 审计

| Fold | Library templates | 正类 / 负类 | FMT宽度 | 标准化拟合数据 | 泄漏检查 |
|---|---:|---:|---:|---|---|
| `holdout_half_cylinder` | 50,770 | 25,385 / 25,385 | 161 | 仅该fold平衡模板库 | 完整排除`half_cylinder`，PASS |
| `holdout_boeing_747` | 86,728 | 43,364 / 43,364 | 161 | 仅该fold平衡模板库 | 完整排除`boeing_747`，PASS |

两个fold都重新拟合161维FMT feature mean和population standard deviation；未复用3.1的96,160模板库或其标准化参数，也未使用query统计量。`library_manifest.json`明确记录`held_out_family_leakage=false`，`preprocessing_manifest.json`明确记录`query_statistics_used=false`和`parent_fitted_artifact_reused=false`。

## 逐图结果

Average Precision（AP）衡量正类排序质量；balanced accuracy（BA）是正类召回率与负类召回率的平均；Area Under the Receiver Operating Characteristic Curve（AUROC）衡量跨阈值排序能力。下表数值来自不可变的`per_figure_metrics.csv`；TP/FP/TN/FN分别为true positive、false positive、true negative和false negative。

| Dataset | Scale block | Valid / 64,000 | Coverage | TP | FP | TN | FN | Accuracy | AP | F1 | BA | AUROC | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cylinder3D Re160 | `legacy_2_1` | 60,560 | 0.9463 | 2,936 | 10,373 | 46,583 | 668 | 0.8177 | 0.3583 | 0.3472 | 0.8163 | 0.8457 | 0.2206 | 0.8147 |
| Cylinder3D Re160 | `expanded_3_1` | 43,347 | 0.6773 | 2,906 | 23,766 | 16,354 | 321 | 0.4443 | 0.2272 | 0.1944 | 0.6541 | 0.7479 | 0.1090 | 0.9005 |
| Half-cylinder Re640 | `legacy_2_1` | 60,555 | 0.9462 | 2,918 | 21,866 | 35,311 | 460 | 0.6313 | 0.3000 | 0.2072 | 0.7407 | 0.8145 | 0.1177 | 0.8638 |
| Half-cylinder Re640 | `expanded_3_1` | 42,463 | 0.6635 | 2,448 | 32,499 | 7,376 | 140 | 0.2314 | 0.1788 | 0.1304 | 0.5654 | 0.7354 | 0.0700 | 0.9459 |
| Half-cylinder Re6400 | `legacy_2_1` | 62,313 | 0.9736 | 1,495 | 2,477 | 56,629 | 1,712 | 0.9328 | 0.3360 | 0.4165 | 0.7121 | 0.7182 | 0.3764 | 0.4662 |
| Half-cylinder Re6400 | `expanded_3_1` | 57,906 | 0.9048 | 2,508 | 14,536 | 40,272 | 590 | 0.7388 | 0.3024 | 0.2490 | 0.7722 | 0.8028 | 0.1471 | 0.8096 |
| Boeing 747 | `legacy_2_1` | 61,432 | 0.9599 | 1,985 | 15,980 | 42,460 | 1,007 | 0.7235 | 0.2288 | 0.1894 | 0.6950 | 0.6975 | 0.1105 | 0.6634 |
| Boeing 747 | `expanded_3_1` | 17,601 | 0.2750 | 405 | 8,225 | 8,834 | 137 | 0.5249 | 0.1199 | 0.0883 | 0.6325 | 0.6875 | 0.0469 | 0.7472 |

四个source ordinal 2、两个scale block合计406,177条valid query。每个block的valid population不同，因此上表的legacy/expanded差值不是控制其他变量后的因果比较。

## 结果解释

- Cylinder3D Re160的`legacy_2_1`在这8张固定图中具有最高BA（0.8163）和AUROC（0.8457），空间图中预测正类主要集中在上游涡区域，但仍有10,373个FP。
- Re160和Re640的`expanded_3_1`明显偏向预测涡类：recall分别为0.9005和0.9459，但precision仅0.1090和0.0700；对应23,766和32,499个FP。高recall不能单独解释为分类更好。
- Re6400的`legacy_2_1`呈相反取舍：accuracy最高（0.9328）且precision为0.3764，但recall只有0.4662。`expanded_3_1`把recall提高到0.8096，同时precision降至0.1471。
- Boeing 747的证据较弱。`legacy_2_1` coverage为0.9599，但precision仅0.1105；`expanded_3_1` coverage只有0.2750、precision 0.0469、F1 0.0883，不能据此评价完整长弧尺度空间。
- 各图正类只占valid rows的约3.1%–7.4%，所以accuracy容易被大量TN抬高；判断分类效果应同时看coverage、AP、F1、BA、AUROC、precision和recall。

## 图件与质量检查

每个dataset输出`legacy_2_1`和`expanded_3_1`两张三联图，共8张；每张都有7560×1800、360 dpi PNG，以及带可编辑文字的PDF和SVG。全部8份panel-alignment audit通过，PDF最小文字为7 pt，未发现小于5 pt的文字。Collision audit为0 FAIL、105 WARN；逐张叠加层目视复核确认WARN只来自三维坐标刻度与坐标面或栅格边缘的预期接触，没有标题、panel label、坐标标签或数据被裁切。八张图本身没有可见legend，因此交付时必须同时说明：panel b蓝色为预测非涡、红色为预测涡；panel c浅蓝为TN、红圆为TP、紫三角为FP、橙色`x`为FN。图件适合带上述说明的本项目实验报告；由于目标画布为21×5英寸且PNG为360 dpi，不标记为Nature投稿终稿。

## 可复现路径与哈希

- Ibex run：`/ibex/user/zhanx0o/pathline-template-matching/Other_MainExp31FamilyHeldOutVisualization_1.1/runs/slurm_51029080_86be29698eb6`
- 本地下载：`outputs/Other_MainExp31FamilyHeldOutVisualization_1.1_download/slurm_51029080_86be29698eb6`
- `result_manifest.json` SHA-256：`57f03ba16ad8cfa0e1e0a9efd93f2dde7ae5866f173fad20055efb6939d4188e`
- `RUN_COMPLETE.json` SHA-256：`63325cf51bd9a4322ef5ba6385d00aa063b30096134679d153aa6733ac28314b`
- `per_figure_metrics.csv` SHA-256：`3fd087b27b9997ca07c57f7720864e9c1f7ade3c12ace4b898e72381b1d5533c`
- stdout / stderr SHA-256：`16a0dddc7db4300eae678d83d1339cd351a862f1a390d5080c70e67899c69b10` / `7dd8e59643f361152f97aa89fd90349cda353f05ff72923e85fee8b0e475720c`
- Result manifest列出的70个artifacts在Ibex和本地均完成size与SHA-256复核，0缺失、0不一致；run目录另有`result_manifest.json`和最后写入的`RUN_COMPLETE.json`，总计72个文件。
- 结构化摘要：`docs/evidence/Other_MainExp31FamilyHeldOutVisualization_1.1_ibex_summary.json`
