# Other_MainExp31FamilyHeldOutVisualization_1.1：四个已暴露训练流场的完整物理族留出分类图

状态：**`frozen_pre_run_not_run`**。唯一配置为 `config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml`。本实验只回答用户要求的空间分类可视化问题，不修改 `mainExp_TemplateMatching_3.1` 的 descriptor、primitive、尺度、标签或主结果。

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
