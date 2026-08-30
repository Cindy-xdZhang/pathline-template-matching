# Other_NegativeDistanceSpatial_1.1：负模板距离与空间后处理诊断

状态：**`frozen_pre_run_not_run`**。本版本不修改 `mainExp_TemplateMatching_3.1`，只复用两次已完成运行保存的逐 query FMT 距离。唯一 config 为 `config/Other_NegativeDistanceSpatial_1.1.yaml`，冻结 SHA-256 是 `e891af14037c464a6042143625646be0d2f71c37e5e9ff30e50cc30dd553c141`。

## 为什么做这个诊断

3.1 的分类分数是“最近负模板距离减最近正模板距离”。审计发现真实训练候选的正类约占 4%–5%，但模板库按每个可用分层各取一正一负，最终成为 50:50；少数稀疏正模板还会成为吸引大量 query 的错误 hub。已暴露结果进一步显示，真涡 query 往往同时远离正、负模板，二者相减会消掉有效异常信号。故本版本检验只保留 `distance to nearest negative template` 是否更合理。

这里的负模板距离仍继承父实验“平衡模板库拟合 scaler、每个分层只取一个负模板、跨2000尺度全局搜索”的限制。本版本只能隔离 score 与空间后处理的作用；若方向成立，后续 train-only 版本必须用自然非涡候选重新拟合 preprocessing，并检验 exact-scale negative k-nearest-neighbor，而不能把本诊断当作最终 one-class matcher。

## 冻结方法

每个 `dataset × source ordinal × scale block` 独立处理，禁止跨 source 或 block 平滑：

1. 以 `fmt_nearest_negative_distance` 为基础异常分数，越大越可能是涡。
2. 对分数组内稳定排序；同分按 `center_seed_index` 打破，百分位为 `(rank+1)/N`。
3. 按 3.1 的 `40×40×40` seed index 恢复 `[z,y,x]` 网格，只对 valid rows 做 mask-normalized Gaussian 平滑。sigma 固定扫描 `0, 0.5, 0.75, 1, 1.5, 2` 个网格间隔。
4. 每个分数分别产生两种无标签预测：一维确定性 two-means 中高均值簇判涡；以及按 IVD-p95 定义先验冻结的组内 top 5%。真实 valid-query 正类比例不一定是 5%，因此 top-5% 只是明确标注的 transductive 诊断。
5. 两份父 CSV 在物理上包含 `reference_label`，CSV reader 读取整行字节后才做列投影；因此本版本不声称预测阶段“没有打开标签所在文件”。预测阶段的显式投影和下游逻辑不接收 `reference_label`，预测文件也不含标签。预测文件完成并关闭后，runner 才执行第二次显式 reference 投影，按稳定 row key 合并并评测。

输出同时给 Accuracy、Average Precision、F1、balanced accuracy、Area Under the Receiver Operating Characteristic Curve、precision、recall、coverage。oracle threshold 只作为“当前排序的理论诊断上界”，不得用于命名赢家或声称可部署。

## 证据边界

Tangaroa、Smoke、三个 cylinder 和 Boeing 747 的标签与旧预测均已看过，且 sigma 网格是在机制诊断后冻结。本实验只能回答故障机制和候选方法问题，不能称为无偏模型选择、formal confirmation 或主方法成功。若结果支持该方向，下一步必须新建 train-only 完整 physical-family nested validation；只有该验证冻结唯一方法后，才能建立 `mainExp_TemplateMatching_4.1`。

成功目标仍不降低：后续 train-only nested family/source macro F1 目标为 `0.70`，precision 与 recall 都不低于 `0.60`。本 exposed diagnostic 即使出现 F1 `0.7–0.8` 也不替代该门槛。
