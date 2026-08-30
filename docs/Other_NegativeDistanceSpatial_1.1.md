# Other_NegativeDistanceSpatial_1.1：负模板距离与空间后处理诊断

状态：**`exposed_development_diagnostic_completed`**。本版本不修改 `mainExp_TemplateMatching_3.1`，只复用两次已完成运行保存的逐 query FMT 距离。唯一 config 为 `config/Other_NegativeDistanceSpatial_1.1.yaml`，冻结 SHA-256 是 `e891af14037c464a6042143625646be0d2f71c37e5e9ff30e50cc30dd553c141`；Ibex 作业 `51039505`使用 numerical commit `7118af6c17b964b5561e6e297609f431f81aa020`完成。

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

## Ibex 结果

作业 `51039505`在 `cn511-13` 上以 CPU 完成，exit `0:0`，用时 `00:01:49`，MaxRSS `1396428K`；124/124 tests 通过。输出含 884,698 条无 reference 列的预测、384 条逐组指标、144 条汇总指标和 192 条只作上界的 oracle 记录；三层 manifest、行数、文件大小与 SHA-256 均复核通过。

为了得到一个可进入后续验证的单一候选，在四个目标流场的八个 `dataset × source × block` 已暴露组上等权比较全部预先冻结候选，选得 `masked Gaussian rank sigma=1 grid + fixed top 5%`。这个选择使用了已暴露标签结果，因此只是下一步 train-only nested validation 的候选，不是已验证方法。

| 范围 | Accuracy | AP | F1 | Balanced accuracy | AUROC | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 四flow、8组等权宏平均 | 0.9527 | 0.5955 | 0.5451 | 0.7562 | 0.9559 | 0.5710 | 0.5350 | 0.7933 |
| Re160 | 0.9546 | 0.6905 | 0.6141 | 0.7626 | 0.9696 | 0.7160 | 0.5405 | 0.8118 |
| Re640 | 0.9430 | 0.4739 | 0.4751 | 0.7082 | 0.9336 | 0.5139 | 0.4422 | 0.8048 |
| Re6400 | 0.9511 | 0.5949 | 0.5229 | 0.7432 | 0.9678 | 0.5356 | 0.5109 | 0.9392 |
| Boeing 747 | 0.9621 | 0.6226 | 0.5682 | 0.8108 | 0.9525 | 0.5185 | 0.6465 | 0.6174 |

父 family-held-out exact-1NN 在同一八组上的等权 F1 为 `0.2278`；统一候选提高到 `0.5451`。这支持“平衡正负模板下的 signed 1NN margin 与稀疏正模板 hub 是主要故障源之一”，但并不支持“已达 F1 0.7–0.8”：任何非 oracle 方法都没有达到 `0.7`，且 Re640 仍是明显瓶颈。下一步必须在完全不打开外层留出 family 的情况下，重建自然负类 library-only scaler 和 exact-scale negative k-nearest-neighbor，再用 inner leave-one-family-out 选择 `k/sigma/threshold`。

可追溯证据：`aggregate_metrics.csv` SHA-256 `0a5988cff148be534c27fb98fa2bbddeed37d70b375a9b735bf9d58723b9d6ae`，`per_group_metrics.csv` SHA-256 `f857590d334227829309c7c51b7dcfd71d390909d66ad9744d014610146e455e`，`predictions.csv` SHA-256 `cc4651baefeabda0c4570ad4a3f8a6b855e2616e835af02994294a2095f177f2`，`result_manifest.json` SHA-256 `b4e171d499af12dc5aca102950a33eba6762fb122ebe0b2828a0623236d9bff3`。
