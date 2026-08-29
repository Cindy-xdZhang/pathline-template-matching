# FMT 迁移来源与修改说明

## 审计基线

- FMT 本地仓库：`C:\Users\xingdi\sources\FMT`
- 最终审计时本地 HEAD：`3e5ac058af5a1bb155846542bb2035dfd300a020`
- 最终审计时 `origin/main`：`7fc17c42949160010c0bdaf19a00abecbafabd86`
- 核心迁移审计期间 FMT HEAD 从 `1f486c39fcedef5f384f3060b30ae28047385bba` 前进到上述 commit；下表核心源文件内容 hash 在该次最终复核时稳定、tracked 且 clean。三联图参考发生在后续工作树 HEAD `4ba009ccd9cb604019aeb830591e6c54e2c7742a`，两份视觉源文件当时均未跟踪；其 hash 和“无来源提交”状态在表中单独标明，不能与核心基线 HEAD 混写。
- FMT 工作树另有用户改动；本项目未修改、清理或提交它们。

## 代码映射

| FMT 源文件 | 源文件 SHA-256 | 最近相关提交 | 本项目位置 | 迁移方式 |
|---|---|---|---|---|
| `FMT_Utils/DFT_FMT_3D.py` | `7ff09c1b578d0bd0927ccf0b771de30a01e0b79fff1b7b40576cbca57ce1e6c1` | `07ee64b60451cede031d55a6c0eea7cdec36718a` | `src/pathline_template_matching/fmt_descriptor.py` | 文件逐字复制；wrapper 固定 7×32、Task5 cache 的 1/1 数值约定和内容派生 descriptor ID |
| `FMT_Utils/MultiscalePathline_3D.py` | `455518647250899d5ca3e77357f548b9ba64f53691829f9881e16e0941100b52` | `a01d0576e7edff3ff5e949f06baa8e39eec749f6` | `scales.py`, `primitives.py` | 抽取 scale dataclass、balanced assignment 和多尺度输出契约；新增严格校验 |
| `FMT_Utils/FMT_3D_pipeline.py` | `689c469823420edcd52f1f7840f592dace623196c9507a3e07594fc73bd02ce7` | `f8f56750b65ee01a646ef8c5f71be932ce1057b1` | `primitives.py` | 只抽取 3D seed cross、完整线过滤和 rounded-index 重采样；不复制绘图/marching-cubes 依赖 |
| `FLowUtils/flowlineIntegral.py` | `d7bf58642cdc5f4e42c00bd13a0bd200efcd394019feb30d1857df2dcdc67200` | `944d2062d8bd8bf9210c79007da9ceb3c82f527f` | `integration.py` | 只抽取 Numba 3D trilinear/quadrilinear、Euler/RK4 batch core；不复制 2D、PyCUDA 与 backend 全局状态 |
| `FLowUtils/VectorField3d.py` | `ce982d2faf2f355fe5b0c9a08568201b2161a37c5d3c4ed8b8e1534df9d52e5c` | `a3fd97db7627d657848a6d9e962754c9c8ed3f91` | `vector_field.py` | 重写为最小、不可变、规则网格容器并校验 shape/domain/spacing/time |
| `FMT_Utils/NetCDF_window_3D.py` | `a7ef0ad56d29160f202425ea1e188d1f1bc3015de846352d23bac2253a2d3bd1` | `9748974aeae009d46d958eae3df1aa7d52e747c4` | `netcdf_io.py` | 保留维度感知 transpose/stride；mask、非有限、缺失/坏坐标、非等间距和额外维度均直接失败 |
| `FLowUtils/ScalarField3d.py` | `cae0e07cd9071a6173a7dbeec2eddfb73823c8752435fa9df1cb13f46a969622` | `5cba739502313ca33d1d8374d9de0487c892ae98` | `ivd.py` | 只抽取 signed curl 与 whole-loaded-volume IVD；不带 Lambda-2、marching cubes 和绘图 |
| `config/mainExp_Task5_3D_1.1.yaml` | `620a67fca986ddfc1d427158994c3cb85e7d04eab50aaa23c9d1db9807b3b735` | `a01d0576e7edff3ff5e949f06baa8e39eec749f6` | `config/mainExp_TemplateMatching_1.1.yaml` | 复制 18/6/9 个尺度 tuple；重新定义 retrieval 角色 |
| `Build_Task5_Multiscale_Cache.py` | `a6824df66840871d15df187c7da4bf00758a46847a391b571b8270fb541d45f3` | `a01d0576e7edff3ff5e949f06baa8e39eec749f6` | provenance/data validator 参考 | 该源在编码处硬编码 `neighbor_weight=1, neighbor_scale=1`；新 wrapper 与其一致。2.1/3.1的空间弧长builder是本项目新实现，不是复制该源 |
| `Visualize_Task1_3D_Horizontal.py` | `776527b2d25649cda290104f2100cff639a6e30ba57bfb300f46be5d3d311602` | 无；2026-08-27 审计时为 FMT 未跟踪文件，FMT 工作树 HEAD `4ba009ccd9cb604019aeb830591e6c54e2c7742a` 只提供上下文、不是其来源提交 | `visualization.py` | 重写横向 21×5 三栏、共享正交相机；不复制旧预测/K-Means/cache 路径 |
| `Visualize_Task1_3D_PaperCandidates.py` | `5953b41166c626659fba73b25968bbad04186917686d1190c5da306a227c3410` | 无；2026-08-27 审计时为 FMT 未跟踪文件，FMT 工作树 HEAD 同上仅提供上下文 | `visualization.py`, `development_report.py` | 重写语义配色、IVD/pathline、binary assignment、TP/FP/FN/TN 层；当前 prediction 来自 exact 1NN，IVD-p95 mesh 由本项目 loader/IVD 与 scikit-image Marching Cubes 重建 |

本项目新写的 `encoder.py`、`library.py`、`data_access.py`、dataset registry 和文档不在 FMT 中存在，不能标成逐字复制。只有 `fmt_descriptor.py` 是逐字复制文件。

`config/mainExp_TemplateMatching_1.1_development.yaml` 及其 1.2 空类-stratum 修订、development-only retrieval、统计和三联图协议都是本项目新定义，不存在于 FMT。它们只消费旧 Task5 cache 中已有的 `raw_features`、`fmt_features`、`reference`、`seeds`、`scale_id` 和 metadata；不把历史 cache 重新解释为 sealed confirmation，也不声称重新积分了 Ibex 上缺失的 raw fields。

本项目的 `development_data.py`、`development_library.py`、`development_experiment.py`、`development_report.py`、`matcher.py`、`metrics.py`、`pca.py` 和 `visualization.py` 均为新写代码，不是 FMT 文件逐字复制。上表最后两行记录的是视觉语义、固定相机和配色的参考来源；两个源文件没有 Git commit，不能用 FMT HEAD 冒充来源版本。1.1在指标前失败；1.2、2.1和3.1的development结果、数值commit及证据哈希分别记录在各自版本文档和`docs/experiment_log.md`中，均不属于FMT来源结论。

## 明确排除的实现

- `FMT_Utils/FMT_encoder.py`：含 Batch Normalization affine/running statistics；启用 temporal Discrete Fourier Transform 时还有 `torch.nn.Parameter`，不符合严格无可训练参数要求。
- `FMT_Utils/DCT_FMT_encoder.py`：只使用二维 `(x,y)`，不进入三维主路线。
- `Task5FeatureRecipes_3D.py` 的 44D kinematic recipe：用当前 batch 的平均涡量，同一 query 会随 batch 组成改变。
- FMT 的 2D、PyCUDA、旧 K-Means/prediction pipeline、hard-coded raw/cache 路径和 marching-cubes 代码：未迁移。只重写了上表明确记录的三栏视觉语义、相机和配色。

## FMT 结论的允许引用范围

来源是 FMT 的 `docs/experiment_log.md` 中 `mainExp_Task5_3D_1.1` 及其版本文档。它支持在当时监督网络、whole-loaded-volume IVD p95、10 个数据条目和已测试尺度内，FMT 相对同维 Raw Principal Component Analysis（PCA，主成分分析）residual 的 dataset-macro F1/Average Precision 增益为 `+.0891/+.1116`。

它不证明 FMT Euclidean nearest neighbor 有效、161D descriptor 尺度不变、连续任意尺度外推有效、单一 recipe 对所有 physical family 最优，也不使旧 10 个数据条目成为新项目 sealed confirmation。
