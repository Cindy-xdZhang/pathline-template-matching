# FMT 迁移来源与修改说明

## 审计基线

- FMT 本地仓库：`C:\Users\xingdi\sources\FMT`
- 最终审计时本地 HEAD：`3e5ac058af5a1bb155846542bb2035dfd300a020`
- 最终审计时 `origin/main`：`7fc17c42949160010c0bdaf19a00abecbafabd86`
- 审计期间 FMT HEAD 从 `1f486c39fcedef5f384f3060b30ae28047385bba` 前进到上述 commit；下表源文件内容 hash 在最终复核时稳定、tracked 且相对最终 HEAD clean。
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
| `Build_Task5_Multiscale_Cache.py` | `a6824df66840871d15df187c7da4bf00758a46847a391b571b8270fb541d45f3` | `a01d0576e7edff3ff5e949f06baa8e39eec749f6` | provenance/data validator 参考 | 该源在编码处硬编码 `neighbor_weight=1, neighbor_scale=1`；新 wrapper 与其一致，正式 builder 尚未实现 |

本项目新写的 `encoder.py`、`library.py`、`data_access.py`、dataset registry 和文档不在 FMT 中存在，不能标成逐字复制。只有 `fmt_descriptor.py` 是逐字复制文件。

## 明确排除的实现

- `FMT_Utils/FMT_encoder.py`：含 Batch Normalization affine/running statistics；启用 temporal Discrete Fourier Transform 时还有 `torch.nn.Parameter`，不符合严格无可训练参数要求。
- `FMT_Utils/DCT_FMT_encoder.py`：只使用二维 `(x,y)`，不进入三维主路线。
- `Task5FeatureRecipes_3D.py` 的 44D kinematic recipe：用当前 batch 的平均涡量，同一 query 会随 batch 组成改变。
- FMT 的 2D/PyCUDA/绘图/marching-cubes 代码：与首版检索核心无关，未迁移。

## FMT 结论的允许引用范围

来源是 FMT 的 `docs/experiment_log.md` 中 `mainExp_Task5_3D_1.1` 及其版本文档。它支持在当时监督网络、whole-loaded-volume IVD p95、10 个数据条目和已测试尺度内，FMT 相对同维 Raw Principal Component Analysis（PCA，主成分分析）residual 的 dataset-macro F1/Average Precision 增益为 `+.0891/+.1116`。

它不证明 FMT Euclidean nearest neighbor 有效、161D descriptor 尺度不变、连续任意尺度外推有效、单一 recipe 对所有 physical family 最优，也不使旧 10 个数据条目成为新项目 sealed confirmation。
