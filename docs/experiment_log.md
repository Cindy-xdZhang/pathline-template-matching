# Pathline Template Matching 实验版本表

本文件是方法级结论的唯一汇总位置。每个版本必须保留；失败、负结果和被修订结论不得删除。任何结论必须同时给出 config、Git commit、逐次结果、汇总文件和设备记录。

| 版本 | 日期 | 状态 | 研究问题与固定技术 | 主要代码 / config | 数据与拆分 | 指标与证据 | 当前结论 |
|---|---|---|---|---|---|---|---|
| `Other_ProjectBootstrap_1.1` | 2026-08-27 | COMPLETED | 审计 FMT、迁移独立 161D descriptor/尺度协议/IVD/NetCDF loader/RK4 primitive core，建立 exact 1NN library 和 Ibex 数据验证 | `src/pathline_template_matching/`, `config/datasets.yaml`, `docs/source_provenance.md` | 无性能数据；只做单元、smoke、Git/Ibex/data access 验证 | code commit `f202095c1572c668716a407d987fb0882add3ab6`；Ibex 28/28 tests、5/10 raw、10/10 cache/100 slices；报告 SHA-256 `609bd188229c3c7a40db2d8ea1648c517a7bd3a0ced5bccec4a0c207b714a84d` | 支持“Git 同步、核心代码和既有数据访问可运行”；不产生模式匹配性能结论 |
| `Verify_IndependentFMT_1.1` | — | PLANNED | 验证 descriptor 在 single/batch/chunk 下逐位一致，并比较 161D base、63D time-local Gram、224D independent concat 的尺度敏感性 | 待创建，不得使用 sealed confirmation | 仅旧 FMT development families/scales | batch invariance、within-pattern/cross-pattern distance ratio | 尚无结论 |
| `mainExp_TemplateMatching_1.1` | 2026-08-27 | DEVELOPMENT_FAILED_PRE_METRIC_EMPTY_STRATUM | Task5-cache-compatible 161D FMT、library-only mean/std、exact Euclidean 1NN；672D Raw、library-only Raw-PCA 161D、prior 对照；任一 `flow×time×scale` 缺类即 fail | `config/mainExp_TemplateMatching_1.1.yaml`, `config/mainExp_TemplateMatching_1.1_development.yaml`, `docs/mainExp_TemplateMatching_1.1.md` | cache-backed development：7-family leave-one-out，seen/unseen scale 分开；sealed confirmation access forbidden | AP、F1、逐 family/scale、bootstrap；但本次未进入指标计算 | Ibex job `50930751` 在首折发现 `channel/ordinal0/lib_o025_d0125_n48` 为 negative=228、positive=0，按冻结规则失败；无性能结论。后续不得静默修改1.1，必须新版本 |
| `mainExp_TemplateMatching_1.2` | 2026-08-27 | DEVELOPMENT_COMPLETED_CONFIRMATION_NOT_RUN | 继承1.1；唯一修订是空类 library stratum 两类模板都选0并审计，query/PCA-fit/prior不删除该数据 | `config/mainExp_TemplateMatching_1.2_development.yaml`, `docs/mainExp_TemplateMatching_1.2.md`; numerical commit `700d392b590f46a68f8ef6e973524ee0a7886c62` | exposed legacy Task5 cache，7-family leave-one-out；seen/unseen scale 分开；sealed confirmation access forbidden | Ibex `50932239`；100 cache files/390,140 primitives；result manifest file SHA `217cdaf7...`; physical-family macro；5000 paired source-timeslice bootstrap；20 fixed triptychs | Mixed development observation：FMT−Raw 的 seen/unseen AP/F1 点估计都为正；FMT−Raw-PCA 的 seen AP/F1 为 `+0.0527/+0.0806`，unseen 为 `−0.0726/−0.0711`，对应95%区间不跨0。按冻结规则只作描述，不宣告主命题通过/失败；formal confirmation 未运行 |
| `Other_MainExp12FigureLayout_1.1` | 2026-08-27 | COMPLETED_LAYOUT_ONLY | 修复原三联图标题接触画布上边界；只从已哈希 scene 重绘，禁止重算指标或覆盖原图 | `scripts/rerender_triptychs_from_scenes.py`, `src/pathline_template_matching/visualization.py`; renderer commit `cfe2afcf01133a3a7034db05175710e8f9dd70fe` | 精确复用 `mainExp_TemplateMatching_1.2` 的20个固定 scene | 20/20 为7560×1800；第一像素行非白像素最大值0；counts/camera/bounds/scene hash全部与原图一致；manifest file SHA `8fd85b96...` | 仅改布局，不改变任何数值结论；修正版是展示首选，原 Ibex 图仍保留为数值产物 |

## 修订格式

修改旧结论时必须新增版本，并明确写成：

```text
旧结论 → 当前结论 → 为什么改变 → 旧结论错在哪里或适用范围哪里更窄
```
