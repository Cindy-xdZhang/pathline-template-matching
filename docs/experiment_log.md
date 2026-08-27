# Pathline Template Matching 实验版本表

本文件是方法级结论的唯一汇总位置。每个版本必须保留；失败、负结果和被修订结论不得删除。任何结论必须同时给出 config、Git commit、逐次结果、汇总文件和设备记录。

| 版本 | 日期 | 状态 | 研究问题与固定技术 | 主要代码 / config | 数据与拆分 | 指标与证据 | 当前结论 |
|---|---|---|---|---|---|---|---|
| `Other_ProjectBootstrap_1.1` | 2026-08-27 | IN_PROGRESS | 审计 FMT、迁移独立 161D descriptor/尺度协议/IVD/NetCDF loader/RK4 primitive core，建立 exact 1NN library 和 Ibex 数据验证 | `src/pathline_template_matching/`, `config/datasets.yaml`, `docs/source_provenance.md` | 无性能数据；只做单元、smoke、Git/Ibex/data access 验证 | 本地测试已完成；须在首次 push→Ibex clone→全 slice cache/raw validation 后填写 deployment table | 当前只支持本地代码检查；不产生模式匹配性能结论 |
| `Verify_IndependentFMT_1.1` | — | PLANNED | 验证 descriptor 在 single/batch/chunk 下逐位一致，并比较 161D base、63D time-local Gram、224D independent concat 的尺度敏感性 | 待创建，不得使用 sealed confirmation | 仅旧 FMT development families/scales | batch invariance、within-pattern/cross-pattern distance ratio | 尚无结论 |
| `mainExp_TemplateMatching_1.1` | — | METHOD_PREREGISTERED_NOT_FULLY_FROZEN | Task5-cache-compatible 161D FMT、library-only mean/std、exact Euclidean 1NN；672D Raw、library-only Raw-PCA 161D、prior 对照 | `config/mainExp_TemplateMatching_1.1.yaml`, `docs/mainExp_TemplateMatching_1.1.md` | development: leave-one-physical-family-out；confirmation manifest 尚缺 | AP、F1、逐 family/scale、timeslice paired bootstrap | 尚无性能结论；不得启动主性能 job |

## 修订格式

修改旧结论时必须新增版本，并明确写成：

```text
旧结论 → 当前结论 → 为什么改变 → 旧结论错在哪里或适用范围哪里更窄
```
