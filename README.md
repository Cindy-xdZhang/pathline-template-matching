# Pathline Template Matching

本项目研究：能否把不同邻居距离、积分步长和积分长度得到的 pathline primitive，编码为无可训练参数的 FMT 特征，并通过有标签特征库中的精确最近邻，在未见流场中识别涡区域。FMT 沿用原项目名称；原项目文档存在不同历史全称，本项目不另造展开名称。

当前已完成 raw-flow-backed `mainExp_TemplateMatching_3.1` exposed-development 主实验。Ibex job `50999189` 在数值 commit `260a07ad380d64fc300cabe8926244e92d8ba04a` 上，以8个完整流场建库、2个完整流场测试，运行H48/49帧、2000个 `(dx, RK4 ds, target spatial arc length)` 尺度、四方法对照、5000次配对source-timeslice bootstrap和4张固定`dataset×scale block`三联图；formal confirmation未运行。当前流程是：

```text
3D velocity field
→ variable-scale 7-line primitive
→ fixed [7,32,4] (x,y,z,t) tensor
→ [7,32,3] xyz descriptor view
→ Task5-cache-compatible independent 161D training-free FMT
→ library-only standardization
→ exact Euclidean one-nearest-neighbor
→ vortex / non-vortex
```

必须先读：

- [项目总览](docs/project_overview.md)
- [唯一研究协议](docs/research_tasks_and_protocol.md)
- [当前3.1实验与结果](docs/mainExp_TemplateMatching_3.1.md)及[结构化证据](docs/evidence/mainExp_TemplateMatching_3.1_ibex_summary.json)
- [2.1的1000尺度结果](docs/mainExp_TemplateMatching_2.1.md)与[1.1失败记录](docs/mainExp_TemplateMatching_1.1.md)
- [实验版本表](docs/experiment_log.md)
- [当前模板匹配失败的第一性原理诊断](docs/first_principles_template_matching_analysis.md)
- [当前PerScale方法的四流场三联图与分类表](docs/Other_PerScaleNegativeMetricVisualization_1.1.md)
- [Ibex 运行表](docs/ibex_run_registry.md)
- [FMT 代码迁移来源](docs/source_provenance.md)
- [Ibex 部署与数据清单](docs/ibex_deployment.md)

本地检查：

```bash
python tests/test_all.py
python scripts/smoke_test_template_library.py
python scripts/smoke_test_development_pipeline.py
```

Ibex 数据检查：

```bash
bash ibex/validate.sh
# 3.1 必须严格按 Phase A、train portable、train cache、Phase B、
# all portable、test cache、evaluation 的门禁顺序执行，见实验文档。
```

首次Ibex clone时只确认5/10 raw可直接从集群读取；3.1随后在同一冻结commit下由Ibex和Windows分别生成并汇合10个数据集的40个49帧portable windows，10/10 manifests及40/40 files均实际加载并通过size/SHA-256门禁。旧cache可用、raw源可访问和新portable/cache可重建仍分别报告。

3.1的观察结果是混合的：在两个已暴露test families的source-timeslice等权宏平均上，FMT161的Accuracy/Average Precision/F1为`0.6041/0.3621/0.3787`。相对Raw672，FMT的Accuracy、Average Precision、F1、balanced accuracy、Area Under the Receiver Operating Characteristic Curve和precision差值区间均高于0，但recall更低；相对强Raw-PCA161，FMT的Average Precision、F1、Area Under the Receiver Operating Characteristic Curve和recall更差。Smoke buoyancy的expanded-block coverage仅`0.5727%`，arc length `80 h_min`为0个有效query。不能宣称FMT或长弧扩展普遍优越；完整表、反例和证据哈希见[3.1实验文档](docs/mainExp_TemplateMatching_3.1.md)。

重要边界：FMT Task5证明的是可变尺度监督分类，不是模板最近邻；旧Task5的主268维配方还包含依赖同批样本均值的44维特征，不能直接用于任意单primitive查询。因此本项目主实验固定使用逐primitive独立的161维FMT描述符。详情见项目总览。
