# Pathline Template Matching

本项目研究：能否把不同邻居距离、积分步长和积分长度得到的 pathline primitive，编码为无可训练参数的 FMT 特征，并通过有标签特征库中的精确最近邻，在未见流场中识别涡区域。FMT 沿用原项目名称；原项目文档存在不同历史全称，本项目不另造展开名称。

当前仓库完成的是可复现的项目起点，不包含模式匹配性能结论。首个已预注册、但尚未完整冻结的基线是：

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
- [首个主实验预注册](docs/mainExp_TemplateMatching_1.1.md)
- [实验版本表](docs/experiment_log.md)
- [Ibex 运行表](docs/ibex_run_registry.md)
- [FMT 代码迁移来源](docs/source_provenance.md)
- [Ibex 部署与数据清单](docs/ibex_deployment.md)

本地检查：

```bash
python -m unittest discover -s tests -p test_all.py -v
python scripts/smoke_test_template_library.py
```

Ibex 数据检查：

```bash
bash ibex/validate.sh
```

该检查要求 10/10 旧 cache 全部逐 slice 通过；原始场按实际状态单独报告，当前 Ibex 预期为 5/10 可重新读取，不会把 cache 可用误写成 raw 可用。

重要边界：FMT Task5 证明的是可变尺度监督分类，不是模板最近邻；旧 Task5 的主 268 维配方还包含依赖同批样本均值的 44 维特征，不能直接用于任意单 primitive 查询。因此 `mainExp_TemplateMatching_1.1` 预注册逐 primitive 独立的 161 维 FMT 描述符。详情见项目总览。
