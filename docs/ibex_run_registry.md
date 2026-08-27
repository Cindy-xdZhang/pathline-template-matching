# Ibex 运行与部署验证表

## Scheduler job 强制规则

1. 每个 Slurm job 提交后立即新增一行，状态先写 `QUEUED`。
2. 开始后补实际开始时间、node、GPU 型号和数量；结束后写 `COMPLETED`、`FAILED`、`TIMEOUT`、`CANCELLED` 或 `INVALID`。
3. 失败和重提交必须保留不同 job ID。
4. 每行必须包含实验版本、config、Git commit、stdout/stderr 和 output 路径。
5. “支持/反对的结论”必须可证伪；基础设施 job 不得产生方法性能结论。

| Ibex job ID | 实验版本 | 提交/开始/结束 | 状态与结果 | 支持、反对的结论 | 设备 | Git commit / config | 日志与输出 |
|---|---|---|---|---|---|---|---|
| `50930724` | `mainExp_TemplateMatching_1.1` cache-backed development | 提交 2026-08-27 17:15:27 +03:00；未开始；取消 17:18:54 | `CANCELLED`（主动取消，elapsed `00:00:00`） | 默认12 h请求预计 2026-08-30 才开始；短队列 V100 副本已运行，因此取消以避免重复计算。不产生性能证据 | 未分配节点/GPU | commit `ccc34bae4d9e683ab01b742aac6b817feb64c1ae`；`config/mainExp_TemplateMatching_1.1_development.yaml` SHA-256 `477e949f730b1e987cdff6d01f97c0e14743d8766d5a5e626de105328155b93f` | 无 job 输出；预定 stdout/stderr `slurm_logs/PTMdev11.50930724.{out,err}` |
| `50930751` | `mainExp_TemplateMatching_1.1` cache-backed development（debug 2h 替代投递） | 提交 2026-08-27 17:17:22 +03:00；开始 17:18:17；结束 17:19:36 | `FAILED`，exit `1:0`，elapsed `00:01:19`；43/43 job内测试与 CUDA matcher gate 先通过 | 首折建库在 `channel/ordinal0/lib_o025_d0125_n48` 发现 negative=228、positive=0；1.1 冻结规则要求空类 stratum fail closed，故在任何性能指标前停止。反对“现有1.1平衡规则可直接运行”，不产生方法性能证据 | `gpu510-32`；1×Tesla V100-PCIE-32GB、16 CPU、64 GB；MaxRSS `1769144K` | commit/config 与 `50930724` 完全相同 | stdout `slurm_logs/PTMdev11.50930751.out`；stderr `slurm_logs/PTMdev11.50930751.err`；保留 partial output `outputs/mainExp_TemplateMatching_1.1_development/runs/slurm_50930751_ccc34bae4d9e`（仅 input manifest/run state） |
| `50931410` | `mainExp_TemplateMatching_1.1` cache-backed development（debug P6000 替代投递） | 提交 2026-08-27 17:18:13 +03:00；开始 17:18:17；主动取消 17:18:54 | `CANCELLED`（elapsed `00:00:37`） | V100 副本同时获得资源后主动取消，避免重复计算；未产生性能证据 | `dgpu609-14`；1×P6000、16 CPU、64 GB | commit/config 与 `50930724` 完全相同 | stdout `slurm_logs/PTMdev11.50931410.out`；stderr `slurm_logs/PTMdev11.50931410.err`；partial output 若存在保留在 `outputs/mainExp_TemplateMatching_1.1_development/runs/slurm_50931410_ccc34bae4d9e` |
| `50932239` | `mainExp_TemplateMatching_1.2` cache-backed development（debug 2h） | 提交 2026-08-27 17:37:17 +03:00；开始 18:04:42；运行中 | `RUNNING`；job 内 44/44 tests 与 CUDA matcher backend gate 已通过 | 1.2 只修订空类别 library stratum：两类 template 都选0并审计；当前尚无性能结果 | `gpu510-32`；1×Tesla V100-PCIE-32GB、16 CPU、64 GB | commit `700d392b590f46a68f8ef6e973524ee0a7886c62`；`config/mainExp_TemplateMatching_1.2_development.yaml` SHA-256 `1af4bd91bcc9621570a91748c8c4bbc9493a76d17feed55ea03188742607f72f` | stdout `slurm_logs/PTMdev12.50932239.out`；stderr `slurm_logs/PTMdev12.50932239.err`；output `outputs/mainExp_TemplateMatching_1.2_development/runs/slurm_50932239_700d392b590f` |

## 非 Scheduler 部署验证

| 验证 ID | 时间 | 主机/环境 | Git commit | 命令 | 结果 | 证据文件 |
|---|---|---|---|---|---|---|
| `deploy_2026-08-27_1` | 2026-08-27 15:31 +03:00 | `login510-27` via `glogin.ibex.kaust.edu.sa`; `deepvortex` | `f202095c1572c668716a407d987fb0882add3ab6` | clone；`bash ibex/validate.sh` | `COMPLETED`：28/28 tests；library smoke；raw 5/10；cache 10/10、100/100 slices；canonical config digest 一致。只支持部署/数据访问，不支持模式匹配性能结论 | `docs/evidence/Other_ProjectBootstrap_1.1_ibex_summary.json`；远端 full report `/home/zhanx0o/pathline-template-matching/outputs/Other_ProjectBootstrap_1.1/ibex_data_access.json`; SHA-256 `609bd188229c3c7a40db2d8ea1648c517a7bd3a0ced5bccec4a0c207b714a84d` |
