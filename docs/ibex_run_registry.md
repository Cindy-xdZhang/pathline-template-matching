# Ibex 运行与部署验证表

## Scheduler job 强制规则

1. 每个 Slurm job 提交后立即新增一行，状态先写 `QUEUED`。
2. 开始后补实际开始时间、node、GPU 型号和数量；结束后写 `COMPLETED`、`FAILED`、`TIMEOUT`、`CANCELLED` 或 `INVALID`。
3. 失败和重提交必须保留不同 job ID。
4. 每行必须包含实验版本、config、Git commit、stdout/stderr 和 output 路径。
5. “支持/反对的结论”必须可证伪；基础设施 job 不得产生方法性能结论。

| Ibex job ID | 实验版本 | 提交/开始/结束 | 状态与结果 | 支持、反对的结论 | 设备 | Git commit / config | 日志与输出 |
|---|---|---|---|---|---|---|---|
| `50930724` | `mainExp_TemplateMatching_1.1` cache-backed development | 提交 2026-08-27 17:15:27 +03:00；开始/结束待更新 | `QUEUED` | 待运行；只允许产生 exposed-development 描述性证据，不是 sealed confirmation | 待调度；请求 1×A100 或 V100、16 CPU、64 GB | commit `ccc34bae4d9e683ab01b742aac6b817feb64c1ae`；`config/mainExp_TemplateMatching_1.1_development.yaml` SHA-256 `477e949f730b1e987cdff6d01f97c0e14743d8766d5a5e626de105328155b93f` | stdout `slurm_logs/PTMdev11.50930724.out`；stderr `slurm_logs/PTMdev11.50930724.err`；output `outputs/mainExp_TemplateMatching_1.1_development/runs/slurm_50930724_ccc34bae4d9e` |

## 非 Scheduler 部署验证

| 验证 ID | 时间 | 主机/环境 | Git commit | 命令 | 结果 | 证据文件 |
|---|---|---|---|---|---|---|
| `deploy_2026-08-27_1` | 2026-08-27 15:31 +03:00 | `login510-27` via `glogin.ibex.kaust.edu.sa`; `deepvortex` | `f202095c1572c668716a407d987fb0882add3ab6` | clone；`bash ibex/validate.sh` | `COMPLETED`：28/28 tests；library smoke；raw 5/10；cache 10/10、100/100 slices；canonical config digest 一致。只支持部署/数据访问，不支持模式匹配性能结论 | `docs/evidence/Other_ProjectBootstrap_1.1_ibex_summary.json`；远端 full report `/home/zhanx0o/pathline-template-matching/outputs/Other_ProjectBootstrap_1.1/ibex_data_access.json`; SHA-256 `609bd188229c3c7a40db2d8ea1648c517a7bd3a0ced5bccec4a0c207b714a84d` |
