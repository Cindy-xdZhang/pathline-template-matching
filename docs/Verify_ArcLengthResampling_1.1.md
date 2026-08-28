# Verify_ArcLengthResampling_1.1

本组件实验只验证 `mainExp_TemplateMatching_2.1` 新增的弧长 primitive 数值契约，不产生流场分类性能结论。

验证输入是解析常速度场 `v=(0.5,-1.0,1.5)`，规则网格间距为 `0.1`，source-frame interval 为 `0.1`，共 13 帧。三个速度分量均非零，使正式 oracle 能检出任一坐标方向、符号或时间通道错误。1000 个 seeds 分别对应 1000 个 `dx × ds × target spatial arc length` tuple。验证程序必须证明：

1. scale table 为唯一且固定顺序的 1000 个 tuple，`scale_id` 为 `int32`；
2. 1000 个七线 primitive 全部达到目标空间弧长；
3. 每条线输出 32 个等弧长采样点，最后一个 RK4 polyline segment 在目标处线性截断；
4. `dx`、RK4 时间步长 `ds` 和目标空间弧长的物理换算与 2.1 config 完全一致；
5. 完整一次调用、137-seed 外部分批调用和 PCG64 确定性输入重排后的全部数组结果逐元素相同；
6. 零速度无法达到目标、初始邻线越界时都 fail closed。

冻结配置为 `config/Verify_ArcLengthResampling_1.1.yaml`，执行入口为 `scripts/verify_arc_length_resampling_1_1.py`。正式证据必须由干净、已提交且与 Ibex checkout 一致的 Git revision 生成，输出到新的 `outputs/Verify_ArcLengthResampling_1.1/slurm_JOBID_COMMIT12/verification.json`。主实验投递前需记录该文件 SHA-256、commit、设备和运行日志。

当前状态：`frozen_pre_run_not_run`。因此目前只能说明本地单元测试通过，尚不能声称 Ibex verification 已通过。
