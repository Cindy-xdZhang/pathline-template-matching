# Verify_LongArcHorizon_1.1

状态：**`completed_passed`**。冻结配置为 `config/Verify_LongArcHorizon_1.1.yaml`；Phase A job `50997878` 与Phase B job `50998592`均在Ibex完成并通过，两个completion markers已由`mainExp_TemplateMatching_3.1` result manifest锚定。

本组件是 `mainExp_TemplateMatching_3.1` 的强制前置门禁，只验证 H48 数值契约、2000-tuple union、双 block assignment 和 train-only coverage 技术可行性，不产生模板匹配性能结论。

## Synthetic 数值与 union 检查

Phase A 不允许绕过 production 路径直接调用解析场。它必须先物化一个最小的有限 49 帧 dense velocity volume，再通过与主实验相同的 `UnsteadyVectorField3D` 和 production primitive integrator 积分；解析式只提供 expected oracle。基础常速度场固定为 `v=(0.01,0.20,0.01)`，source-frame interval 为 `0.1`，XYZ grid shape 为 `21×7×3`，bounds 为 `[-0.1,0.1]×[-1.5,1.5]×[-0.5,0.5]`，spacing 为 `(0.01,0.5,0.5)`。最大 physical `dx` 为 `2.5×0.01=0.025`，最大目标弧长为 `80×0.01=0.8`；速度模为 `0.200499376558`，在 H48 的 `4.8` 时间单位内可产生 `0.962397007477` 弧长，最大目标的解析 crossing time 为 `3.990037344431`。冻结 seed 与整个成功路径的七条线均须留在有限 dense domain 内。

为证明实现没有在 metadata 写 H48 后仍暗中按 H12 截断，还必须通过三例目标弧长 `0.48` 的慢常速度 oracle：`v=(0,0.2,0)` 在 `t=2.4` 到达，预期 valid；`v=(0,0.1,0)` 恰在 `t=4.8` 到达，预期 valid；`v=(0,0.08,0)` 要到 `t=6.0`，预期 invalid。三例在 H12 的 `t=1.2` 均未到达。另须用同一 finite dense production 路径验证 time-linear velocity `v(t)=(0,0.05+0.05t,0)`：目标 `0.48` 的解析 crossing time 为 `3.494441010849`，终点位移为 `(0,0.48,0)`，即确实使用第13帧以后、第49帧以前的时间插值。

验证必须证明：

1. `legacy_2_1` 恰为ID 0–999，全部tuple值与2.1逐项相同；把这些rows投影为2.1相同四字段后，canonical SHA-256必须为`d3577011be68ee710d42f65d70ea7791428f71297471ff0468f4980fbfc558f3`；含`block_id`的hash另存且不得冒充parent equality；`expanded_3_1`恰为ID 1000–1999；union恰为2000个无重复tuple；
2. H48和49帧窗口进入primitive/cache身份，H12 cache必须fail closed；
3. 2000个七线constant-velocity primitives都通过 finite dense production field/integrator 达到目标，输出32个等弧长点，并在最终segment上精确截断；
4. dx、RK4 ds、arc length和maximum time的换算与3.1 config一致；
5. single call、external batches和输入重排的结果在冻结容差内一致；
6. zero velocity、boundary exit和H12身份错配都fail closed；
7. after-H12/before-H48、exact-H48、beyond-H48 三个慢常速度边界oracle分别得到 valid、valid、invalid；
8. time-linear velocity 的 crossing time 与终点在冻结容差内匹配解析值，证明13–49帧区间被实际读取。

## 双 block assignment 检查

每个source只有同一个endpoint-inclusive `40³=64,000` center-coordinate grid，不得生成或声称一个`40×40×80`坐标网格。共享center domain按两个block的最大dx `2.5`内缩。

每个center产生两行：old block使用`PCG64(15068)`并逐项复现2.1的seed-index→scale-ID mapping；每个source的legacy assignment array按项目canonical-array规则计算后必须等于`21cdb937f57baf1a786a6a4622870e234074b684e5a5cda4c4271837631e0fee`，不能用裸`tobytes` hash替代。new block使用事先冻结的独立`PCG64(35068)`。每个block的每个scale必须恰有64 rows，因此每source总计128,000 rows。同一`seed_index`在两个block中重复是预期行为，必须以`block_id`区分。

## Train-only coverage 技术门禁

诊断只允许读取八个train datasets及其49帧窗口。`tangaroa`和`smokeBuoyancy`的文件、manifest、cache、label、validity、coverage、feature、prediction和metric全部禁止打开。

诊断必须保存每个`dataset×source×block×scale`的assigned、valid、invalid、coverage及正负候选计数，包括zero-valid和single-class strata。通过条件为：

- expanded block的10个arc-length levels各自在全部train dx/ds/dataset/source汇总后至少有1个valid primitive；
- 按3.1冻结library规则，expanded block全局至少有1个selected positive train template和1个selected negative train template；
- 所有tuple和stratum都完整报告；不得静默丢弃低coverage或空类记录。

这是技术门禁，不允许据此删除tuple、改H48、改尺度范围、调descriptor、library、normalization、distance、score、threshold或metric。若失败，应保存失败证据、禁止3.1主评测，并为任何修改创建新版本。

## 输出与运行纪律

正式运行必须来自同一个 clean committed Git revision，并严格分为两个不可互换的 Slurm phases；两者使用同一份冻结 Verify/main configs，但各写入新的WekaFS run directory。

Phase A `synthetic` 不读取任何真实流场，先验证 finite dense production field/integrator、2000-tuple union、assignment 和 H48 数值边界：

```text
/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/synthetic/slurm_JOBID_COMMIT12
```

Phase A required outputs 是两份 frozen configs、`synthetic_verification.json`、`scale_union_manifest.json`、`assignment_verification.json`、`environment_versions.json`，并在这些文件全部落盘后最后写 `SYNTHETIC_PASS.json`。只有有效 marker 及其 file SHA-256 已记录，才可 stage 49帧 train windows；随后必须实际加载并校验8个train manifests和32个windows的size/file SHA-256，写出`TRAIN_PORTABLES_PASS.json`后才可构建恰好 `8 datasets×4 sources=32` 个 train cache shards。此时仍禁止构建或打开 test window/cache。

Phase B `train_coverage` 只能读取这32个 immutable train caches及sidecars、冻结configs与Phase A marker；禁止读取raw fields或任何test材料：

```text
/ibex/user/zhanx0o/pathline-template-matching/mainExp_TemplateMatching_3.1_development/verification/Verify_LongArcHorizon_1.1/train_coverage/slurm_JOBID_COMMIT12
```

Phase B required outputs 是两份 frozen configs、`train_cache_input_manifest.json`、完整 `train_only_coverage_diagnostics.csv`、`train_only_coverage_summary.json`、`environment_versions.json`、记录 Phase A marker SHA-256 的最终 `verification.json`，并最后写 `TRAIN_COVERAGE_PASS.json`。只有两个phase markers都存在、两阶段记录同一configs与Git commit hashes、且Phase B记录 Phase A marker SHA-256时，`Verify_LongArcHorizon_1.1`才可判定通过。每个Slurm job都必须立即登记到`docs/ibex_run_registry.md`，但作业数不等于cache shard数；冻结要求是产出恰好32个train cache shards。

## 已完成证据

| Phase | Slurm job | Device | Result |
|---|---:|---|---|
| A: synthetic | `50997878` | `cn604-15`, CPU, 16 CPU, 32GB | `COMPLETED 0:0`, 55s；102/102项目测试通过；2000/2000 constant-field primitives valid；after-H12/before-H48、exact-H48、beyond-H48分别为valid/valid/invalid；time-linear oracle通过 |
| B: train coverage | `50998592` | `cn604-14`, CPU, 16 CPU, 64GB | `COMPLETED 0:0`, 31s；只读32个train caches；64,000个`dataset×source×block×scale` strata完整报告；expanded十个arc levels均有valid；全局正负selected templates均非空 |

冻结证据身份：

- numerical Git commit：`260a07ad380d64fc300cabe8926244e92d8ba04a`；
- Verify config SHA-256：`222caf2e70e099f5d4fd50cb8d9489781b5825b574f9abd77ff4ed96298205d9`；
- main config SHA-256：`771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e`；
- Phase A marker SHA-256：`002887a4e0a354e5e8a287b1782fa13914660252353f92445061857b271f35c4`；
- Phase B marker SHA-256：`1792d08f344a630ec1205e71617eb308b4b36ffe2338a6789b24e8d5f07f1981`；
- Phase B `verification.json` SHA-256：`19d1da9046cd8f91b1a882459e92ecf7b51ecf726af367c323d16ebd5fc9480a`。

Expanded block按arc level汇总的valid train counts依次为：`13→189,880`、`20.4444→183,657`、`27.8889→126,185`、`35.3333→90,996`、`42.7778→79,825`、`50.2222→75,141`、`57.6667→71,881`、`65.1111→68,021`、`72.5556→63,087`、`80→58,525`。按冻结library规则，legacy选中negative/positive各30,052，expanded各18,028。Phase B没有打开Tangaroa或Smoke文件，不产生性能结论；其唯一结论是H48和2000-tuple配置通过了预注册技术门禁。
