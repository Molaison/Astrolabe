# Proton-PottsMPNN / pH losses 与四轨 GPU 验收

记录日期：2026-10-22。代码基线：`261b240ce85886a1eabfef80372e6b0be0076a80`；
工作分支：`codex/proton-pottsmpnn-ph-losses`。使用说明见
[pH design guide](ph_sensitive_design.md)。此页是随代码保留的精简验收记录，
不包含私有运行环境、许可软件或大体积轨迹。

> 历史记录：以下结论只适用于本文注明的原分支及当时运行，不是当前 main 的验收回执。
> 当前 checkpoint / EV6 pkl 作为外部 runtime assets 提供；见 [安装说明](proton_installation.md)。

## 结论与范围

- **工程与性能验收通过**：真实 evaluator → 联合 mutator → 重新预测 → loss →
  接受/拒绝 → 落盘；固定五对完整吞吐比较通过预声明门槛。
- **科学筛选为阴性**：正式复跑最终 pH gate 通过数 0、接受 proposal 数 0。
  每个 variant 只有 1 条不同候选序列与 1 条最终保留序列；40 条同 seed 技术复跑
  不是 40 个独立科学设计，不证明实验 pH-sensitivity 或论文逐值复现。
- 提交前审查未发现阻断性代码缺陷。核心 AF3 service/wrapper 与 Proton mutator/
  runner/daemon 六个文件均与实测 `45_persistent_pipeline/measurement_source` 逐字相同；
  随后只整理文档。独立只读检查覆盖 mutator 约束、loss 方向、iRMSD 和 overlay。

## 冻结配置

- 9TP A 链 85 aa 为 binder，B 链 87 aa 为固定**序列** target，坐标不固定；
  从已有复合物直接 refinement，不跑 DPLM。
- seed `20261020`；11 步 = 初始评估 + 10 个 proposals。AF3 原模型/精度、cuDNN
  attention、3 recycles、1 seed / 1 sample，无新增 MSA/template。
- Proton：v6 checkpoint，interface 8 Å、mutation budget 8、block size 2、
  lambda 0.3、temperature 0、禁 C；中性 `HIS-S` 为 target，`HIS-P` 为 contrast。
- 目标：原结构质量目标 `Q + 0.2 * (2 * sigmoid(ph_release / 5) - 1)`。
  `ph_release >= 0`、`ph_acceptors >= 1` 仅用于最终筛选；ph_score/iRMSD 仅诊断，
  不放宽 gate 或强制接受候选。尺度 5 REU 与权重 0.2 是工程配置，不是论文权重。
- 同一张 TY `pm-d3cc` A800-SXM4-80GB，UUID
  `GPU-1cc6821f-1722-42a8-7116-a0addb08fac0`，driver `550.54.15`；
  1 GPU、CPU affinity 0–3、`memory.max=64 GiB`。TY 未委派用户 CPU controller，
  因此不声称 `cpu.max` quota 生效。全部进程共享上述资源边界。

## 参考与缓存修复

用户明确批准以同一张 A800 的**原版 JAX 0.4.34 完整轨迹**建立新参考，再冻结后
运行资格与五对测试。没有用 candidate 选择参考，也没有放宽容差：分数绝对差
≤1e-5、原子坐标绝对差 ≤1e-3 Å，序列、决策、mask、budget、seed 精确一致。

同卡原版与补丁版初始输出完全相同，但均与旧 RTX 3090 参考不同：最大坐标差
6.943999767 Å、ipTM 差 0.0115228565、ph_release 差 4.65495976。旧差异及失败保留，
不能声称跨硬件严格等价。该初始分因检查不替代后续完整轨迹资格。

私有 PJRT 仅回移 [XLA #21149](https://github.com/openxla/xla/pull/21149) 的
`dnn_compiled_graphs: string → bytes` 修复；补丁随代码保留在
`patches/vendor/jax_cudnn_cache/01_cudnn_graph_bytes.patch`。
源码为 JAX `jax-v0.4.34`、XLA `cd6e808c59f53b40a99df1f1b860db9a3e598bff`，
构建使用 Clang 18.1.8、CUDA 12.3.2、cuDNN 9.1.1、SM80。
JAX/JAXlib/CUDA kernel plugin 仍为 0.4.34；实际运行保持原 SIF/CUDA/cuDNN。
没有修改共享环境、驱动、MPS 或 SIF。整体升级 JAX 0.5 的尝试因科学漂移被拒绝。

两个 fresh owner 的真实持久 cache 写入/命中证据通过；四条完整轨迹资格通过后才
进入正式五对。baseline 为原共享两轨分两波、每波 fresh owner；candidate 为
缓存修复后一个 fresh owner 服务四轨，最多 4 in-flight。两边保留各自自然缓存状态，
不人为清空 baseline、不排除模型 warmup；无预测或设计结果缓存。

## 固定五对结果

每个 variant 每对均完成 4 条完整 11 步轨迹。时间含启动/JIT、44 次预测与评分/CIF、
40 次 mutation/raw、排空退出及独立验证；两组直接 parity 的时间均分，排队不计入。
固定顺序 C-B、C-B、C-B、B-C、B-C，恰好五对，保留最后较慢的 baseline。

| Pair | 顺序 | Baseline 秒 | Candidate 秒 |
| --- | --- | ---: | ---: |
| 0 | C-B | 511.502065213 | 319.307745371 |
| 1 | C-B | 514.542856693 | 322.977300329 |
| 2 | C-B | 505.598954337 | 321.513568895 |
| 3 | B-C | 498.680171049 | 317.128823216 |
| 4 | B-C | 601.656519265 | 312.395792989 |

配对几何吞吐比 **1.6479530604**，整对 bootstrap 95% 区间
**[1.5766206427, 1.7825225390]**，满足下界 >1.05。
计时窗内 candidate/baseline GPU 秒比 **0.6053704389**，满足 ≤1.05 guardrail。
提交前从五份 `block.json` 独立重算几何比与 GPU 秒比，与汇总一致。
五对是最低统计 guardrail，区间仍依赖整对代表性/独立性假设。

| 诊断 | Baseline | Candidate |
| --- | ---: | ---: |
| 四轨批次中位耗时 | 511.50 s | 319.31 s |
| 合并技术轨迹吞吐 | 27.36 / h | 45.19 / h |
| 运行采样窗时间加权 GPU-util | 24.54% | 66.20% |
| 声明稳态窗时间加权 GPU-util | 40.39% | 83.53% |
| owner 加载次数 | 10 | 5 |
| 请求间空档累计 | 1003.42 s | 575.32 s |
| 跨轨 CPU-score/GPU-forward 重叠下界 | 239.01 pair-s | 520.10 pair-s |

GPU 每 2 秒采样；运行采样窗包含启动/退出，但止于独立 validator 前。
baseline/candidate 未采样的验证 E2E 窗口分别 34.49/32.29 秒，未知不记作 0。
稳态为各 owner 首个 gpu_end 到最后 gpu_end 的并集；forward 含 host/JIT，
不是纯 kernel 时间；pair-seconds 可重复计入并行 CPU 轨迹。
收益属于**缓存修复＋owner 生命周期＋四轨供料**组合，不能单项归因或宣称 GPU 满载。

正式 40 条技术轨迹共有 **440 次预测、440 行 scores、440 个 CIF、400 个原生 mutation
JSON**。全部通过原独立科学 validator、冻结参考逐步 parity 与每对直接 parity。
另有参考 1 轨、资格 4 轨，不混入正式五对。最终筛选使用最后 accepted 状态，
不把被拒绝的末次 proposal 当作最终设计。

TY 可量化 driver 窗口共 5134.46 GPU 秒；含历史累计 13835.46 GPU 秒。
少量外层 queue-wrapper 窗口未覆盖，不冒充 Slurm 账单。CPU 构建累计 49176 分配 CPU 秒。
本次审查没有新跑 GPU。原生队列 exit 0，最终记录为 5 对完成、active GPU tasks 0。

> 本文记录原分支历史验收，不是当前 main 架构或本次发布的验收回执。
> 当前源码分发不包含 v6 checkpoint / EV6 pkl；它们是独立 runtime assets。
> 当前安装契约见 [Proton 安装与来源](proton_installation.md)。

## CPU 检查与分发边界

提交前使用已有只读测试依赖与原许可容器运行以下 6 个测试文件：

```bash
rtk proxy bash logs/proton_pottsmpnn_20261020/02_run_acceptance.sh \
  tests/test_af3_service.py tests/test_ph_sensitive_losses.py \
  tests/test_proton_pottsmpnn_mutator.py tests/test_cli_run_dry_configs.py \
  tests/test_env_setup_selection.py tests/test_third_party_license_inventory.py
```

结果 **103 passed in 42.34s**，日志 `46_precommit_tests.log`。覆盖真实本地 RPC/排空、
CPU worker 隔离、真实 PyRosetta loss/iRMSD、mutator、overlay、source/license inventory。
首次误用模型 launcher 时缺 pytest；改用既有 acceptance harness 后通过，未安装新依赖。
native mask 回归另在专用 Python 3.12 中已有 4 passed；本轮不重复运行。
早期全套为 2135 passed / 9 failed / 3 skipped，9 个失败 nodeid 随后定向复验通过；
**最终 candidate 没有重新跑完整全套**，不将专项通过写成最终全套通过。
自有改动的 staged whitespace 检查通过；全量检查只报告 vendor 快照的行尾空白与
EOF 空行。涉及的 6 个文件已与 GitHub 固定 revision 原文逐字比较相同，保留上游格式，
不为消除格式警告改写 snapshot。

Proton snapshot 为固定 upstream `09682abfa7d20e0abcdeea0490b7a4b1c190aee3` 的
历史试验使用的 124 个选中文件与 provenance manifest（约 56 MB，含原许可、v6 checkpoint/EV6 folds），
其中一个文件应用明确记录的 mask-preservation patch。MIT/BSD 许可均保留。
HBPLUS 经用户确认有效许可后私有安装，源码/二进制不纳入提交；SIF、venv、PJRT wheel、
采样/轨迹及仓库外个人 watchdog 修复也不纳入提交。补丁不自动改运行环境。

## 原始证据与恢复入口

此页不是原始实验档案的替代品。档案保留于本 worktree 的忽略目录
`logs/proton_pottsmpnn_20261020/`，同份 TY 副本位于
`/sfs/home/zengzhengpeng/experiments/astrolabe_proton_ph_20261020/source/logs/proton_pottsmpnn_20261020/`。
这些路径依赖原站点与授权 runtime，不保证新 checkout 自带运行资产。

- `17_acceptance.md`：完整集成与执行历史；以其顶部 TY 最终结果为准。
- `18_gpu_trajectory.overlay.yaml`：冻结科学配置；`39_analyze_gpu_trajectory.py` 为原科学检查。
- 下列路径均相对 `45_persistent_pipeline/`：
  `08_contract.json`、`99_ty_reference_contract.json` 为合同与批准；
  `99_ty_stock_reference/`、`99_ty_cache_adoption/cache_probe.json`、
  `99_ty_qualification/qualification.json` 为参考、缓存与资格。
- `benchmark/pair_00` 至 `pair_04`：原始 block、validation、scores/raw/CIF；
  `benchmark/merged/{acceptance,costs,diagnostics,baseline,candidate}.json` 为统计。
  `05_workload.py` 为原驱动，`102_ty_finalize.py` 为 CPU 汇总，不应重复执行已接受 blocks。
- `104_ty_final_summary.json`、`105_ty_final_screen.csv`：汇总与最后 accepted 状态筛选。
  `96_ty_hardware_diagnostic/diagnostic.json` 保留旧 3090 差异。
- `101_ty_full_submission.json`：已成功终态的 owner、原生监督与恢复记录。
  持久任务 owner 为 Beads `general_hallucination_git-tzd`；commit hash 在该记录保存。
