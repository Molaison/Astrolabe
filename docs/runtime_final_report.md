# Astrolabe Runtime 增量交付报告 — 2026-11-04

## 结论与交付范围

**整体状态：PARTIAL。** 本轮交付了实际生产代码、三代理独立调查与交叉审查、可复现 CPU 端到端对照、科学正控制和完整 backend inventory。R1 在固定 workload 上取得可重复的 CPU 吞吐改善；冻结的 20% 工程目标尚未达到。真实 pretrained GPU 认证、GPU 吞吐改善和目标服务器目录写入均为 **BLOCKED**。

代码以 [draft PR #9](https://github.com/Molaison/Astrolabe/pull/9) 叠加在用户指定分支上。初始 HEAD 为 `d66ee714f2e755948534f95d9d125287ce8dbd91`，已有 Repository Mental Model 的来源为 `b537b8a`；调查时将历史模型与当前源码逐项区分。原有 worktree 中的未提交资源文件保留。最终代码 HEAD 为 `9d62b3eb86cb75a2266af49dc48cb851e69e0341`，已完成完整 CI 和独立科学正控制复验。最终验证证据保存在 `evidence/runtime-final-validation-20261104` 分支，PR 的已验证代码 HEAD 保持固定。

主要入口：[架构 ADR](runtime_architecture_decision.md)、[冻结 benchmark 合同](runtime_benchmark_contract.md)、[backend 矩阵](runtime_backend_certification.md)、[证据索引](../evidence/runtime-next/README.md)、[hillclimb 决策记录](../evidence/runtime-next/decision.tsv)。

## 最终 HEAD 复验

**DONE（CPU 验证范围）**：[最终 CI 41766956074](https://github.com/Molaison/Astrolabe/actions/runs/41766956074) 在 `9d62b3e` 完成 248 项 runtime 回归、104 项 CPU Torch/结构测试及 30/30 固定 W0 trials，零 skip。Python 3.11/3.12 的 push 和 PR 安装验证全部通过；主代理在该 HEAD 另跑 backend matrix 4 项检查，以及真实 CLI acceptance/stage/RNG 正控制，全部通过。

| 模式 | Baseline 中位耗时（s） | 最终 HEAD 中位耗时（s） | 中位耗时降低 | 总有效 eval/s 提高 |
|---|---:|---:|---:|---:|
| direct | 2.9347 | 2.6723 | 8.94% | 11.55% |
| resident_serial | 7.5480 | 6.7115 | 11.08% | 13.27% |
| resident_adaptive | 7.6069 | 6.6627 | 12.41% | 13.42% |

比较总 wall time 172.18 s，在原 900 s 预算内。三个模式再次各 5/5 组更快、前后范围严格分离；两轮 R1 阶段 CI 共 60 trials，每轮只在本轮 runner 内计算前后比率。最终 resident serial/adaptive 的有效 eval/s 分别为 1.1898→1.3477、1.1816→1.3402；这些 CPU 改善仍低于 20% 目标，resident 仍约为 direct 的 2.5 倍耗时。

最终 W0 再次包含 90 trajectories、270 evaluations、180 rejected mutation attempts；accepted 初始化不计进展。独立科学正控制再次出现一次 accepted mutation 和一次 stage transition，rows、终态 owner、Python/NumPy/Torch CPU RNG、输出 hash 精确相同。主代理与 C 分别从最终原始文件复核；[最终验证记录](../evidence/runtime-next/final/README.md) 给出源码、CI、raw archive 与复现入口。正文第 2 节保留首次 R1 对照，便于追踪两轮证据；其绝对时间未与本轮相除。

## 1. 实际实施与审查

采用已安装 pstack 的 `poteto-mode` 选择 `orchestrate`；主代理负责 workload 冻结、最小 pilot、集成与独立复验，实际启动三个角色代理。三者先只读调查并比较架构，pilot 独立验证后才进入 `swarm` 并行修改。每轮记录 hypothesis、实现、验证和 keep/reject，失败记录随证据交付。`interrogate` 交叉审查发现的真实缺陷均先复现再修复。

| Owner | 生产改动 | 独立验证与结论 |
|---|---|---|
| C：Producer & Scientific Correctness | worker 回复必须匹配提交的 `(candidate_id, generation)`；整批 generation 校验先于 dispatch；同 generation 的重复及完成/失败后的 replay 拒绝 | B 与主代理独立验证 pilot；A 与主代理验证 batch/replay 边界。已有合法任务结果和跨 generation 同 ID 保持可用。KEEP correctness |
| A：Runtime & Performance | 一个 admission deadline 覆盖 pending bytes、compute、worker lock 与 entry bookkeeping；仅持有 worker call lock 的请求可清理 delegation | C 与主代理独立重现并验证三项缺陷。过期请求不会重新获得 dispatch budget；排队超时不会破坏正在执行的请求。KEEP correctness |
| B：Evaluator & Model Lifecycle | 对精确审计的 ESMFold native helper，atom conversion、所需 CPU materialization 与 RNG capture 留在 lease 内；纯 CPU PDB formatting 移到 lease 外；未知 helper 保守 fallback | C：63 项聚焦测试、36 组独立差分与 4 个 fallback 控制；主代理复验边界。实际 CPU process CI 含 synthetic weights。KEEP 生命周期/表示正确性，性能未认证 |
| A：R1 | `interaction`、`repeat_symmetry`、`design_screening` 的 SciPy spatial 依赖延迟到实际几何调用 | B：67 项测试、77 个独立真实 SciPy 数值/异常案例、169 registry metadata/signatures 和 CPU RNG 一致；主代理独立 39 项及源码/缺失依赖检查；真实 W0 CI 通过。KEEP 已测 CPU 改善 |

生产差异限于 8 个文件，240 行新增、30 行删除；计算公式、MCMC proposal/acceptance、stage owner、repeat seeds、样本数、recycles 与精度设置均未为性能调整。单轨迹仍按前一步结果顺序推进，独立 repeats 使用原有进程隔离。没有引入第二个科学状态 writer、通用 actor 框架或未经验证的模型缓存。

两个重要限制：admission deadline 防止过期后 dispatch，并非硬实时返回保证，bookkeeping/cleanup 仍可能等待；candidate identity 集合是 scheduler 生命周期内的内存去重，不提供 durable exactly-once 或完整恢复保证。

## 2. R1：真实 CPU 性能结果

W0 沿用 `runtime_cpu.yaml`：Dummy evaluator、实际 MCMC、3 repeats、seed 17/18/19、每 repeat 3 steps。每 trial 为 9 次真实 CPU evaluator call。科学投影 SHA-256 为 `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`；原有 4 个测量模块和 fixture 均与 d66 逐字节一致。

正式比较为同一 runner、同一 Python/dependency 环境、5 组交错 baseline/candidate、3 个模式，共 30 trials；轮换模式、交替 AB/BA，不加入 scientific warmup。900 s 比较预算和 260 MB 输出预算在修改前冻结。下表来自 [CI 41764768462](https://github.com/Molaison/Astrolabe/actions/runs/41764768462)，实际比较总耗时 123.16 s、GPU 使用 0。主代理从原始 CSV/provenance/trace 复算，C 再独立核对。

| 模式 | Baseline 中位耗时 / 范围（s） | Candidate 中位耗时 / 范围（s） | 中位耗时降低 | 有效 eval/s，前 → 后 | 总有效吞吐提高 |
|---|---:|---:|---:|---:|---:|
| direct | 2.0261 / 2.0186–2.2950 | 1.8798 / 1.8484–1.9530 | 7.22% | 4.3165 → 4.7783 | 10.70% |
| resident serial | 5.4685 / 5.3690–5.5285 | 4.7865 / 4.7416–4.8503 | 12.47% | 1.6513 → 1.8790 | 13.79% |
| resident adaptive | 5.4677 / 5.3627–5.5632 | 4.7920 / 4.7309–4.9220 | 12.36% | 1.6470 → 1.8726 | 13.70% |

有效 eval/s 使用每组 45 次通过科学与 release gate 的 evaluation 除以五次 workflow 总 wall time，包含 startup/shutdown；与中位耗时比率是不同统计量。三个模式均 5/5 paired improvement，前后范围各自严格分离。serial paired speed ratio 为 1.1197–1.1512、adaptive 为 1.1191–1.1759；这些是观察范围，不是置信区间。样本仅覆盖这份 CPU workload 和该 runner，不能外推所有模型/机器。

对照比较的是整合 candidate 与 d66，包含此前正确性及生命周期修改，未逐 commit 做 ablation；全部收益不能单独归因于三处 lazy import。

**20% 目标未达到，resident 仍约为同版 direct 的 2.55 倍耗时。** 本轮保留超过运行波动、科学检查全绿的增量改善；R1 大目标继续标为 PARTIAL。没有通过绕过 broker 的资源监控来制造 fast path。历史 9.69/32.42/33.45 s 与本轮机器不同，仅用于归因导航，未用于计算 speedup。

之前两轮仅含正确性修复的同机对照分别为 `6620da6` 和 `db937f9`，范围重叠、没有可信性能收益，原始数据完整保留。ESMFold CPU microbenchmark 在独立复验中还出现 +9.31% / +6.68% 的耗时回退。其请求 host payload 的 96.75% / 99.05% 减少是 synthetic CPU tensor 字节统计，不能当成实测 D2H 或 GPU 吞吐改善。

[R1 round2](../evidence/runtime-next/agent-a/r1-round2/REPORT.md) 又做了限定只读 cold-path 调查：剩余 Hydra/NumPy/FileLock 是实际需要的依赖；唯一可动的 Bio.SeqIO 有部分依赖随后仍被 Bio.PDB 复用，不能把累计 importtime 全算成可省时间。3 次有界冷调用包含一次解释器路径错误，新增 scientific evaluation 为 0。没有发现足够大的低风险候选，故停止扩大生产改动、冻结当前源码；20% 缺口作为负结果保留。

## 3. 科学一致性证据

W0 的 30/30 trials、90/90 trajectories、270/270 evaluations 全部成功；各 seed 的原始 score/proposal/acceptance/stage CSV 完全相同，所有已观察进程身份均确认 release。**90 条 accepted 初始化不计为科学进展：180 次 mutation attempts 的 accepted 数为 0。** W0 不输出结构，因此未提供 coordinate parity，也不能证明 acceptance/stage 正分支或 checkpoint/resume。

为覆盖正分支，另行冻结了 `cpu_scientific_control.yaml`（SHA-256 `ddbd295188ed22241736750bed4fa613c553dbf29b5423de0f9454b7c90cbde9`）。真实 direct CLI 执行相同 seed 17、4 steps：AAAA → GAAA 在 step 1 接受，进入 `accepted_mutation` stage；后续 SAAA/AAAA 拒绝。每个版本 3 次 mutation attempts、4 次 backend evaluation、5 条 score rows、1 次 stage entry。重复 stage scoring 复用相同 evaluator 的已有结果，所以不是第 5 次 backend evaluation。

C 首轮误将期望调用数定为 5，gate 报 PARTIAL；原始失败保留。源码核对后仅修正计数预期，workload、seed、steps、预算和科学输出均未改变，重新 freeze/run 通过。主代理在包含全部生产修改的集成源码上独立再跑：全部 rows、终态 scientific owner、Python/NumPy/Torch CPU RNG 与 scientific output hash 完全一致。完整 RNG 状态保存在无损归档中。这个控制没有修改生产科学函数，也没有加入 W0 的性能样本；终态快照未被描述为 resume 认证。

R1 的正常依赖环境保持全部 169 loss entries 和公共签名。缺失 SciPy 的故障环境中，延迟失败使可见 registry 由 117 增至 169；实际几何调用仍不可执行。三个 design-screening 入口保持原 ImportError 类型和文本，增加显式 dependency cause。此处失败时点/可见性的变化是已知兼容性限制，详见 [B 的独立审查](../evidence/runtime-next/agent-b/review-a-r1-report.md)。

## 4. 测试、故障与保留的负结果

`71c7592` 的实际 GitHub Actions 完成 **248 项 runtime 回归、104 项独立 CPU Torch/结构测试，零 skip**，并完成上述 30 次端到端运行。不同角色的 39/67 等聚焦复验包含重叠测试，未相加包装成覆盖数。CPU Torch 版本为 2.7.1+cpu、CUDA build 为 null；其中 fake-weight resident process 的通过只覆盖相应 contract。

独立故障复现包括 foreign candidate/future generation、generation mismatch、批内 duplicate、完成或失败后的 replay、入队/compute/worker-lock 等待耗尽 deadline、active delegation 被 queued timeout 清除、bookkeeping 吞掉原 budget、未知 ESMFold helper 的 fallback。现有 OutcomeUnknown/release/RNG/stage 契约继续由回归门禁覆盖；真实 pretrained OOM/crash 后的恢复仍待设备认证。

拒绝的方案和失败尝试可在 decision.tsv 找到：blocking producer prototype 在相同 3 个任务下把首个选择从 ID 0 改为 ID 1；无 broker 的 direct shortcut 会丢失 allocation-derived monitoring；R1 最初只改两个文件仍加载 spatial，第三个入口由实际 import stack 找到；缺 RDKit/Torch 的测试 collection 失败按环境失败保留；初始 CI 的 runner context 和 fixture child import 失败也保留。没有将失败/不完整 trial 用作更快的分母。

本地执行器禁止 AF_UNIX 且 PID API 与 `/proc` 不一致，guard 保持启用。真实 broker/guardian E2E 因而转到实际可用的 GitHub Linux runner；本地 mock/contract 测试没有被替代标注为真实进程或 GPU 成果。

## 5. Backend 全量矩阵

完整矩阵按实际 registry key + factory + variant + configuration 生成：**14 evaluator keys / 13 factories，20 mutator keys / 17 factories，共 63 个明确 scope**。覆盖清点 DONE，认证仍有 **28 个 BLOCKED、6 个 PARTIAL** registry-key records；没有因 import 成功、别名或 code wrapper 而发放 pretrained 认证。

矩阵保留 Boltz1/非 Boltz2、Mol-1 独立家族及 aliases、Protenix requested/effective constraint variant、Chai offload/device residency、NA-MPNN seed 0 one-shot、four far-apart aliases 实际 ProteinMPNN factory 等区别。每项列出需要冻结的 source/checkpoint/config、已有证据级别和未运行的 numerical/residency/cache/fault/throughput gates。

| Scope | 已获得证据 | 尚缺 |
|---|---|---|
| Dummy + MCMC 的精确 W0 / 正控制 | 真实 CPU workflow、科学与 release gate、重复吞吐比较 | 其它科学配置和结构输出覆盖 |
| ESMFold native output split | 真实 CPU tensors、固定 native helper、PDB/score/RNG differential、fake-weight resident process | 真实 pretrained inference、CUDA boundary/transfer、冷/热/连续请求与 OOM/restart |
| 其余 learned evaluator/mutator variants | 实际注册/调用边界/版本配置清点；已有源码及各类 contract tests 按 scope 引用 | 独立设备与本地权重上的逐变体完整认证 |

## 6. Backlog 状态

| 项 | 状态 | 本轮到达的边界 / 剩余条件 |
|---|---|---|
| R0 | PARTIAL | CPU freeze、实际 benchmark/原始证据 DONE；W1 pretrained/GPU workload 尚无获批资源 |
| R1 | PARTIAL | 归因、实际修复、可重复 CPU 改善 DONE；20% 目标与 resident/direct 差距未解决 |
| R2 | PARTIAL | identity、整批 admission、去重与科学 owner 边界 DONE；通用有界异步 producer/背压/取消 E2E 未完成 |
| R3 | PARTIAL | audited ESMFold host/device representation 边界实现与 CPU contract DONE；GPU pilot BLOCKED |
| R4 | DONE | 当前 owner map、caller-first 接口、三个架构备选及 ADR；沿用已有生命周期机制 |
| R5 | BLOCKED | 缺真实 GPU allocation、pretrained assets 和可连接的服务器执行环境 |
| R6 | BLOCKED | 没有 R5 证据，不开启同卡并发/microbatch 档位 |
| R7 | PARTIAL | 沿用已有 cache/identity/reset 机制并清点覆盖；新增 dependency cache 的 stale-negative/真实收益尚未认证 |
| R8 | PARTIAL | admission/delegation/deadline 缺陷修复与 CPU release 通过；动态多 GPU/fairness 吞吐认证未完成 |
| R9 | PARTIAL | timeout、late/foreign/duplicate reply 和现有 OutcomeUnknown 契约验证；完整 pretrained OOM/crash/resume 未认证 |
| R10 | PARTIAL | W0 精确一致性、真实 CLI acceptance/stage/RNG 正控制 DONE；坐标及 checkpoint 恢复覆盖未完成 |
| R11 | PARTIAL | 全部 registry/63 scopes 矩阵 DONE；真实逐 backend 认证受 R5 阻塞 |
| R12 | BLOCKED | 后续新版本/异构多 GPU 部署依赖 R5–R11，未虚列完成 |

## 7. 资源与目标路径

当前环境无 `/dev/nvidia0`，安装的是 CPU-only Torch；未配置 pretrained checkpoint，也没有共享 GPU allocation 授权。本轮使用 0 GPU-hours。目标 `/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104` 不存在于当前 mount，且获准检查的 native executor 报 disconnected，目标目录写入 **BLOCKED**。没有建立一个同名本地空目录冒充服务器交付。

实际工作及证据在 `/workspace/scratch/1a0b1639644d/cloud-20261104`，审查代码和全部归档通过 PR 持久交付。资源恢复后，按矩阵先选已支持的精确 pretrained variant，冻结 checkpoint/source/config，获得独立 allocation 后执行连续请求、数值/科学 parity、restart/OOM 和至少 5 组交错吞吐对照；不得用本报告 CPU 数字升级 GPU 状态。

## 8. 复现入口

CPU 环境、固定命令与 900 s/260 MB 预算见 [benchmarks/runtime/README.md](../benchmarks/runtime/README.md)。原始 archive 的 SHA-256 在 `evidence/runtime-next/archives.sha256`；先核对再解压。各代理 archive 还提供逐文件 SHA manifest。科学正控制采用独立输出目录：

```bash
python scripts/validation/runtime_scientific_control.py freeze \
  --baseline /absolute/baseline-d66 --candidate /absolute/candidate \
  --output /absolute/new-scientific-control
python scripts/validation/runtime_scientific_control.py run \
  --plan /absolute/new-scientific-control/plan.json
python scripts/validation/runtime_backend_certification.py \
  --output /absolute/backend-certification.json \
  --markdown /absolute/backend-certification.md
```

故障 probe、固定 native fixture 的 source pins/licences、计时原始 stdout/stderr、完整配置、provenance、JSONL traces、telemetry、JUnit、科学 CSV 和独立审查记录都在证据索引中；范围外的项目保持 PARTIAL/BLOCKED。
