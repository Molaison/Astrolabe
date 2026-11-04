# Astrolabe Runtime 优化报告 — 2026-11-04

## 结论与当前证据边界

**整体 PARTIAL；CPU W0 的 20% 工程目标已达到。** 在 f102 的同一 SIF 环境中，与此前通过验证的 `9d62b3e` 相比，整合后的 `f9ce87b` 在 direct / resident serial / resident adaptive 的有效 CPU evaluation 吞吐分别提高 **69.08% / 69.99% / 54.92%**。30 个 trial 全部有效，三个模式均 5/5 配对更快、耗时范围严格分离；独立审查核对全部原始科学与进程释放记录。改进复用了上轮尚未合并的 PyRosetta 延迟加载和 requirements 源码扫描优化，并保留已经验证的 SciPy 延迟导入。

这份源码提交中的报告记录 **GPU 执行前的已验证状态**。真实 pretrained GPU control、最终源码 HEAD 的 CI 和科学正控制结果，将在独立 [最终证据分支](https://github.com/Molaison/Astrolabe/tree/evidence/runtime-f102-final-20261104) 更新，以保持被验证源码不因结果文件而改变。不得把本节 CPU 数字当成 GPU 改善，也不得把计划中的 GPU 请求计为已完成工作。

代码通过 [draft PR #9](https://github.com/Molaison/Astrolabe/pull/9) 审查，不合并生产分支。实际交付目录已经建立并回读验证：`/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104`。最初服务器断连与网络 clone 超时的失败记录保留；后续通过校验过的 Git bundle 交付源码与证据，没有用本地同名目录代替服务器路径。

入口：[架构 ADR](runtime_architecture_decision.md)、[benchmark 合同](runtime_benchmark_contract.md)、[完整 backend 矩阵](runtime_backend_certification.md)、[决策记录](../evidence/runtime-next/decision.tsv)、[证据索引](../evidence/runtime-next/README.md)。

## 1. HEAD、历史工作与实际组织方式

初始 HEAD 固定为 `d66ee714f2e755948534f95d9d125287ce8dbd91`，历史 Repository Mental Model 来源为 `b537b8a`。本轮先检查当前源码、提交历史、merge-base 和 worktree；原 `261b240` 等工作线没有被覆盖。服务器恢复后进一步确认 `cloud-20261103-review/fix/opt` 是同一个 Git common directory 的 linked worktrees，并完整保存 fix 的 14 个 tracked 修改/7 个新文件、opt 的 16 个 tracked 修改/9 个新文件及其逐文件 digest。原目录保持原样，新代码只进入本任务的隔离 worktree。

使用安装的 pstack，以 `poteto-mode` 选择 `orchestrate`。主代理实际调度 A Runtime & Performance、B Evaluator & Model Lifecycle、C Producer & Scientific Correctness 三个子代理；先并行只读调查与 `architect` 比较，再冻结 workload、完成被独立复验的 identity pilot，随后以 `swarm` 分工实施。各轮 `hillclimb` 记录假设、改动、计时及 keep/revert；`interrogate` 要求可复现反例，主代理独立复验关键修复。没有使用不存在的 Cursor orch/Task 能力冒充调度。

| Owner / 范围 | 实际代码与复用 | 通过的证据及限制 |
|---|---|---|
| C / 科学状态边界 | candidate/generation identity、有限整批 prevalidation、scheduler 生命周期内 replay 拒绝 | foreign/stale/duplicate/replay 反例 red→green；不提供 durable exactly-once 或完整异步 producer |
| A / Runtime | 复用旧 fix 的绝对 deadline、独立 ingress lanes、cancel/OutcomeUnknown、CPU worker 与物理 GPU failure fencing、prepared 资源保留；修复实际 admission 后 waiting 计数泄漏 | 主代理重现旧 fix `waiting=1`、修复后为 0；真实 f102 Linux/AF_UNIX 合同 309 passed，零 skip；完整 raw 归档 |
| B / Evaluator | 精确审计 ESMFold native helper，保留 atom conversion、必要 host materialization、RNG capture 于 lease 内；纯 CPU formatting 移到 lease 外，未知 helper fallback | 原生格式函数的 CPU 数值/PDB/RNG differential、synthetic weights process contracts；没有 pretrained ESMFold GPU 认证 |
| A / 启动性能 | 已有 SciPy lazy imports，加上复用旧 opt 的 lazy PyRosetta 和按 AST UTF-8 byte offsets 提取 source | 已安装 PyRosetta 的 C 独立真实科学 gate、主代理复算、固定完整 W0；科学函数体、registry metadata/signatures 保持 |
| B + A/root review / GPU 验证入口 | 固定 ProteinMPNN public-runner control、一次性 Slurm launcher、完整身份/释放/账单检查 | child identity、矛盾 release、UUID remapping、时间/TRES 反例均先失败再通过；此处合同测试不是实际 GPU 执行 |

当前生产差异相对 d66 为 16 个文件，主要 Runtime owner 改动来自已保存的旧 fix，未重造已有机制。未改 MCMC proposal/acceptance、stage owner、repeat seeds、模型样本数、recycles 或精度来取得性能收益；没有把同一自适应轨迹的未来 proposal 伪装成独立任务。

## 2. 完整 CPU 性能对照与因果证据

固定 W0 为 Dummy evaluator + 实际 MCMC，3 repeats、seeds 17/18/19、每 repeat 3 steps。每 trial 完成 3 条轨迹和 9 次 evaluator call。科学协议 SHA256 为 `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`。四个原始测量模块保持 d66 字节；wrapper 只新增严格的基线选择和 mandatory test suites。

每次正式比较采用五组交错 A/B，三模式轮换，共 30 trials/90 trajectories/270 evaluations，900 秒 wall-time、260 MB 输出上限；不加 warmup，不混入 profiler，不删慢样本。下表为 **同一次 f102/SIF 实验 `9d62b3e` → `f9ce87b`**：

| 模式 | Baseline 中位 / 范围（s） | Candidate 中位 / 范围（s） | 有效 eval/s，前 → 后 | Aggregate 吞吐增益 | 更快配对 |
|---|---:|---:|---:|---:|---:|
| direct | 10.8418 / 9.2925–14.6258 | 6.2164 / 5.9465–8.0968 | 0.78696 → 1.33057 | **69.08%** | 5/5 |
| resident serial | 34.3922 / 33.0141–38.8537 | 19.9586 / 19.5803–23.7200 | 0.25310 → 0.43025 | **69.99%** | 5/5 |
| resident adaptive | 33.6292 / 31.6925–36.3168 | 20.0297 / 19.3712–25.8335 | 0.26328 → 0.40787 | **54.92%** | 5/5 |

Aggregate eval/s 为每 cell 的 45 个有效 evaluation 除以五次完整 workflow 时间之和，含进程启动与退出；它与 paired median ratio 是不同统计量。paired median 吞吐增益依次为 60.21% / 65.41% / 63.61%。全部 15 个配对更快，三种模式范围均分离；这不是置信区间。完整实验 wall time **657.008684340 s**，GPU 使用 0。

**保留优化，但不宣称 resident 已优于 direct。** candidate resident 耗时仍约为同版 direct 的 3.2 倍。这份短 CPU workload 的启动成本改善已超过冻结的 20% 目标，长稳态 break-even、自动 fast path 和真实 GPU 吞吐仍需各自证据。

因果诊断与正式计时分开执行。4 个 ABBA 冷进程合计 44.985457 CPU 秒：baseline 的 PyRosetta cumulative import 为 3.217/3.008 s，candidate 在不用 native loss 时没有该 import；第一次 requirements 扫描从 1.183/1.191 s 降至 0.06157/0.06172 s。Torch import 在两侧仍存在，未计作被消除的成本。两项 startup 改动组合验证，未做单独端到端 ablation，不将整体收益精确归到某一个函数。

### 原始负结果全部保留

| Attempt | 相同机器内的结果 | 决策 |
|---|---|---|
| 仅第一轮 correctness fixes | 两轮 CPU 比较范围重叠，无可靠提速 | KEEP correctness，不发放性能结论 |
| SciPy lazy imports，CI03 与最终 9d/CI04 | CI04 aggregate 增益 11.55% / 13.27% / 13.42%；均 5/5 更快，但未到 20% | KEEP 该 CPU scope；不与 SIF 绝对耗时混算 |
| 恢复 broader Runtime owner，f102 `c90fadf` vs 9d | +4.2869% / −0.1072% / +0.8966%；配对 4/5、3/5、3/5，范围重叠；811.100408536 s | KEEP correctness，**无额外性能认证** |
| ESMFold CPU output microbenchmark | 独立复验出现 +9.31% / +6.68% 耗时回退 | 仅保留生命周期/表示边界；host payload 字节减少不是实测 D2H |
| capacity-one blocking producer 原型 | 同样 3 个任务的首个选择由 ID 0 变成 ID 1 | REJECT；不改科学语义换并发 |

历史报告 9.69/32.42/33.45 s、GitHub runner 与 f102/SIF 的耗时来自不同环境，只作历史导航，未跨机器计算 speedup。所有 raw archives、失败控制和独立 reader 可通过证据索引复查。

## 3. 科学一致性与测试范围

本轮 f102 W0 全部 30 trials 科学分数/决策一致，189 个数值 tolerance 均为 0；全部已观察进程身份释放、terminal ledger 为空。原始 90 个 CSV 共 270 行。**90 行 accepted 是初始化；180 次 mutation attempts 全拒绝，accepted mutation progress 为 0**。W0 不输出结构，也不验证 acceptance/stage 正分支或 checkpoint/resume。

另有独立的真实 CLI 科学正控制：固定 seed 17/4 steps，AAAA → GAAA 在 step 1 接受并进入 `accepted_mutation` stage，之后 SAAA/AAAA 被拒绝；每版 4 次 backend evaluation、5 条 score rows、1 次 accepted mutation、1 次 stage entry。首轮错误预期 5 次 backend calls 的失败保留，修正的只是计数预期，科学配置未变。9d 最终正控制已经完成，终态 owner、Python/NumPy/Torch CPU RNG 与输出 hash exact；本次新最终 HEAD 的重跑结果将附在最终证据分支，不将旧结果冒充新 HEAD。

针对 PyRosetta lazy first-use，C 在真实 SIF/PyRosetta 2026.03+releasequarterly.5e498f 环境独立运行两进程，实际合计 30.020256 CPU 秒、GPU 未初始化。169 项完整 registry metadata/signatures、169 项 requirements、40 项 PyRosetta 分类、7 个 native symbols 全部一致；完整 27-residue 三链 fixture 的所有原子坐标、实际 SASA 和两次 registered losses 完全一致：`chain_buried_fraction=0.28873283581553477`、`com_distance=9.000000000000002`。首次 require 与科学阶段 Python/NumPy/Torch CPU RNG exact。主代理在未修改的原始 JSON 上独立重跑全部 28 个比较检查，通过。这是该 native CPU loss 的科学对照，不是 rosetta_crystal 整个 backend 或 GPU 认证。

已完成的真实 f102 process contracts 为 309 passed、零 skip，来源 60684bd。原 9d GitHub final CI 为 248 runtime +104 Torch/geometry，零 skip。子代理与主代理的聚焦测试互有重叠，不相加成总覆盖数。新 mandatory suites 覆盖 scanner、launcher、PyRosetta lazy import 和 GPU checker；需要 RDKit 的 lazy-import suite 放入已有 Torch/geometry job，保留 minimal CPU benchmark dependencies。主代理曾在 minimal CPU env 得到 90 passed/8 个缺 RDKit 的 failures，原 JUnit 保留，未以 skip 或改 benchmark 环境掩盖。

已复现并修复的边界包括 admission 后 waiting 泄漏、过期后重新 dispatch、queued timeout 清除其他 owner delegation、foreign/stale/replayed candidate，以及 GPU checker 的 child identity/release 记录矛盾。launcher 额外检查指定 UUID、UTC/standard 时间格式、有效终态日期和一致 GPU TRES；原失败案例与最终 50 单测、7 个 metadata cases、5 个 caller cases 均保留。后者是 CPU controlled-contract evidence，实际 Slurm/GPU release 必须等待真实运行。

## 4. Backend 认证与真实 GPU control

[全量矩阵](runtime_backend_certification.md) 枚举 **14 evaluator keys /13 factories、20 mutator keys /17 factories、63 variant/config scopes**。生成器核对 registry、factory owner、source digest、named evidence 与测试文件；inventory DONE。完整 registry 认证仍是 **28 BLOCKED、6 PARTIAL**，别名不另算 family，导入或 synthetic checkpoint 不等于真实 pretrained 认证。

ProteinMPNN 的实际 CPU preflight 已验证 checkpoint `proteinmpnn_v_48_020.pt`，SHA256 `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`；strict architecture load 接受 118 tensors、1,660,485 float32 parameters，零 missing/unexpected keys、零 forward/sample。4GYT 完整 A177+B177，共 354 residues，输入 SHA256 `c05c06773ae8c00d2d033f7f7a103c4048981d79418c6c1a2579b15da455b2b7`。这些均是实际资产预检，不是 GPU 推理结果。

准备的 GPU control 固定 5 seeds、10 fresh trials、每 trial 一次 public `build_runner` 和三个连续 `design_sequences`，共计划 10 次 model loads/30 次 decode；完整输入、temperature 0.3、omit C、zero-occupancy parser、全部 native kwargs、随机数消耗与 float tolerance 均冻结。比较同一源码的 legacy 与 private resident；两边原本都保留模型，不制造反复加载的弱 baseline。它只覆盖该 public-runner 的构造/完整 decode/CPU-ready output/释放，**不覆盖完整 registry selector、MCMC acceptance、shared broker、W1 evaluator、OOM/restart 或本轮 revision-attributed GPU speedup**。

ESMFold W1 的具体阻塞是：已定位的 2,097,152,000-byte checkpoint 为 invalid ZIP，所检 SIF 缺少 fair-esm/OpenFold distribution metadata 和预期 helper；没有把 metadata 缺失写成已经执行过 import failure。其它 AF/Protenix/Boltz 资产仅完成 metadata 清点，未冻结认证。没有声称服务器完全没有 pretrained assets。

## 5. Backlog 状态

| ID | 状态 | 当前边界与剩余条件 |
|---|---|---|
| R0 | PARTIAL | W0 同机冻结/全量原始证据 DONE；GPU mutator control 协议已冻结、执行身份 freeze 待完成，完整 evaluator W1/W2 尚未认证 |
| R1 | PARTIAL | 主导 startup 成本与 >20% 可重复 CPU 改善 DONE；resident/direct 差距、长稳态 break-even/auto routing 未完成 |
| R2 | PARTIAL | identity、整批校验、进程内 replay guard DONE；没有完整有界 asynchronous producer、GPU overlap 或 durable ordered commit |
| R3 | PARTIAL | ESMFold audited 边界与 CPU contracts DONE；该 evaluator 的真实 GPU 阶段/传输实测 BLOCKED |
| R4 | DONE | owner map、至少两种独立架构备选、caller-first 接口及选型 ADR；复用已有机制 |
| R5 | BLOCKED | full-scope evaluator W1/长期故障与吞吐认证未完成；真权重 CPU preflight/窄 GPU control 入口 DONE，实际 runner 结果待最终证据 |
| R6 | BLOCKED | 不在缺少 backend 证据时提高同卡并发或 microbatch 档位 |
| R7 | PARTIAL | 沿用现有身份/cache/reset；没有新增跨 residue dependency cache 的完整 stale-negative/收益认证 |
| R8 | PARTIAL | admission、三 ingress lanes、prepared ownership 和 failure isolation CPU gate DONE；GPU fairness/多 GPU 尚未认证 |
| R9 | PARTIAL | deadline/cancel/OutcomeUnknown、late/replay 与进程释放有证据；不是 restart exactly-once，pretrained OOM/crash/resume 未认证 |
| R10 | PARTIAL | W0、旧最终正控制、新 PyRosetta native parity DONE；新最终 HEAD 正控制待附，坐标型 evaluator/checkpoint 恢复未完成 |
| R11 | PARTIAL | registry/variant 清点 DONE；真实逐 backend 的完整认证仍有明确缺口 |
| R12 | BLOCKED | 新版本和异构多 GPU 部署不在现有证据上扩张 |

## 6. 资源、交付与复现

服务器目录已真实写入并回读核对，离线 bundle 有完整 SHA/先决 commit/每块校验。新 clone 曾继承 Git object alternates，已只对本任务 clone 完成 repack、保存并移除 alternate、fsck 和必需 commit 核验；旧 worktrees 未被修改。初次 clone timeout、原 wrapper 未隐藏所有 accelerator 的预检失败、工具/依赖环境错误均留存。

GPU 请求准备为一个新 Slurm job：xulab/boron/normal，1 g4090、4 CPUs、32 GiB、900 秒上限、admission 最多等待 60 秒，总上限 0.25 GPU-hour。只用 scheduler 分配的设备并在任何 CUDA 初始化前绑定已核对的 UUID；不复用用户现有 jobs，不扫描共享 GPU 来挑空卡。实际 cost、driver、UUID、whole-job terminal accounting 和 process release 以最终运行原始证据为准。**截至本源码报告冻结时 GPU 使用仍为 0。**

CPU 可复现入口见 `benchmarks/runtime/README.md` 与冻结 source manifests；GPU 命令/预算/拒绝条件见 `benchmarks/runtime/gpu_mutator_control.md` 和 `gpu_slurm_launcher.md`。所有性能/科学输出、完整 RNG codec、JUnit、环境指纹、真实配置与 raw telemetry 均以 archive 和逐文件 manifest 保存；历史证据不被新结果替换。最终报告应同时阅读精确 source HEAD、同树映射、最终 CI、实际 GPU supplement 及剩余缺口，不能只凭一个 PASS 标签扩大认证范围。
