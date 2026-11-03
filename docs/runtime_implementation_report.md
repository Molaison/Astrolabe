# Astrolabe 统一执行机制：实现与验收报告

## 结论和证据边界

本分支已经落地公共资源 owner、后端生命周期接口、实际阶段边界、科学请求身份、
自适应在途控制、失败释放协议，以及固定工作量的三模式对照入口。
改造覆盖实际 registry 中的 14 个 evaluator key（13 个 family）和 20 个 mutator key。
精确 variant、外部环境、embedding 和子进程边界分别列在
[evaluator 矩阵](evaluator_lifecycle_matrix.md)与[mutator 矩阵](mutator_lifecycle_matrix.md)。

**当前不能声称完成预训练 GPU 后端认证，也不能声称已经获得真实 GPU 吞吐提升。**
执行环境没有继承 GPU 作业分配，且没有确定逐后端权重/版本、科学 workload 和实验预算。
所有本轮远端测试都显式隐藏 GPU；真实进程、CPU 模型机制测试与 GPU 科学认证分开记录。
本报告以实际可复现结果说明实现边界，不以 daemon 存活、mock CUDA 或 busy 数值代替验收。

最终 CPU 三模式对照九轮全部通过，固定的 **27/27 个轨迹**完成且评分/决策一致。
该配置下，旧直连、常驻串行、常驻自适应的中位端到端耗时分别为 **9.69、32.42、
33.45 秒**；新模式没有提速，CPU 与内存成本增加。两次实验合计 **445.12 秒**，
包括首轮失败；不能把这一 CPU 结果外推为任何 GPU 后端的收益或退化结论。

## 1. 保留的基线和实施范围

| 对象 | 固定身份 / 处理 |
| --- | --- |
| 两份输入文档 | `b537b8a81ff0738922e80a997ffeb9700cdafe09` |
| 公开实现分支的检查点 | `7a0e2ef43ac15229ab7024ddc88fda00a5adb56a` |
| 实际继承的后续实现 | `834402daade9f62e9dec0b08041dade163a89369`，包含尚未推送的后续提交 |
| 本轮工作分支 | `feat/runtime-resource-lifecycle-20261103` |
| 对照旧 checkout | 单独冻结到 `834402daade9f62e9dec0b08041dade163a89369`，不从当前工作树伪装旧提交 |
| 另一条 `261b240` 工作线及其未提交改动 | 未重置、未覆盖、未合并；最终目标分支由用户选择 |
| 原有资源下载临时文件 | 原样保留，不加入补丁或提交 |

输入文件 SHA-256、实际 registry 到 factory 的映射与相关说明见
[implementation-inventory.json](../evidence/runtime-20261103/implementation-inventory.json)。
保留提交列表见 [preserved-commits.txt](../evidence/runtime-20261103/preserved-commits.txt)。
`initial-status.txt` 在三个代理开始工作后采集，其中任务产生的改动不能当成原有用户改动。
本轮由 runtime、evaluator、mutator 三个真实子代理并发实施；主代理负责科学边界集成、
独立验收入口、跨组件复验和证据交付。

Mol-1 已在继承基线中接入，属于本轮现有后端改造。Boltz 的实际 loader 仍是 Boltz1；
未假装已接入 Boltz2。Protenix v0.5 default/constraint、v1、v2 分开保留身份。
MPNN 五种 `model_type` 和 grouped/multistate/旧别名不互相代证。
DynamicMPNN 的既有 callback/CLI 没有可观察模型 owner 时报告未知；显式 factory
接口可以拥有真实模型，不能把任意桥接进程描述成已常驻的上游模型。

## 2. 已实现的公共机制

### 模型生命周期与请求生命周期

worker 的环境/解释器、模型代码、权重文件、框架版本、设备、precision 和影响初始化
的配置进入兼容身份。公共 worker 执行代码和 codec 也参与摘要。实际启动环境与
worker 内部重算的身份对齐，GPU-hidden owner 的环境不能误当成 GPU worker 环境。

`process_alive`、`weights_loaded`、`gpu_resident`、`ready` 与实际 loader 计数独立报告。
未知值为 `null`，不记成零。Chai 逐组件、ESMFold2 的 folding/ESMC、RhoFold 的
主模型/RNA-FM 分别观察；Boltz 的 lazy load 不在 hello 时虚增计数。
释放需要实际进程树退出与账本确认，删除 Python 对象不构成释放证明。

AF3 pilot 保持 CPU featurisation → 连续 JAX forward/D2H → CPU 编码/结构提取。
其他后端沿相同契约接入；含设备操作的 native 复合调用明确保留为 composite，
例如 Lightning predict/writer、部分 diffusion pipeline、LM embedding/dataset transform。
原生联合解码、活跃 GPU 中间量和 RNG 消耗不能为了缩短一个 lease 被拆散。

### 统一资源账本和有界执行

同一物理 GPU UUID 使用同一个 owner 锁。模型常驻与请求 scratch/compute 分账，
CPU slots、RAM、prepare 队列、待处理数量和输出有显式预算。
prepare、execute、finalize 用同一科学请求身份，保留跨阶段 RAM/输出预约。
composite、load 和 probe 同时占 CPU/GPU 预算；managed 子图只在真实设备段领取
设备配额，CPU prepare/finalize 可与另一条已配置独立轨迹的设备工作重叠。

worker 执行 CUDA-capable 程序前向 guardian 登记实际 PID/start-time；broker 崩溃时
guardian 保留设备锁，直到登记进程得到清理确认。历史数字 PGID 不足以安全清理；
无法验证的组会停止准入、保留材料并报告未知。

默认从一个在途轨迹开始，同卡 forward 当前保持一个。控制器按后端性能身份、
真实 shape、冷启动/稳态观察完成率、ready 供应、阶段耗时、CPU/GPU 活动、显存/
RAM 压力和失败；使用观察窗口、试增、最小收益、冷却期及 OOM/退化收缩。
试增只使用原配置中已有 repeats，不增加科学样本。冷启动完成会重置稳态窗口，
避免把初始化时间放入稳态吞吐分母。阶段完成率是控制代理信号，完整轨迹完成吞吐
另计，不能互相替换。

### 科学身份、RNG 与失败语义

科学请求只读绑定既有 trajectory、step、accepted revision、assessment、stage/gate
attempt、输入摘要和实际输入文件内容。seed gate 尚未产生结果时从已有 stage
transaction plan 读取身份。结果返回后验证 owner 状态和文件依赖未变化。
client/worker epoch、传输请求 ID 与科学 ID 分层，重复、乱序、迟到结果不能提交。

保留 Python/NumPy/Torch/CUDA RNG 状态及原同进程后端会影响后续执行的数值设置。
需要保留 constructor RNG 的后端用轨迹私有模型，避免因跨轨迹共享跳过科学 RNG
消耗。RF3 保留原 post-init RNG 回放语义。NA-MPNN 原生 seed=0 的熵语义保持
one-shot；固定源代码摘要和非零显式 seed 才允许对应 resident adapter。

`OutcomeUnknown` 在子进程、server、client、broker、owner 间保持类型和身份，
不自动重试推理。释放错误是 sticky 状态；第二次 close 不能伪装成功。启动或释放
不明时保留 bootstrap/request/artifact，阻止 TemporaryDirectory 的 GC 自动删除。
trajectory 只有在释放和必要清理成功后才能写 `completed`。

## 3. 测试记录

远端使用现有容器，Python 3.11.14、Torch 2.7.1、JAX/JAXlib 0.4.34、NumPy 1.26.4、
ProDy 2.6.1，CPU affinity 0–19。`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`，
没有 `--nv`。完整环境记录见
[cpu-validation-environment.json](../evidence/runtime-20261103/cpu-validation-environment.json)。

| 快照 | 结果 | 解释与原始证据 |
| --- | --- | --- |
| 继承基线选定回归 | 39 passed、1 skipped；173.42 s | [baseline-runtime.log](../evidence/runtime-20261103/baseline-runtime.log) |
| snapshot01 | 112 passed、1 failed、1 skipped；157.58 s | [原始日志](../evidence/runtime-20261103/snapshot01-backends.log)；ProDy Path 字符串接口修复 |
| snapshot02 | 321 passed、4 failed、1 skipped；278.78 s | [原始日志](../evidence/runtime-20261103/snapshot02-integration.log)；修复 CPU 阶段事件和过时 unsupported 测试 |
| snapshot03 | 377 passed、2 failed、1 skipped；408.65 s | [原始日志](../evidence/runtime-20261103/snapshot03-integration.log)；阶段重叠 fixture 适应保守启动，清理 CPU 测试设备环境污染 |
| snapshot04 | 403 passed、1 failed、1 skipped；469.72 s | [原始日志](../evidence/runtime-20261103/snapshot04-integration.log)；全部后端/runtime 回归通过，唯一失败为新增 benchmark fixture 非原子发布 PID 的空串竞态 |
| snapshot05 定向复验 | 21 passed；2.92 s，无 skip | [原始日志](../evidence/runtime-20261103/snapshot05-validation.log)；进程观察 8 项与验收工具 13 项全部通过，关闭 snapshot04 唯一失败 |
| snapshot06 释放/观察与集成复验 | 148 passed；181.03 s，无 skip | [原始日志](../evidence/runtime-20261103/snapshot06-lifecycle.log)；真实进程枚举漏项、client/guardian/NVML 记账故障注入和生命周期、CLI、AF2 共享、CPU MPNN 复验均通过 |

上述失败日志保留，不以新成功日志覆盖。快照文件 SHA、逐文件内容摘要及准确命令
位于 `evidence/runtime-20261103/source-snapshot-*.json` 和 `snapshot*-command.json`。
snapshot05 仅修 fixture 原子发布和与 deadline 一致的等待；该轮没有修改生产进程
观察实现。snapshot06 则包含下文 attempt01 暴露后修复的公共进程观察实现。
snapshot04 的 404 个非跳过用例中，
403 个当轮通过，剩余一项在 snapshot05 修复后通过；不把跨轮结果伪写成一次全绿运行。
唯一 skip 为旧 daemon 模拟准入测试，测试源码说明其对应行为由 broker 准入测试覆盖。

实际 ProteinMPNN CPU 架构配合合成 checkpoint，验证 direct 两次请求与 resident
两次请求的完整输出/RNG 等价、实际 load count=1、CPU 绑定和失败退出。
这证明执行机制的具体行为，不证明预训练质量、五种 MPNN variant 的 GPU 容差
或生物学工作流收益。其他 fake native model/JSONL 测试同样不升级为真实 GPU 认证。

覆盖的科学边界包括固定/shared segment、native ties、multistate、stage transaction、
transition gate、reference score identity、接受流程、日志 RNG 隔离和失败清理。
重叠测试必须实际观察控制器试增和 CPU prepare 与另一 owner 的 forward 区间重叠，
不能仅凭线程存在判定通过。

## 4. 可复现性能入口

完整说明见 [runtime_execution_guide.md](runtime_execution_guide.md)。三模式工具为
`scripts/validation/runtime_workflow_benchmark.py`；时间线解释工具为
`scripts/validation/runtime_trace_report.py`，输出结构逐原子比较为
`scripts/validation/runtime_structure_parity.py`。

性能工具保持原配置及 `${seed}` 等每轨迹插值，在运行前验证科学配置摘要和预期
repeat/seed 集合。每轮轮换旧直连、常驻串行、常驻自适应的顺序，全部已有任务和
失败计入总预算；没有额外 warm-up 科学样本。独立退出观察隔离遥测采样对计时的影响。
两次以上 direct 的重复误差给出逐字段数值容差，离散决策、接受/stage/early-stop/
选中样本和序列单独严格比较。重复、额外、缺失结果不能获得有效吞吐。

源码 manifest 包含新增未跟踪模块，避免只有 `git diff` 而遗漏新实现。
结构文件的原子身份/坐标误差只证明已交付 PDB/mmCIF 的文件精度等价；未输出候选、
内部 tensor 和 cache intermediate 仍需各后端 conformance 证据。

CPU fixture 的实测结果单独记录。它只检验完整 CLI 协议、隔离及控制开销，
不代表大型 GPU 模型性能；没有足够稳态样本时控制器保守不试增是预期行为。

### CPU 对照 attempt01：保留失败，不得认证

[原始完整报告](../evidence/runtime-20261103/cpu-three-mode-01/report.json)和
[可读报告](../evidence/runtime-20261103/cpu-three-mode-01/report.md)记录了九个交错 trial。
前八个完成；最后一个 resident_serial 在独立进程观察中遇到全局 `/proc` 枚举漏项，
随后逐 PID 检查仍确认该身份存活，观察器保守中止并清理了试验。
原始错误是 `known live process missing from snapshot: 3160744`。

本轮总耗时 **214.460871 秒**，完成 **25/27 个轨迹**，CPU 时间采样下界
**195.34 CPU 秒**；GPU 分配为零。已完成的可比较评分/决策没有差异，但完整固定
工作量未完成，所以各模式的 `effective_trajectories_per_second` 均为 `null`。
没有结构输出，坐标等价未测量。失败 trial 的 `release_confirmed=false` 不能改写
为正常成功，即使后续最后一次观察已经没有已知残留 PID。

这一故障促成严格 `process_record` / 已登记 PID 复核的公共修复。针对性故障注入
同时覆盖 daemon 关闭不能提前确认释放、guardian 不能因首次枚举漏掉登记 root
而释放锁、NVML 驻留计账不能因漏 PID 低估显存。后续修复和新的完整对照单独记录。
第二次对照最多使用 600 秒，连同 attempt01 仍小于原定的 900 秒总计时预算；
失败时间和全部原始材料均保留。

### CPU 对照 attempt02：固定工作量通过，该配置没有加速

[原始完整报告](../evidence/runtime-20261103/cpu-three-mode-02/report.json)、
[可读报告](../evidence/runtime-20261103/cpu-three-mode-02/report.md)和
[准确运行命令](../evidence/runtime-20261103/cpu-benchmark-02-command.json)均已保留。
配置为 MCMC + Dummy + `sequence_residue_fraction`，输入 `AAAA`，每次调用原有
3 个 repeats、seed 17/18/19、每轨迹 3 steps。九个 trial 轮换三模式顺序；旧直连
运行冻结的 `834402d` checkout。两次实验的科学协议摘要相同：
`f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`。
容器、CPU affinity 0–19、线程设置和科学工作量保持一致；没有获配或使用 GPU。

| 模式 | 三次端到端耗时（秒） | 中位耗时（秒） | 有效轨迹/秒 | 进程树 RSS 峰值（GiB） | 采样 CPU 时间下界（CPU 秒） | 采样峰值核当量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 旧直连 | 13.969 / 9.691 / 9.232 | 9.691 | 0.27362 | 1.112 | 26.27 | 1.039 |
| 常驻串行 | 32.418 / 33.305 / 31.832 | 32.418 | 0.09226 | 2.239 | 91.74 | 1.144 |
| 常驻自适应 | 33.624 / 33.455 / 32.911 | 33.455 | 0.09001 | 2.238 | 93.68 | 1.147 |

每个模式的有效吞吐是 9 个完成轨迹除以该模式三次 trial 的总完成耗时，经过完整
工作量和重复 direct 评分/决策等价门控；不是 3 除以中位耗时。所有九轮均退出成功，
`release_confirmed=true`，没有科学输出差异。无结构文件输出，坐标等价为未测量。

CPU 秒是每个模式三次运行的采样总和，可能漏掉短命进程；峰值核当量根据约 0.5 秒
采样窗口内累计 CPU 时间增量计算，不是瞬时硬件峰值。RSS 是各进程 RSS 之和，
共享页可能重复计入，因此不是物理 RAM/PSS。计时由独立 30 ms 进程退出观察完成，
资源采样不决定计时分辨率。原始 `telemetry.jsonl` 与 resident `runtime.jsonl` 随每轮保存。

三次自适应运行的控制上限始终是 1；时间线记录 `cold_start_observed` 和
`observing_steady_work`，没有足够的稳态观察支持试增。每次 resident trial 有 9 次
CPU prepare、9 次 CPU execute、9 次 CPU finalize，无 device stage。该轻量 CPU
工作负载表现为额外进程和调度成本；这些证据不能进一步精确归因各项开销，亦不能
作为真实模型常驻机制没有收益的结论。

所有 CLI trial 都从新进程开始。Dummy 没有推理权重，真实模型的加载次数、冷启动/
稳态 GPU 吞吐、GPU activity、VRAM 峰值及 GPU 空闲原因均不适用或未测量。
零 `model_start` 事件不能充当预训练模型零加载或模型成功复用的证据。

attempt02 全循环耗时 **230.658022 秒**；连同 attempt01 的失败成本共
**445.118893 秒**，低于两轮共同的 900 秒预算；GPU 分配成本为 **0 GPU 小时**。
首轮不能认证的结果没有混入第二轮有效吞吐。
[cpu-evidence-summary.json](../evidence/runtime-20261103/cpu-evidence-summary.json)
记录各项口径和原始文件摘要，可在仓库根目录离线重算：

```bash
python evidence/runtime-20261103/summarize_cpu_evidence.py
```

### 阶段重叠与控制器试增的独立机制证据

[overlap-cpu/README.md](../evidence/runtime-20261103/overlap-cpu/README.md)和
[逐事件摘要](../evidence/runtime-20261103/overlap-cpu/summary.json)对应 snapshot04
中的真实 CLI/进程测试。两个 owner 的 10 个请求使用同一个 worker；有一段实际
SHA-256 CPU 工作 **0.104781211 秒**完整落在另一 owner 的 forward 区间中。
10 个 forward 仍串行；控制器实际出现 **1→2→1→2**，其中收缩原因是
`throughput_regression`。两个 seed 对应的 `scores.csv` 逐字节相等。

该 fixture 的 forward 为 CPU 同步/sleep，并有专门 rendezvous；reported load=1
也是 fixture 声明，权重状态明确为 false。它证明公共执行与控制机制能运行，
不证明 GPU 并行、真实权重复用或自然工作流加速；旧分支未执行该 fixture 额外的
hash/sleep，因此这里也没有把两边墙钟时间当作公平性能对照。

## 5. 尚未完成的真实验收及明确限制

| 项目 | 当前状态 / 需要的证据 |
| --- | --- |
| 各预训练后端真实单请求与连续复用 | 未完成；需要准确环境、权重和获配 GPU |
| 各 exact variant 的 cache 等价、隔离和失败清理 | CPU/协议证据已有，真实 GPU 版本配置需逐项测试 |
| 数值容差与决策一致性 | 工具和机制已实现；真实容差必须来自该后端 direct 重复误差 |
| GPU 端到端有效吞吐与成本 | 尚无可宣称提升的配置；不能以 busy、mock overlap 或单阶段计数代替 |
| 任意第三方脱离进程组的 doublefork/setsid | 不在完整 containment 认证范围；须有作业 cgroup 等外部硬约束 |
| RAM/VRAM/vendor 文件硬上限 | 资源准入和采样超限终止已实现；不冒称新建 kernel/cgroup/filesystem quota |
| MIG、未审查的 GPU hooks/predockers/initializers | 显式拒绝，不静默逃出统一 owner |
| 动态私有模型 eviction/reload | 未启用；避免重放构造器 RNG，容量不足会失败，不缩样本或降精度 |
| 所有现有评分代码进入 CPUExecutor | 未做此扩展；原 CPU 路径受作业约束，不宣称全部已池化 |

需要用户确定的事项是最终集成分支、是否新增 Boltz2/其他外部 API、AF3 pilot 和后续
各后端的完整科学配置/准确权重/环境、GPU UUID 或调度器分配、CPU/RAM 和总实验预算。
这些决定不影响已完成的代码、CPU 科学回归和可审查证据交付；它们是实际 GPU 验收的
必要输入，不能由“机器上有空闲 GPU”推断。
