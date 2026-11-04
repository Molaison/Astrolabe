# Agent C：最终集成前的科学边界与剩余缺口复核

**结论：CPU 启动优化的明示退出门槛已通过；完整 Runtime backlog 尚未完成。**
f9 相对已验证 9d 的 W0 三种模式均有 5/5 配对改善且耗时范围分离，aggregate
有效吞吐分别提高 69.08%、69.99%、54.92%。resident 仍约为 direct 的 3.22–3.23
倍耗时。任务身份、故障隔离与实际 native PyRosetta 检查可给出窄范围 DONE；
不能据此宣称完整异步 producer、pretrained evaluator W1、持久化唯一提交或
全部 backend 已认证。

本报告是给主代理的只读交叉审查，不替代其最终报告。冻结的 owner 源为
`1e5771b8e2d8f00eeb8658b224d00790c0d6e392`，额外审查启动候选
`f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f`。`source-manifest.json`
逐一记录 17 个文档/源码的 SHA；两个 snapshot 中仅 interaction 与
runtime_requirements 不同，其余被审科学 owner 相同。此时未收到最终集成
HEAD 的完整终态 gate 或实际 GPU raw，因此下表是 provisional 状态。

## 对照原始 R0–R12

状态按条目的完整范围判定；窄范围完成项在同一行明确列出。尤其 R1 的文字
“找到主导开销并证明改善超过噪声”退出判据已经 DONE，而原条目要求的长稳态
break-even 与自动 routing 尚未补全，不能与该退出判据混称。

| 项目 | 建议状态 | 已有可验证结论与尚缺内容 |
| --- | --- | --- |
| R0 workload/测量 | PARTIAL | 固定 W0、30 trials、90 trajectories、270 eval、全部 raw/config/source/science/release 审核 DONE；W1 evaluator+实际 mutator、W2 有意义 CPU/GPU overlap 未认证。 |
| R1 resident 回退 | PARTIAL；冷启动退出 DONE | 实际 SIF ABBA profile 定位未使用 PyRosetta import 与 source scan，f9 W0 改善明显超过噪声。resident 仍慢于 direct；长稳态 break-even、按成本自动 routing 未验证。 |
| R2 producer/consumer | PARTIAL | 外来/stale identity 拒绝、整代输入检查与 lifetime duplicate guard 已红绿复现。科学 producer queue 尚无有界异步背压/持久化唯一 commit；无 W2 overlap 证据。 |
| R3 backend 阶段边界 | PARTIAL | ESMFold 原生 helper 的 CPU output/lease contract、既有 prepare/execute/finalize 得到验证；无 pretrained evaluator GPU pilot。requested payload 减少不能当作实测 CUDA transfer。 |
| R4 ownership/ADR | DONE（设计范围） | 既有 ADR 明确 caller-first 数据、唯一科学 writer、3 种实质不同方案及拒绝理由。DONE 指设计决策，不表示 deferred actor/queue/durable ledger 已实现。 |
| R5 首个 pretrained evaluator | BLOCKED | ProteinMPNN 真权重 strict CPU load 与全输入 parse 是 readiness；完整 evaluator W1、不同输入、重启/OOM、数值/驻留/稳定吞吐证书仍缺。 |
| R6 GPU concurrency/batch | BLOCKED | 保留同设备 forward=1；未做真实同卡 1/2/N、microbatch、显存和干扰对照。 |
| R7 增量缓存 | PARTIAL | 复用已有严格 key schema、bounded native parse cache 和局部 altered-input tests；未认证完整 pretrained mutation dependency graph、stale negatives 与实际净收益。 |
| R8 动态资源/多 GPU | PARTIAL | 回收旧 fix 的 bounded broker ingress、admission deadline、cancel ownership、release 已有 capable-host CPU contract evidence。它不是 bounded scientific producer；真实多模型共卡/多 GPU fairness/VRAM 仍缺。 |
| R9 故障恢复 | PARTIAL | CPU timeout/queued cancel/late reply/fence/cleanup/no-auto-replay 有实际与受控证据；无 pretrained GPU OOM、forward 中取消、重启或 durable outcome/commit recovery 全证书。 |
| R10 科学一致性 | PARTIAL | W0 完整 CSV；独立 accepted mutation+stage positive control；真 PyRosetta 两 loss/geometry/3 CPU RNG 流均 PASS。W0 自身全拒绝；完整 checkpoint/crash-resume、全部 native/GPU RNG 与最终集成 control 另需验证。 |
| R11 backend 矩阵 | PARTIAL | inventory DONE：14 evaluator keys +20 mutator keys，版本/工厂/config 范围明确；实际 registry 行仍 6 PARTIAL/28 BLOCKED。不能用一个 runner 覆盖全部 MPNN aliases。 |
| R12 新版本/异构部署 | BLOCKED | task-specific Slurm launcher 与 CPU contract 不能替代新版本、异构多 GPU、部署 ADR 与实际验收。此项不阻塞既有 P0/P1 范围交付。 |

本轮 W0 事实来自 C commit `b846518710ee12c5cabc095fa5afb32e1a92e455` 的
`evidence/runtime-next/agent-c/f102-startup-w0/`，含全部 634 raw files、逐文件
SHA、不可变 reader adapter 与复验命令。此前 broader Runtime 的 c90/9d 负
结果及实际 native control 在 `666c4988b63b76996b8aad55d4926458b4bef1e8`
的 `evidence/runtime-next/agent-c/f102-startup/`。两轮不可合并为一次测量。

## 科学因果与恢复：不可扩大解释的五个边界

**单轨迹仍需按因果提交。** `candidate_batch_runner.CandidateDecisionRun.run`
在选定上一代结果后才构造下一代；同代是有限 fixed-parent proposals，显式
seed。默认 MCMC 的后续 proposal 继续依赖 evaluation/acceptance，不能被
这种实验性 best-of-first-completions 协议替代。旧 blocking-submit prototype
在相同三个任务下改变 winner 的负结果仍有效。`CandidateTask.payload` 是
`Any`，frozen dataclass 也不是全系统深度不可变 DTO；后续任何跨轨迹队列都
还需完整 lineage、RNG、stage/parent/config identity 约束。

**in-memory identity 不是 durable commit。** `candidate_batch_policy` 的
submitted identities 与 `broker_state.State.used` 只在进程/session 生存。
`execution_identity.scientific_operation` 在远端调用前后校验 owner/file
fingerprint；`decision_maker` 和 `stage_transaction` 仍是 acceptance/stage
writer。这个检查上下文没有成为第二个 writer，也没有建立跨崩溃的 commit
日志。因此不能把拒绝 duplicate reply、取消 ack 或保留 artifacts 叫作
持久化 exactly-once/recovery。

**取消与未知执行必须保持保守语义。** recovered State 的 cancel ack 明示
`release_confirmed=false`；派发后的不确定超时走 OutcomeUnknown/fence，不能
自动重放。明确未 dispatch 的 AdmissionRejected 可按已有规则退回 ID；
prepare 已完成但 execute 未 dispatch 的 owner/token 保留，不应误清或重算。
实际释放还依赖 observer/worker cleanup。deadline 限制新 dispatch 的预算，
不是保证 bookkeeping/cleanup 的硬墙钟返回上限。这些是被保留和验证的契约，
不是本轮新发现的未修复 defect。

**现有缓存可复用，认证仍需到实际 backend。** `feature_cache` 要求
source/weights/dtype/mapping/chains/topology/constraints/seed/MSA/templates/
CCD/options 等 key；key 字段齐全不自动证明各 caller 填入了全部真实依赖。
LigandMPNN CPU parse cache 对完整 PDB digest、chains、side-chain context 和
zero-occupancy 取 key，有限 bytes/entries，prepared 输入使用前校验 digest。
它缓存结构解析，不是跨 mutation 复用 sequence embedding/recycle graph。
相同完整 PDB 的 GPU control 可证明其固定输入路径，不能覆盖改变结构、mask、
约束和其他模型配置的全部 invalidation。这里只列证据缺口，不推断未经复现的
stale-cache bug。

**科学正控制与吞吐数据分开。** W0 的 90 accepted rows 全是初始化，180
mutation 均 rejected、0 stage entry；不能视作科学进展。另一个固定 control
已有真实 1 accepted mutation、1 stage entry、4 backend calls、5 score rows；
最初错误预期 5 calls 的 PARTIAL 也被保留。实际 PyRosetta 控制只认证两个
registered losses、全坐标、169 registry metadata/signatures 和 Python/NumPy/
Torch CPU states；它显式冻结 native Rosetta seed，未认证所有生产 packing
或 native RNG 配置。最后应把这个 positive control 在最终 HEAD 复现，而非
靠旧 HEAD 结果覆盖 checkpoint/restart 等缺口。

## GPU control 与 backend 矩阵的准确范围

B 的 `27f8a6d0660867ff11823ab890655eb01125d868` 新 f102 矩阵已独立只读
复核。执行其纯 JSON/hash/static-registry reader，6 个 evidence files 的
manifest、真实 completed command 与 CPU summary、native source、权重/输入
hash、parse/settings 全匹配；结果在 `b-f102-matrix-review.json` 与
`b-f102-preflight-independent.json`。没有新 native import、forward、benchmark
或 GPU 运行。ESM 主 checkpoint 的 BadZipFile 有实证；缺 distribution metadata
不被表述为实际 `import esm/openfold` 已失败。

冻结 GPU control 只调用 `build_runner("ligandmpnn", model_type="protein_mpnn")`
的 public runner，在同一个 candidate 比较 existing legacy 和 private resident
模式；两边均一次构造后连续三次请求。完整 4GYT A177+B177、真实 checkpoint、
5 seeds、10 loads、30 sequential decode，无 warmup/请求间 reseed；S/order/RNG
exact 与浮点容差均已冻结。其实际 GPU raw 在本报告时仍未交付审查。

将来即使该控制 PASS，也只能增加这个 exact runner scope 的真实 GPU、连续
请求和模式性能证据；不能改写为 registry selector/proposal/acceptance、完整
MCMC、shared broker、pretrained evaluator W1、其他 MPNN types/ties/multistate，
也不能把同 candidate 的模式差异写成“本次 revision 加速”。完整 R5 与 R11
认证仍应按各自的未覆盖门禁保留 PARTIAL/BLOCKED。

## 最终集成待办（由主代理实施）

最终 HEAD 必须单独保存精确源码身份、全量 mandatory CPU/geometry suites、
科学 positive control 和所选正式 W0 的结果。PyRosetta lazy import tests
依赖 RDKit，已转入仍为 mandatory 的 geometry/Torch gate；core env 的真实
8 个缺依赖失败需保留，不能改冻结 CPU 依赖或跳过测试来变绿。GPU 实验若获
实际 allocation，C 再从原始请求/概率/RNG/设备/释放记录独立审核；没有 raw
时不提前升级任何认证。

此审查没有新增生产修改，也没有发现经过新 failing run 证实的生产 defect。
未完成项是根据原 backlog 的验收范围和现有源码/evidence 划定，不用于推动
未经授权的新框架或扩大 benchmark 预算。
