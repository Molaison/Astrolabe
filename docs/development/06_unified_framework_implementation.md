# 统一框架当前实现

本页描述包含 `astrolabe/design/` 的开发树。
2026-11-06 按当前源码更新 API 和恢复说明；
文末测试结果保留原验收日期，不表示本次重跑。
发布分支与开发副本的差异见[源码版本边界](../developer_guide.md#source-version)。

在这份开发树中，Astrolabe 已形成一条公共控制面：
`astrolabe.design.coordinator.drive()` 是唯一搜索控制循环。现有默认运行、
直接分子组合、通用 branch/collection 以及兼容入口都通过这一驱动前进。
这不表示全项目已通过 GPU、真实模型 provider 或数值等价验收。

后续全仓复查补上了配置 factory 的真实通用装配、统一前置校验和保存配置的
重放路由，具体入口矩阵及能力边界见
[全仓入口与控制流复查](07_repository_control_audit.md)。通用
`native_refinement` 现直接复用既有静态 stage、组件注册和 loss 表达式；
档案科学协议通过显式 `archived_refinement` factory 装配，不再占用通用默认。

本文描述当前可运行实现和明确边界；目标和用法仍未实现的部分见
[统一框架](03_unified_framework.md)、
[接口草图](04_unified_framework_sketch.py) 和
[使用方式草图](05_unified_framework_usage.md)。

## 公共入口和生命周期

| 入口 | 当前数据/职责 | 进入公共控制的路径 |
| --- | --- | --- |
| `Astrolabe(composed_cfg).run()` | 保留现有 repeat 生命周期、配置、组件和输出 | 每个 pipeline repeat 经 `run_optimization_loop()` 进入 `run_native_optimization()`、`coordinator.run()` 和共享 `drive()` |
| `Astrolabe(DesignRun(...)).run()` | 已完成 `System`/`Candidate`/`Trial`/`SearchPolicy`/`Runtime` 的分子组合 | `coordinator.run()` 执行预检、root、typed Trial 和同一 `drive()` |
| `Astrolabe(ControlRun(...)).run()`，`runtime=None` | 非分子或兼容 typed control；调用方声明 request/result 类型和初始状态 | 直接调用共享 `drive()` |
| `Astrolabe(ControlRun(...)).run()`，提供 `Runtime` 和三类 codec | 可记录状态、请求、结果和 RNG 的 typed control | control journal 包装 Trial 后调用同一 `drive()` |

`ControlRun` 不会把任意 external job、process payload 或 benchmark 结果包装成
合成分子 `Candidate`。它保留其真正的 typed action/event；分子科学工作使用
`DesignRun` 的完整身份校验。

`CompletionPool` 内部仍有一个 futures 等待循环。这是资源收集器等待已经提交
工作的实现，不是第二个搜索状态机；generation、预算、去重、父代选择和停止
原因由 policy 持有。`gather_ordered()` 的 free-slot refill 同样只处理执行资源。

## 已迁移的旧入口

- 默认 `run_optimization_loop()` 是薄兼容入口。root 初始化和单步推进分别是
  有界操作；兼容 wrapper 没有 while，也不持有独立 generation 搜索。
- 实验性 candidate batch 使用 `CollectionPolicy`、`CollectionTrial` 和
  `ControlRun`。真实 `ProcessCandidateWorker` 路径提供持久 Runtime 和
  receipts；dry-run 或其他 worker 的兼容路径不提供同样的持久恢复。
  `CandidateDecisionRun.run()` 经 `Astrolabe(...).run()` 进入同一驱动。
- native refinement 使用 `BranchingPolicy` 和 `ControlRun`。
- isolated protein beam 使用 `NativeBranchTrial`、`BranchingPolicy`，并提供
  Runtime/codec 以便恢复控制元数据；科学 worker 和输出 owner 留在原模块。
- RNA/bivalent-RNA 入口保留自己的科学 adapter，但搜索经
  `run_native_refinement()` 和 `ControlRun` 前进，不再拥有独立搜索循环。
- paper `_run()` 只负责配置、输入、路径和调用普通 run 入口；不按 execution
  engine 分派搜索，也不构造 RNA 专用搜索或按 `run_count` 驱动实验。为提供
  明确诊断，它可以读取 deprecated engine，仅用于在 non-trajectory 且缺少
  `cfg.search` 时拒绝该路径。

旧 `BeamSearchState` 预算 owner、`fill_generation()` 控制和
`protein_beam_dispatch.py` 已从迁移路径移除。结构化验收会拒绝这些符号或
控制循环重新出现在迁移入口中。

## BranchingPolicy

`BranchingPolicy` 是非分子公共策略；科学 adapter 只提供 proposal、evaluation、
measurement、selection 和 owner callback。策略拥有预算、去重、父代、frontier
与停止原因。

| 选项 | 当前语义 |
| --- | --- |
| `quota_mode="attempts"` | proposal attempt 在发出时占用父代配额；无效或重复 proposal 也消耗该配额 |
| `quota_mode="admitted"` | 只有合法、非重复且实际进入 evaluation 的 proposal 才计入父代 admitted 配额 |
| `expansion="balanced"` / `"round_robin"` | 在父代间分配 proposal slot；保留各 recipe 的父代顺序 |
| `proposal_batch_size` | 每个 proposal action 的逻辑批量；仍由 Trial 执行一次有界请求 |
| `evaluation_width` | policy 允许同时 pending 的 evaluation 数；不等于 GPU 并行能力 |
| `pipeline` | proposal 与 evaluation 两个 lane 可重叠，已完成的 evaluation 不等待完整 generation |
| `immediate_evaluation` | 每个 proposal 直接进入 evaluation；要求 proposal/evaluation 批量宽度均为 1 |
| `max_evaluations` / `proposal_cap` | 分离 effective evaluation 预算和 proposal 工作量上限 |
| `archive_selection` / `max_generations` | 选择 archive 或当前 frontier，并可显式限制 generation |

proposal seed 和 selection callback 的身份必须显式且稳定；capturing callback
需要 caller 提供 identity。只有相同 `comparison_context` 的成员才能排序；
改变 objective 或样本归约协议时，需要显式重新评价。

## CollectionPolicy 和 CompletionPool

`CollectionPolicy` 负责 decision generation、next job identity、selection 和
adoption callback；`CompletionPool` 只负责提交、等待和收集 typed job result。
外部 job/result 可以保持 process-specific 类型，不要求转成分子对象。

collection 保留以下旧语义：

- `min_completed` 统计所有 ready 结果，失败结果也计入 readiness。
- `max_wait_s` 是当前 collection window，不是 worker cancellation。
- window 内至少一个结果可用时返回部分 collection；零结果超时则报错。
- 未完成 job 原样 carry-forward，不重新编号、不重复提交。
- decision generation 结束后必须显式 `drain()` 未完成工作并聚合遗留结果。
- 结果保持实际完成/交付顺序；不为了显示或 tie-breaking 静默排序。

## 兼容面和科学输出

`astrolabe.experimental.candidate_batch_policy` 保留 candidate 命名的兼容别名；
generic 实现位于 `astrolabe.design.collection` 和 `collection_search`。
`CandidateBatchRun` 是 `CollectionRun` 的兼容名。

保留的兼容面包括：

- process candidate 的既有配置、CLI 参数、candidate 编号、结果表和 summary；
- protein beam 的 native proposal/evaluation、calibration、budget reserve/probe、
  checkpoint、CSV/JSON 输出和旧公开入口；
- RNA/bivalent-RNA 的 native mutator/evaluator、seed、去重、父代次序、指标和
  结果 writer；
- default native 的 stage、acceptance、mutation stack、early-stop 和原有摘要；
- legacy scientific translation、score/evidence adapter 与输出 consumer。

兼容入口可以保留原有名称和输出格式，但不能恢复第二份控制循环或第二个预算
owner。配置入口通过 `(search.algorithm, search.owner)` 注册表选择工厂；
未知选择报错。repeat 的起始序列、seed 优先级和目录模板归共享生命周期处理，
paper 不再维护专属分派或重复执行循环。

## 恢复边界

默认 native 运行使用内存 Runtime，恢复能力明确为 `process_lifetime`。
`execution.json` 记录 `runner=astrolabe.design.coordinator`、恢复能力和尝试状态。
PyRosetta 完整 Pose、活的 native scores 和模型 owner 不能在进程重启后伪造；
可表示的结构字节和指标进入 typed store；带 owner/candidate/revision 的 process
handle 仅在当前进程有效，禁止直接或间接落盘。再次推进科学状态前，旧 handle 失效。

`ControlRun(runtime=None)` 同样没有跨进程持久恢复。
不能据此推断所有 candidate batch 都只有 process lifetime。

当前 `experimental/candidate_batch_runner.py` 中，真实
`ProcessCandidateWorker` 且非 dry-run 的路径在 `.framework/journal.sqlite`
创建持久 Runtime，并使用 `ProcessCandidateReceipts`。
`--resume` 要求 journal 存在；已有 journal 而未指定 resume 时拒绝重新启动。
其他 worker 或 dry-run 的路径拒绝 durable resume。
缺少完整完成证据的 issued 请求保持 unknown，不能自动重发。
这些行为由 `tests/test_candidate_recovery.py` 和相关配置测试覆盖。

无论是否提供 durable Runtime，policy 的 `finished`、`ask`、`admit`、`tell`
和 `selected` 都必须是 RNG-pure transition。coordinator 会在 transition
前后捕获 Python/NumPy 等共享 RNG；检测到状态变化时恢复调用前状态并抛出
`ValueError`，且不会把请求交给 Trial。随机工作只能留在 Trial 的有界科学
调用中。该约束不要求为 opaque process payload 增加 journal，也不改变
process-lifetime control 不承诺跨进程恢复的事实。

显式提供 Runtime 和 state/request/result codec 时，control journal 原子保存
policy checkpoint 与 pending request，并为完成结果保存独立结果/RNG receipt。
恢复顺序为 state/request 与 result receipt 对齐后再交付；已 issued 但无结果的
native job 必须由所属 owner 的 probe/reconcile 返回原始 typed result 和 RNG
receipt。不能确认时不自动重发，也不把 PDB、空分数或 metadata 当成完整
科学状态。

## 已记录的历史验证

以下数字来自原交付记录。文档同步没有重新运行这些测试；
当前验收按[验证阶梯](../developer_guide.md#validation)选择。

新增的 `tests/test_unified_control_surface.py` 使用轻量 typed stepper 和 AST
结构检查，不加载模型：

```text
6 passed
```

它验证 `Astrolabe(ControlRun(runtime=None))` 恰好调用一次共享
`coordinator.drive()`；process-lifetime policy 若消耗 Python/NumPy RNG，
必须抛出 `ValueError`、精确恢复 RNG 且不调用 Trial。迁移入口没有本地
while/薄 wrapper generation loop，
并以实际源码拒绝 `BeamSearchState`、`fill_generation` 和
`protein_beam_dispatch` 回退。RNA 三个公共 facade 必须存在并向其有界
scientific adapter 前向，同时整个模块不得恢复 while 控制循环；测试不依赖
私有 helper 名称。paper `_run()` 的 AST 不包含 RNA 专用字面量或 `run_count`
循环，也不按 execution engine 驱动循环或分派。

2026-10-31 的最终整合范围为 **469 个去重用例**，覆盖统一内核、所有迁移
入口、paper/repeat、默认阶段/接受、调用配额及既有结构的根校准/几何检查。
主批为 463 passed、6 failed；五处失败来自旧 pycache 中的容器挂载路径，
一处来自模拟运行缺少路径资产。用独立 `PYTHONPYCACHEPREFIX` 和明确的
仅供 metadata 测试使用的资产修正环境后，补验 9 passed，覆盖全部六处原失败。
该范围的每个用例均有通过记录；不是声称单次运行显示 469 passed。

测试使用真实 jsonschema 4.26.0、现有 Python 3.11 CPU 镜像；原工程输入及
已有预测结果只读绑定，未重新推理。工程 fixture 按字节复用原有文件；仅被
mock 的模型测试使用假资产验证路径/stat 合约，未将其用于科学计算。
candidate batch 单独 27 passed，控制 journal/branching/executor 等批次有重叠，
不额外相加作为总量。

各 wrapper 产生的 typed `Scored` candidate/evidence/assessment 身份、完整
history 和 domain 校验由对应领域回归覆盖；本控制面测试不重复这些科学测试。

这些结果只证明 CPU 控制结构、typed 交接和既有测试中的科学行为，不是 GPU
数值等价、吞吐、provider 部署、真实 checkpoint 或完整历史恢复的资格证明。
GPU/provider 验收仍待完成。
