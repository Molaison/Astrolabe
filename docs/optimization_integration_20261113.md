# 2026-11-13 优化方案整合

统一分支为 `integrate/optimizations-20261113`。整合以刷新后的
`origin/main@a29d3165f34d392585c6550704774eee998b0658` 为基线，合入
`feature/pipeline-persistence-optimization@418df6d6c2b55f2b317efddacf04e25641dec8b8`。
两者的合并提交为 `8aad56f`；原始提交历史均保留。
随后用 `7d48b5f` 窄移植旧空间依赖延迟加载方案，保留现行初始化 owner。

## 最新方案及相互关系

| 方案 | 来源 | 整合结果与判断 |
| --- | --- | --- |
| 运行时、持久 Score、显式并行 Beam | `integrate/runtime-score@25a2b3a` | 已在双方共同历史中，随主线保留。是后续优化的基础；不能将早期 Score 微基准收益当作全流程收益。 |
| 残余状态与恢复边界 | `fix/residual-state-owners@e7b1fed`，共同基线 `3eecc7b` | 已在双方历史中，保留失败状态、恢复和资源释放合同。 |
| 启动依赖延迟加载 | 11/3 startup 补丁、`4d9dffc`、`109f654` | PyRosetta 延迟导入和源码扫描复用已覆盖；将尚缺的 SciPy 空间依赖延迟加载局部移植为 `7d48b5f`，并补齐当前 SC 模块的导入路径。 |
| vdM pair / Proton 与归一化 pH 评分 | `8e055b5` | 随优化分支合入。属于既有设计能力和评分接口集成，不作为运行时加速证据，也没有在本次更改科学方法。 |
| 可配置模块 profiling | `feature/modular-profiling@109f286` | 合入诊断采集、模块计时和报告入口；默认关闭。profiling 本身不是已证明的提速方案。 |
| AF3 PAE sidecar 定位 | `feature/profiling-pae-recovery@40fd7ea` | 合入按结构路径、预测标识及 evaluator alias 定位 confidence 的正确性修复，防止读取其他预测的文件。 |
| 私有 Score 恢复 | `feature/profiling-native-score-recovery@336129f` | 合入原生结构路径及字节来源恢复，与 PAE 定位配套；保留私有目录和 assembly 检查。 |
| 持久化复用、异步 Mutator、独立 Score 限额 | `cad170c`，交付 `418df6d` | 三项属于同一个 r58 方案，均以显式配置开启；保留原始顺序、RNG、持久化和失败处理边界。 |
| Mol-1 自动 base/Constraint 切换 | 主线 `5400e79`、`a29d316` | 保留主线现有实现、两套资产身份、输入路由及旧进程关闭行为；没有用优化分支覆盖这一功能。 |

最近四个具名优化分支是一条连续历史：

```text
3eecc7b
├─ 5400e79 ─ a29d316                       origin/main
└─ 8e055b5 ─ 109f286 ─ 40fd7ea ─ 336129f
                                  └─ cad170c ─ 418df6d
两条历史合入 8aad56f
```

因此合入 `418df6d` 已包含 profiling、PAE、Score 恢复及 vdM/Proton
集成，不需要再次 cherry-pick 这些提交。

## 三项流水线优化的边界

| 设置 | 默认值 | 开启后的作用 | 保留的约束 |
| --- | --- | --- | --- |
| `search.execution.reuse_durable_closures` | `false` | 复用同一 DataStore 中已经成功持久化的内置 codec 闭包证明 | 首次内容验证、fsync 和 journal 前屏障保留；自定义 codec 完整重验。 |
| `search.execution.async_mutator` | `false` | 让远端 transport/wait 与其他工作重叠 | 主线程按原顺序恢复 context/RNG 并应用结果；一个活跃提案；未知结果不自动重放。 |
| `search.execution.score_queue_size` | `0` | 将预测额度与有界 Score 额度分开 | raw receipt 持久化后才释放预测额度；等待按 ID 发布的结果仍计入总容量；保留代际屏障和 drain。 |

显式开启模板为 [`config/pipeline_overlap.yaml`](../config/pipeline_overlap.yaml)，
其中 Score 队列为 4，要求 isolated/shared 执行。它不改任务已有的
worker、CPU 或 GPU 配额。一个任务只传一份合并后的 overlay，避免重复
`--overlay` 覆盖。使用和诊断入口见 [profiling 文档](profiling.md)。

Mol-1 与流水线修改共同涉及配置解析、资产身份、模型执行及 registry。
合并没有文本冲突；审查确认 Mol-1 的别名、双权重身份和 `poolable=False`
私有进程语义，与新增 profiling 和流水线设置同时保留。首次合并仅额外
移除 `ph_binding_endpoint.py` 原有的文件尾空行，不改变运行逻辑；
后续空间依赖移植见下方。

## 性能结论

r58 的既有对照固定 D5056_D15、A73+B198（271 residues）、seed 10079、
同一 GPU，以及 4 prediction workers / 4 Score workers。
11 次资格推理加 6 个完整任务各 12 次推理，共 83 次 AF3 forwards；
当时六任务科学载荷一致，三对 RNG 一致，队列均完成并释放 owner。

关闭 profiling 的 BAAB 对照均值为 **347.722 → 339.769 秒**，按
`(347.722 - 339.769) / 347.722` 计算约改善 **2.29%**，低于原先约定的
**3%** 确认门槛。因此这项方案是已交付的可选实现，
**端到端性能收益仍未确认**。不能把早期 1→4 worker 的收益归到这里。
该负载的 Score 活跃峰值仍为 1，也没有证明新增 Score 并发收益。

以上是原始 r58 的历史科学证据，不是本次统一分支的新 GPU 测量。
本次整合不重跑 GPU 对照、不更改模型、权重、科学预算或默认优化开关。
Mol-1 的确定性 canary 与非确定性比较有不同结论，详见
[Mol-1 使用与验收边界](mol1_constraint.md)。

本机原始证据入口：

- `/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/runs/001_execution_architecture/01_shared_contracts/58_pipeline_persistence_optimization/final_delivery.json`
- `/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/analysis/001_execution_architecture/01_shared_contracts/58_pipeline_persistence_optimization/delivery_metrics.json`
- `/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/publish/001_execution_architecture/01_shared_contracts/58_pipeline_persistence_optimization/README.md`

## 旧方案的范围

11 月初的旧 runtime 分支与当前主线存在架构分叉，Git 中仍有独有提交；
“当前分支已有同类能力”不等于“旧分支全部补丁已经等效合入”。

| 对象 | 可核验状态 |
| --- | --- |
| `origin/feat/runtime-resource-lifecycle-20261103@d66ee71` | 含旧 trajectory、resident backend 和资源生命周期实现，与当前主线分叉。 |
| `origin/feat/runtime-validated-pipeline-20261104@1bdaf3a` | 包含前项，并有后续运行时和启动优化；不是当前主线祖先。 |
| `origin/evidence/runtime-f102-final-20261104@f484452` | 包含 `1bdaf3a`，其后的六个提交为 evidence/docs。 |
| `origin/evidence/runtime-final-validation-20261104@e80b2bf` | 与上一系列共享旧历史；其分叉 evidence 提交在 `1bdaf3a` 历史中有 patch-equivalent 版本。 |
| `resident-af3`、`adaptive-runtime` 工作树 | 位于旧基线且含大量未提交产品代码；全后端自适应任务 `bd_224_astrolabe_architecture_handoff-5m7` 仍为 blocked。 |
| 三个 `binder-interface-*throughput` / `binder-interface-daemon-pool` 工作树 | detached 于 `1c0f1a8`，含未提交产品代码；不能只凭 HEAD 或文件名判断覆盖。 |
| `opt/runtime-startup-20261103` | ref 与原工作树已不存在，原补丁仍保留；没有精确 patch-id 相同的已提交版本。 |

本次保留现行原生模型 owner 和已接受的恢复合同，不整体导入另一套旧
runtime 架构。原工作树、未提交改动和历史证据均留在原位置。

### 启动方案：已覆盖部分与局部移植

对原补丁和 `4d9dffc` 的实现对照确认：

- `interaction._require_pyrosetta()` 已在第一次实际使用时导入，完整导入成功后
  才发布符号，保留失败原因且不在导入时调用 native 初始化。原补丁在
  `8aad56f` 快照的 `interaction.py` 上可反向通过 `git apply --check`。
- `runtime_requirements._source_segment()` 已使用一次 UTF-8 编码及分行结果。
  当前额外识别 pH 模块，继续保留这一扩展。
- 原 startup 检查已在当前分支，不能因 commit SHA 不同就再应用旧补丁。

`109f654` 的 SciPy 空间依赖延迟加载尚未覆盖，因此局部移入
`interaction.py`、`design_screening.py` 和 `repeat_symmetry.py`。
当前 `shape_complementarity.py` 也会在 loss 发现时导入空间依赖，
故一并将它的 `cKDTree` 导入移到 `_trim()`、`_median_shape()` 内；
随后冷启动检查发现 `interface_holes.py` 仍提前加载同一依赖，
也将其导入移入 `interface_residues()`，
不改 Rosetta 表面、浮点运算、邻居平局规则或 SC 统计。

复用 `109f654` 原有的两份检查及 Qhull fallback monkeypatch 适配；
没有新编写单元测试。两份移入的检查同时登记到现行 release CPU 入口。

该移植改变缺依赖时的报错时机：模块仍可完成 loss 注册，
实际调用空间几何时才报告缺少 SciPy。`design_screening` 保留原显式错误文案
并携带原异常链。已安装依赖下的算法、参数、loss 注册和现行
`PyRosettaInitializationOwner` 均保留。没有扩大缓存失效规则，
也不提供动态注册热更新或已证明的启动时间收益。

### 三个吞吐量工作树：已覆盖能力与未保留差异

审查对象为 `binder-interface-throughput`、`binder-interface-address-throughput`
及 `binder-interface-daemon-pool` 的实际 tracked diff 和 untracked 产品源码。

| 旧方案行为 | 统一分支中的实现与决定 |
| --- | --- |
| 空槽立即补交、按输入顺序返回 | `design.branch_execution.gather_ordered` 已覆盖，并在补交前验证 ready group 的错误与身份。 |
| 提案与已接纳候选评估重叠 | `design.branching.BranchingPolicy` 重写覆盖，按连续有序前缀提交；保留冻结父代、去重、配额及既有 seed 公式。 |
| 提前启动后续 CPU worker | `protein_beam._PrimedWorkerPool` 重写覆盖，不创建额外候选。暖启动异常观察方式与旧 sentinel 有差异，不宣称逐字等价。 |
| 多常驻 AF3 daemon 有界分流 | `evaluators.af3_pool.DaemonPool` 保留负载计数、轮转及单次转发，并增加失败隔离与关闭排空。 |
| staged RPC、完成等待、缓冲区保活 | `inference_service.serve_staged` 与旧版 AST 相同；`inference_pipeline` 保留容量和 drain，新增 profiling。 |
| SC/BUNS 检查、配额、父进程退出保护 | 检查已保留；其他职责由 `design.quotas`、`process_lifetime` 承担。 |

没有再移植这些旧 dispatcher 或 pool。仍存在两类真实差异：

- 旧 AF3 把 prepare/upload/model/materialize 交给独立 executor，允许旧式
  跨请求重叠。当前由统一 owner 管理并声明 `pipeline_slots=1`。
  恢复旧策略会改变同时存活的设备输入、输出及请求数，并涉及准入和 RNG
  上传边界；缺少符合现行合同的设备级证据，因此本次不恢复。
- address 工作树的 `af3_diagnostic_service.py` 是旧诊断工具，按第二次
  `prepare_features` 调用假设 warm request，以主机返回计时；
  不作为待补吞吐能力或设备完成性能证据。

`interface_holes.py`、`protein_beam_inputs.py`、`protein_beam_policy.py`、
`protein_beam_seeds.py` 与三份旧快照逐字节相同；
address/daemon-pool 的 `inference_backends.py` 也相同。
其余为上述行为级比较，没有宣称所有脏文件等价。

## 本次验证

本次验证通过。验收快照为 `eb762ee` 加已审查的
`test_af3_lifecycle.py` fixture 修复；随后提交和文档更新不改变已验证的
产品及测试内容。主项目任务记录为 Beads `Astrolabe-aga`。

| 检查 | 结果与口径 |
| --- | --- |
| CPU pytest 组合证据 | **2843 passed、1 skipped**，共 2844 个唯一 nodeid。由初轮、必要环境重测及受影响部分复验组成，不是修复后全量重跑。 |
| 日常 CLI | 145/145 案例、8/8 校准通过；最后两处修复未影响的边界复用先前结果。 |
| 四个 AF3 CPU echo 生命周期场景 | 修正隔离 fixture 后 4/4 通过，原服务和恢复断言保留。 |
| cold/RNG、Holes 几何/mock、registry/资产检查 | 修复后 22/22 通过。原冷启动脚本完整执行，Python、NumPy、Torch RNG 断言均到达且通过。 |
| recovery self-test | 10 项模型/代码正负控全部符合预期；预算 1/2、18 项代码正控、三种模型反例、三种源码 mutation、恢复原代码复验均完成。 |
| 公开源码导出、wheel/sdist、临时非 editable 安装 | 修复后快照全部通过；包内产品修复与快照逐字匹配，源码包内 fixture 也匹配。 |
| checkout 外 CLI | 安装后的 `--version`、`--help` 和 `release_protein --dry-run env=env` 通过。 |
| profiling 官方 CPU 示例 | 配置 dry-run、无模型三步序列组成运行及报告生成通过。 |

初轮 execution 的 2792 个执行实例包含重复收集的 30 个 nodeid，
因此不能将运行次数相加当作不同检查数量。上表使用报告中的去重结果；
CLI、TLC 和故意破坏源码的负控不混入普通 pytest 数量。

唯一 skipped 是原有真实 GPU recovery checkpoint 不可用项。
另外定向 Holes 检查明确排除一项依赖正式 `/case` 工程和真实 PyRosetta
初始化的用例，其余七项几何/mock 检查通过；没有将该排除项计为通过。
本次没有真实模型推理、GPU 或新的端到端性能测量。

环境为既有 a13 SIF、Python 3.11.14、pytest 8.4.2及只读测试依赖，
强制 CPU、数值线程数 1；TLC 复用校验和匹配的原 jar 与 OpenJDK 8。
构建使用既有 `setuptools.build_meta`，安装复用既有依赖。
这不是 Python 3.11/3.12 的 clean-install CI 全矩阵验收。

完整命令、最初失败与修复后结果分别保留：

- [修复后报告](/tmp/astrolabe-opt-integration-20261113-RvffyK/REPAIR_REPORT.md)
- [最初报告及环境修复](/tmp/astrolabe-opt-integration-20261113-RvffyK/REPORT.md)
- [recovery 正负控结果](/tmp/astrolabe-opt-integration-20261113-RvffyK/approved-runtime-recovery-complete/result.json)
- [唯一 nodeid 结果](/tmp/astrolabe-opt-integration-20261113-RvffyK/unique-nodeid-results.json)
- [最终源码身份](/tmp/astrolabe-opt-integration-20261113-RvffyK/repair-source-identity.json)
- [产物身份](/tmp/astrolabe-opt-integration-20261113-RvffyK/repair-artifact-identity.json)

所有本次运行句柄已结束，执行进程已收束。

### 检查发现及修复

- 冷启动检查实际发现了 `interface_holes.py` 的遗漏导入路径，
  修复后已重新完成冷注册及首次空间依赖导入的 Python/NumPy/Torch RNG 检查。
- Mol-1 空配置在 registry 构建描述符时抛出 `KeyError`。
  `local_assets()` 现在对缺失或无效的 `repo_path` / `mol_root_dir`
  给出明确 `ValueError`，由现有 registry 标记为资产不支持内容验证；
  不加载模型、不授予恢复资格，也不放宽真实源码、参考数据或权重检查。
- 四个已有 AF3 CPU echo 生命周期检查依赖外部 `/case/...` 路径。
  它们的合成输入 fixture 已隔离到测试临时目录，保留原服务复用、
  完成后恢复、prepare-only 和环境恢复断言，不修改正式研究输入。
- recovery 的四个源码指纹随已合入优化发生变化。审查后同步至
  `7d48b5f` 的对应源码，并将必经的 `design/continuation.py`、
  `profiling/core.py` 纳入指纹清单；不绕过原模型和代码正负控。

recovery source-sync 的批准范围保持为**单 owner、单活跃调用、
`reuse_durable_closures=false` 的 journal 合同**：
issued 在调用前持久化，未知执行不重放，未知结果不退还预算。
TLC 检查预算 1/2，现有代码检查预算 1。
这不验证同一 owner 下多个悬挂调用交错、缓存开启路径、真实 transport
悬挂恢复或 native/GPU 生命周期；不把普通 18 项行为检查当作完整 self-test。
