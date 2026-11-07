# Evaluator 与 Score 的执行合同

本页描述本开发树实际接入的边界；配合[开发指南](../developer_guide.md#execution-boundary)
使用。阶段名称不构成后端完成认证，CPU doubles 不构成 GPU 或数值验收。

## 来源与迁入范围

本批复用 `generic-evaluator-pipeline` / `evaluator-stages` 开发线的
`KernelStages`、`EvaluatorStages`、完成等待、资源准入和 Score worker，
保留本树的构造 RNG 隔离、构造失败回滚及既有 NativeModelClient。
来源是包含未提交实现的源码副本，因此交付证据使用冻结文件 SHA256 与增量补丁，
不能仅用其旧 Git HEAD 表示来源。

| 路径 | 当前职责 | 不属于该路径的工作 |
| --- | --- | --- |
| `model_execution.py` | 原生 owner 内的阶段执行、全局 RNG 串行区和真实完成等待 | proposal、成员选择、评分公式 |
| `runtime/admission.py`、`telemetry.py` | 已显式启动 broker 时的容量、设备占用与生命周期 | 创建科学运行、改变搜索预算 |
| `runtime/loading.py`、`evaluation.py` | 复用阶段实现所需的加载/执行控制类型；默认未启用自适应 | 宣称模型已支持并发或有吞吐提升 |
| `evaluators/af3_pipeline.py`、`native_daemons.py` | AF3 native daemon 的完整模型边界与协议验证 | 根据通用声明替其他 backend 认证 |
| `design/native_refinement_runtime.py` | 持久 raw、Score、member 三个完成点及身份核对 | 将未完成的设备请求自动重发 |
| `design/refinement_worker.py` | 进程私有的候选、路径、Pose、评分上下文与回执 | 父控制流程的 proposal 和成员提交 |
| `design/configured_refinement.py` | 串行配置入口；与 worker 共用 `measure_configured_scores` | 在本批启用并行搜索 |

本树没有迁入并行搜索 owner、runtime CLI/session/model pool 或动态供给配置。
现有公共 RPC pipeline 保持原路径；AF3 native daemon 使用严格阶段合同。
其他原生 factory 仍为 callable，按保守完整调用执行，未伪装为三阶段。

## 阶段的资源与数据生命周期

`EvaluatorStages.prepare` 包括特征处理与输入转移；`execute` 包括完整模型算法；
`finalize` 保持原生归约、dtype、计算顺序、结构提取和输出物化。

1. 请求容量在 prepare 前取得，finalize 完成后释放。
2. 推理占用只覆盖完整模型 execute，包含 `Pending` 的真实完成等待。
3. GPU prepare/finalize 仍占容量，但不占模型推理槽。
4. 原始 tensor 进入原生 finalize 前不做通用 host 转换。
5. 无 broker 时仍保持完成等待与默认 RNG 串行；不凭空声称已执行容量准入。
6. 设备完成失败时，不把异步 dispatch 当作成功；存在未知设备工作时隔离 owner，
   不自动重发。协议错误应在设备准入前拒绝，健康模型可继续接收正确请求。

AF3 原生 owner 加载时按配置的可见设备 ordinal 放置权重，并等待放置完成；
每次请求的 seed 在 owner 中创建，seed 与输入转移在 prepare 完成。
结构提取属于 owner 的 finalize，调用方只消费返回结果。显式 GPU 不可用时报错。
公共 RPC 保留原执行路径，但复用同一权重加载函数。

AF3 owner 的 cyclic 配置固定。请求配置不同必须显式报错，不能就地修改共享模型
或回退到未分类调用。需要 cyclic 输入时，在构造 owner 的模型配置中预先设置
`cyclic_asym_ids` / `cyclic_relative_patches`，请求必须与之匹配；不同配置使用不同 owner。
显式配置、输入、权重、seed 和源码身份仍须随科学验收保存。

## Raw、Score 与成员提交

串行 durable refinement 使用以下持久轮次：

```text
prediction → RawPredictionReceipt
score      → ScoreReceipt
measurement → BranchMember
```

`ScoreReceipt` 包含 `request_identity`、`raw_scores`、`scoring_identity`、
`measured_scores`、`measurement` 和 `after_rng`。串行 owner 的请求身份由
job、prediction blob、protocol 和 output 共同计算；worker 回执使用其父请求
`_request_hash`。二者各自校验完整请求，不能将不同请求命名空间的回执直接互换。

评分恢复独立的 `StandardScores` 和候选上下文，调用原 `LossEngine`，不改变
loss 注册顺序、归约或表达式。成员提交核对 Score 的请求、raw 和 protocol，
恢复评分后的 RNG，不再加载 evaluator 或再次计算已完成评分。

| 中断位置 | 必须保留的行为 |
| --- | --- |
| 预测已发出、无完整 raw 证据 | unknown，禁止自动再推理 |
| raw 完整、Score 尚未发出 | 从 raw 进入 Score，不重复推理 |
| Score 已发出、终态不明 | 串行 Runtime 保留 unknown，由原 owner 核对 |
| Score 完整、member 尚未提交 | 使用 ScoreReceipt 提交，不重复评分或推理 |
| worker 的 raw 完整、私有 CPU Score 失败 | 核对 raw、sidecar、请求与进程锁后，允许只恢复 Score |
| 任一身份、blob 或 sidecar 不匹配 | 拒绝消费，不将其当作成功结果 |

本批 durable owner 版本为 `astrolabe.native-refinement-recovery/v2`。
旧 codec 内容仍由既有读取器解释，但旧运行不能跳过 owner/protocol 核对而直接续跑。
历史科学输出和失败回执保持原身份。

## 测试与验收标准

验收触发规则见[开发指南的验收配置](../developer_guide.md#validation)。
持久 prediction 和 Score 共用的 `Runtime.call` 另由
[`verify-runtime-recovery`](../../.agents/skills/verify-runtime-recovery/SKILL.md)
检查已发出工作不自动重放、预算不退还。
运行 `python scripts/validation/verify_runtime_recovery.py --self-test`。
该形式模型不覆盖 ScoreReceipt 内容、member 提交或科学数值，仍须执行下表测试。

| 测试文件 | 验收标准 |
| --- | --- |
| `tests/test_evaluator_stages.py` | 内存和推理占用生命周期分开；Pending 完成前不 finalize；错误释放或隔离符合合同 |
| `tests/test_kernel_stages.py` | 构造一次、请求 RNG 次序与失败后状态保留；阶段完成先于后续调用 |
| `tests/test_evaluator_runtime_stages.py` | 调用真实 AF3 factory 和 adapter、使用 vendor doubles；保持原始输出类型，错误协议不准入且健康 owner 可继续 |
| `tests/test_inference_pipeline.py` | 真实线程池、Pending 生命周期、错误传播与 RPC 响应；等待服务的 `service_ready` 后调用，socket 出现不代表权限设置已完成 |
| `tests/test_af3_execution_boundary.py` | 实际 wrapper 的设备选择、权重放置、seed 转移和结构提取边界；synthetic devices，不加载权重 |
| `tests/test_refinement_worker.py` | 实际 spawn、候选/路径/RNG 隔离；raw 与 Score 分离；损坏和 unknown 不导致重新推理 |
| `tests/test_native_refinement_recovery.py` | 各完成点 crash/resume 与不中断路径的候选、成员、分数和最终 RNG 相同；错误 Score 身份拒绝提交 |
| `tests/test_configured_refinement.py` | 实际配置 wiring、完整候选分配、科学配置与资产变更拒绝、已有回执恢复 |
| `tests/test_resource_admission.py` | 真实本地 broker/RPC 与合成设备采样下的容量和清理合同 |

这些文件由[同一验证入口](../../scripts/validation/release_cpu.py)分层执行：

```bash
python scripts/validation/release_cpu.py -q                    # 发布 CI / just test-cpu
python scripts/validation/release_cpu.py --suite execution -q  # just test-execution
```

发布清单只新增 `test_evaluator_stages.py`、`test_evaluator_runtime_stages.py` 和
`test_resource_admission.py`，不要求 Torch/JAX/模型安装。execution 是发布清单的超集，
补充 kernel、worker、持久恢复、配置 wiring、现有 daemon 和 AF3 回归。
execution 需既有科学运行时及 Torch、AF3 依赖；后端缺失引起的 skip 不是验收通过。
两套清单均透传 pytest 参数和退出码，`--collect-only -q` 可核对实际选择。

改动共享核心时仍按开发指南运行完整 pytest；完整 suite 的缺依赖、历史 fixture
缺失和未通过项单独记录。真实后端验证必须使用同一科学配置、输入、权重和 seed，
经公共 `astrolabe run` 入口比较预测、分数、选择、RNG 和恢复产物。
本页描述验收标准，不将源副本的旧 GPU 测量转述成本树已通过的结果。

### 运行环境和证据保留

CPU 执行测试使用 Unix socket、进程锁、SQLite 和真实文件同步。
在计算节点运行时，为每次检查分配独立的节点本地临时目录；容器内的
`/tmp` 也须指向该目录。共享目录用于保留日志和 JUnit。改变临时目录后重跑
受影响的完整清单，保留原失败；一次定向重跑通过不能证明间歇失败的唯一原因。
不要通过放宽超时或删除断言取得成功结果。

`test_gpu_v5_persistent_checkpoint_rehydrates_proposal_report_ids` 读取已有
`outputs/rna-archived-gpu-acceptance/gpu_v5/recovery/configured-output/001_seed_67101/.framework`。
独立工作树通过只读挂载使用经过内容核对的完整副本，包含 journal 及其引用的 blob。
挂载必须覆盖实际运行该测试的每个容器。只有空目录不构成 fixture；缺失时记录
未验收，不能生成一个替代的“历史”运行。检查前后确认原始数据没有变化。

真实 AF3 对照同时设置 `save_all_samples=true` 和持久的
`raw_model_results_dir`，否则成功后的临时目录清理可能移除样本证据。
记录每个样本、排名、选中结构、Score、RNG 和恢复结果；只比较最终总分不充分。
当前 AF3 prepare 在 CPU 上建立 seed，GPU 验收环境须同时提供 CUDA 和 CPU
JAX backend。容器、JAX、JAX-Triton、Triton 及只读覆盖依赖共同定义运行环境；
使用兼容覆盖层取得通过时，不能将其记作原始镜像也已通过。
