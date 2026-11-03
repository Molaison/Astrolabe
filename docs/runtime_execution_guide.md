# 统一执行机制：运行、验收与证据解释

## 范围与科研协议

本分支在保留 `834402daade9f62e9dec0b08041dade163a89369` 全部工作之后实现。
文档固定基线、另一条未修改工作线和验收环境见
`runtime_implementation_baseline.md`；精确后端/variant 范围见
`evaluator_lifecycle_matrix.md` 与 `mutator_lifecycle_matrix.md`。

执行模式只管理已配置的独立轨迹、模型进程及资源。原 seed 序列、RNG 消耗、
sample/recycle 数量、precision、候选选择、接受判定、stage transaction 和
fixed/shared segment 仍由原科学流程负责。控制器不会增加候选或为调参另做模型调用。
同卡 device execution 当前保持串行；原生联合解码、diffusion 和必须连续的 GPU
子图不拆散。没有真实等价性认证的跨请求 batching 或 kernel overlap 不开放。

## 启用方式

保留已有 standalone 科研配置，在其中显式选择执行模式。以下是执行字段，
不是另一套样本或模型参数：

```yaml
execution:
  enabled: true
  max_active: auto
  max_resident_models: auto
  max_resident_per_device: auto
  resources: {}
```

GPU evaluator 对应 term 使用 `execution: daemon`。例如已有 AF3 term 的名称为
`AF3` 时，在 `evaluator.evaluator_terms.AF3` 中增加该字段。mutator 默认
`execution: auto` 在 broker 中走统一执行；也可在具体 term 指定 `daemon`。
CPU mutator、Dummy 和 RosettaCrystal 保留 CPU 路径。不同后端的其余配置和权重
路径必须保持原值，不能因为启用执行机制而换版本。

```bash
python -m astrolabe --config-dir config run --standalone /absolute/path/workload.yaml --no-chdir
```

在作业调度器分配的 shell/容器中运行，继承真实 `CUDA_VISIBLE_DEVICES`。
没有分配时不会自动搜索或借用空闲 GPU。owner trajectory 的 GPU 可见性被清空；
实际模型解释器只看到 broker 为其绑定的获配设备。物理 UUID 决定准入 owner，
CUDA index 与 UUID 别名不能创建两个独立账本。MIG 需要独立的 slice/parent 适配，
当前拒绝猜测其所有权。

同一用户在多个容器运行时，必须让 `ASTROLABE_GPU_OWNER_DIRECTORY` 指向共同的
宿主目录，或由外部调度器保证独占设备。默认目录是 `/tmp`，不能使用每个作业
独立的 `TMPDIR` 作为物理 GPU 所有权域。跨用户及其他应用的 GPU 分配仍由外部
调度器负责。

## 资源预算

`max_active` 限制独立轨迹，控制器从 1 开始；`auto` 的上限受实际 CPU allocation
和已配置 repeat 数约束。`max_resident_models` 和 `max_resident_per_device` 是
注册数量上限，不能解释成可用显存；`auto` 的注册上限为 32，实际加载仍须通过
常驻显存与当次计算准入。

`execution.resources` 支持以下字段。字节值必须是整数，不会隐式改变输入规模。

| 字段 | 管理对象 | 默认来源/行为 |
| --- | --- | --- |
| `cpu_slots` | 已准入 CPU 工作槽 | affinity、cgroup、Slurm 的有效边界 |
| `ram_bytes` | 当前执行进程树 RAM 上限 | 当前作业可用的内存限制 |
| `prepared_bytes` | 已准备、等待/进行中的请求资产 | RAM 的四分之一，最多 1 GiB |
| `output_bytes` | 作业输出预算 | RAM 与 8 GiB 中较小值 |
| `request_output_bytes` | 未知输出大小的单请求声明上限 | 作业输出预算的一半，最多 1 GiB |
| `pending_items` | 在途请求数量 | 32 |
| `vram_fraction` | 获配物理卡显存预算比例 | 0.90 |
| `telemetry_interval` | 资源采样间隔，秒 | 1.0 |
| `telemetry_timeout` | 单次资源查询时限，秒 | 2.0 |

未知值保留为未知：模型加载报告缺失不能当成零显存；单请求输出 cap 是明确的
资源限制，不是对实际产物大小的测量。大输入需要在原输入与科学参数不变的条件下，
为其配置足够的作业预算；超限会明确失败，结果不截断、不偷偷降精度或减样本。
已驻留权重和 allocator/context 内存持续占账，当次临时计算配额单独管理。
标准 tree codec 在写出前强制检查 cap；vendor 直接写文件、进程 RSS 和 VRAM
使用采样检查并终止超限工作，不能解释为内核或文件系统的硬 quota。

prepare、execute、finalize 使用一个科学请求身份和保留中的预算。
已经确认的 CPU 阶段可以与另一请求的设备阶段重叠；复合 native API 的时间
明确标记为 composite。外部模型子进程及其 GPU embedding 属于其 worker 的
资源树。guardian 在 CUDA-capable 程序执行前登记其身份，并在 broker 异常退出后
继续持有设备锁，直至关联进程树退出；仅删除 Python 对象不能释放资源账本。

进程枚举不是原子快照。已登记 PID 没有出现在一次枚举中时，会按原 PID/start-time
单独复核；它仍存活就继续保有资源。已登记进程的权限、I/O 或格式错误保持未知，
停止新准入并保留所有权。与本作业未建立关联的不可读 PID 只作为有界的发现限制
计数，不会被认领为本作业进程，也不会成为已释放的证据。

## 自适应控制与空闲解释

`execution.control` 的默认观察窗口为 5 秒，至少 4 次稳态完成，冷却期 10 秒。
试增在途量需要独立的剩余任务和资源余量；观察同一后端/输入形状的完成率，
最小保留收益为 5%。吞吐下降、失败/OOM、RAM/VRAM 压力会触发收缩或拒绝继续
准入。模型私有实例标识与性能分组分开，真实代码、权重、环境及初始化配置仍保留。

GPU activity 是控制和解释信号。活动低但没有独立供给，不能据此增加科学工作；
CPU 供给饱和、设备已有 ready 工作、工作形状改变、遥测不可用等都有单独原因。
短工作流可能在获得足够稳态观察前结束，这时维持保守执行是正常结果，不能标记
为已经证明自适应加速。

## 请求、错误与释放

科学请求绑定现有 trajectory、step、accepted revision、assessment、目标 stage/
gate attempt、输入内容摘要与参考结果身份。seed gate 尚未产生结果元数据时，
使用其已经存在的 transaction plan 传递只读上下文。执行器不是第二个接受状态
写入者；返回时检查原轨迹是否变化。

每个传输阶段另有请求 ID、client epoch 和 worker epoch。重复、乱序阶段、
错误身份和迟到结果不能提交。worker/连接丢失后的结果未知不会自动重试、换 seed
或新增候选。只有模型释放得到确认后才写入 trajectory `completed`；释放失败
保留诊断文件并写入失败，已经发生的科学异常不会被清理异常替换。

## 固定工作量的三模式对照

先检查完整运行计划，不启动模型：

```bash
python scripts/validation/runtime_workflow_benchmark.py \
  --config /absolute/path/workload.yaml \
  --direct-repo /absolute/path/frozen-legacy-checkout \
  --out /absolute/path/empty-plan-directory \
  --repetitions 3 --plan-only
```

在已经获配的 GPU 作业里，提供明确总计时预算；`ASTROLABE_BENCH_SECONDS` 是你为
本次实验选择的秒数。此预算覆盖所有试验及失败时间，不是每个请求另有同额预算。

```bash
python scripts/validation/runtime_workflow_benchmark.py \
  --config /absolute/path/workload.yaml \
  --direct-repo /absolute/path/frozen-legacy-checkout \
  --out /absolute/path/empty-evidence-directory \
  --repetitions 3 --walltime-budget "$ASTROLABE_BENCH_SECONDS" --require-gpu
```

工具轮换直连、常驻串行、常驻自适应的顺序，以固定配置的所有 repeats 为一次
trial；三轮共九次 trial。全部调用都计入预算，没有额外 warm-up 样本。
科学字段 hash 一致性在执行前检查，时间由外层独立单调时钟测量，性能试验不启用
profiler。原配置的相对输入路径以配置文件所在目录解释，因此请先检查计划中的
工作目录。

配置中的 `${seed}` 等插值保留到原有每条轨迹赋 seed 之后解析，不提前冻结成
base seed。计划列出每个预期的 repeat/seed 身份；重复、缺失或额外身份都无法通过
完成门槛。计时使用独立进程退出观察，遥测查询和采样间隔不会延长已完成工作的计时。
独立观察以 30 ms 间隔核对真实 PID/start-time，并等待已观察的子进程正常退出；
CLI 先退出、daemon 仍占资源时不能算完成。正常清理宽限最多 5 秒，随后只清理身份
仍匹配的进程；强制清理或未知释放会使该 trial 失败并停止后续对照，清理成本计入耗时。
任意未观察到的脱离进程树行为仍需专用作业 cgroup 约束，不以轮询宣称完整 containment。

`--direct-repo` 使直连组确实导入冻结的旧 checkout，同时使用同一个 Python 环境和
完整配置。省略时比较的是当前 checkout 的 legacy 路径，计划中会明确标注，不能
称为旧提交的独立对照。Mol-1 在保留基线中只有 daemon 实现，没有旧直连路径；
该后端的 native reference 需要单独提供并验证，工具会明确拒绝凭空制造直连组。

输出目录必须为空，既有证据不会被覆盖。主要文件是：

- `plan.json`：完整配置、源码状态、设备分配、命令和交错顺序。
- `report.json` / `report.md`：完成吞吐、耗时、成本、失败及科学等价性。
- 每个 trial 的 `stdout.log`、`stderr.log`、`telemetry.jsonl`、`runtime.jsonl`。
- 每个 trajectory 的 provenance、scores、原生结构和保留的模型输出。

数值容差来自重复直连的逐字段误差，接受/stage/序列差异单独检查。
有可消费响应池的后端（如 DPLM）还必须取得相同初始缓存状态的证据；通用工具
不会擅自删除或克隆外部共享缓存。缓存初态、命中/消耗次数和 native 中间结果的
等价性由对应后端的 conformance 实验记录，不能由 score 相等推断。
已输出 PDB/mmCIF 的原子身份和逐坐标误差也与重复直连比较；文件精度下的结构
等价不能替代 native tensor、未输出候选或缓存中间量的等价证据。
不完整、科学比较失败或释放失败不能计入有效完成吞吐。CPU 采样成本注明为可能
漏掉短生命周期子进程的下界；device-wide GPU 活动不被当作本作业独占计算证明。

## 原始时间线与认证边界

正常运行可设置 `ASTROLABE_RUNTIME_TRACE=/absolute/path/runtime.jsonl`。离线汇总：

```bash
python scripts/validation/runtime_trace_report.py /absolute/path/runtime.jsonl \
  --out /absolute/path/trace-summary.json
```

汇总按后端/形状/阶段区分首次与后续观察，保留加载计数、生命周期状态、资源峰值、
控制器决定和低活动期间的控制上下文。它不会把阶段完成数当作有效科学任务数，
也不会把低活动的上下文说成唯一因果解释。

CPU fixture `scripts/validation/fixtures/runtime_cpu.yaml` 用于真实 CLI 的协议与
控制开销对照，绝不代替 AF3 或其他预训练 GPU 模型验收。
逐后端真实单请求、连续复用、cache differential、并发隔离、故障清理和 GPU
收益必须在对应版本、权重、设备与运行配置下分别取得证据；状态不能因 daemon
握手成功而升级。当前尚无新 GPU 加速认证，详见本次集成报告和两份生命周期矩阵。
