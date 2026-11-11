# 配置模块 profiling

在原来的任务配置上开启 `profiling`，即可观察模块耗时、CPU 进程树和可选的
GPU 设备采样。默认关闭；采集不改变模型参数、随机数、候选顺序或资源准入。

```bash
astrolabe run --standalone task.yaml --profile --dry-run
astrolabe run --standalone task.yaml --profile
```

`--profile` 放在 Hydra overrides 之前。它等价于 `profiling.enabled=true`；
`--dry-run` 校验配置而不创建 profiling 产物。也可在同一份 YAML 中配置：

```yaml
profiling:
  enabled: true
  modules: ["*"]
  sample_interval_s: 1.0
  gpu: false
  cprofile: false
  max_events: 200000
  output_dir: null
```

| 字段 | 用途 |
| --- | --- |
| `modules` | 模块名称 glob；例如 `["evaluator.*", "score.*", "mutate.*"]`。`["*"]` 采集所有已接入边界。 |
| `sample_interval_s` | 资源采样间隔，正有限秒数。短间隔增加采样开销。 |
| `gpu` | 使用现有 `NvidiaSMISampler` 采集设备 UUID、利用率、显存和进程；查询失败保留为未知。 |
| `cprofile` | 输出 CPU Python 调用统计 `.pstats`，包含明显的诊断开销。可用 `python -m pstats FILE` 查看。 |
| `max_events` | 每个进程事件上限；超限记录丢失数量，报告不把缺失当作空闲。 |
| `output_dir` | profiling 产物父目录；默认在任务的 `out_dir/profile` 下创建独立会话目录。 |

模块筛选不关闭资源采样：CPU 资源随会话采集，GPU 资源由 `gpu` 控制。首次使用可运行
不调用结构预测模型的 CPU 序列组成示例：

```bash
astrolabe run --task profiling_cpu_example --dry-run
astrolabe run --task profiling_cpu_example out_dir=outputs/profile-example
```

复用 [`config/profiling.yaml`](../config/profiling.yaml) 作为 overlay 也可开启采集。
启用 GPU 采样可加 `profiling.gpu=true`。容器需能访问产物目录和 `nvidia-smi`；
profiling 不申请 GPU，也不改变现有作业、GPU 模式或线程配置。

运行结束（包括异常清理）后，终端显示产物位置。目录中包含 `session.json`、
各进程的 `events-*.jsonl`、`summary.json`、`report.html` 和 `trace.json`。
在浏览器打开 HTML，或将 `trace.json` 载入 Perfetto/Chrome tracing，即可查看
进程与线程的模块跨度。原始事件保留 PID、线程、主机和失败状态。
会话从公共配置校验之后开始，包含后续导入、组件初始化、计算和清理；
启动器和配置解析时间不在这个窗口内。

```bash
astrolabe profile report /path/to/session
astrolabe profile compare /path/to/before /path/to/after --output comparison
```

两个命令只读原始采集数据。比较会显示配置与源码差异、模块耗时比率以及缺失证据；
它是诊断对照，不自动宣称科学等价或性能提升。

## 用现有并发配置做对照

profiling 配置选择观察范围。并发仍由现有执行配置控制，例如
[`beam_refinement.yaml`](../config/beam_refinement.yaml) 的：

```yaml
search:
  execution:
    workers: 2
    in_flight: 2
    model_sharing: shared
execution:
  cpu_slots: 2
  max_concurrency: 2
```

这些字段有不同的边界：`workers` 为隔离 CPU worker 数，`in_flight` 为在途工作
上限，`cpu_slots` 为资源准入容量；它们都不能直接当作 GPU kernel 并发数。
配置须满足现有资源合同。先固定输入、seed、模型、权重与设备，分别保存修改前后
的配置和 profiling 目录，再比较对应模块的墙钟、CPU 工作量、等待和重叠。

Evaluator 性能验收仍分两步：先用预先准备的真实输入与持续就绪队列检查推理能力和
科学一致性；然后按该服务能力检查 Beam、Mutate 和 Score 供给。已有
[`model_gpu_throughput.py`](../scripts/validation/model_gpu_throughput.py)
可用于其支持后端的受控模型对照。profiling 不会自行重放科学请求。

## 阅读指标与增加观测点

模块总耗时是包含子调用的墙钟总和，存在嵌套与并行，不能相加当作完整运行时间。
`thread_cpu_ns` 仅计当前 Python 线程的 CPU 时间；进程树资源采样包含 native
线程，但短命进程和采样间隔中的变化可能漏记。报告保留各自分母和覆盖范围。

普通主机调用跨度不等于 GPU kernel 时间。明确完成边界上的推理跨度包含原有的
完成等待，profiling 不插入 CUDA 全局同步。GPU 设备利用率包括设备上的其他工作；
采样中的进程归属和跨度重叠均不能证明设备独占或 kernel 并行。性能收益需另用
未开启 profiling 的匹配运行验收。

新模块可直接使用同一个观测 API：

```python
from astrolabe.profiling import span, event, profiled

@profiled("custom.prepare")
def prepare(inputs):
    return build_features(inputs)

with span("custom.score", candidate_id="candidate-7"):
    scores = calculate_scores(result)
event("custom.queue", ready=4, in_flight=2)
```

默认关闭时调用不产生文件。子进程通过 `ASTROLABE_PROFILE_SESSION` 继承会话，
每个进程写自己的事件文件；显式过滤环境的启动器应合并 `child_environment()`。
跨容器还需让同一绝对会话路径可见。已经运行的外部共享服务不会因客户端开启
profiling 自动添加内部观测点；客户端只能观察请求/等待边界。

Python 直接调用自定义流水线时可显式拥有会话：

```python
from astrolabe.profiling import profile_session, span

with profile_session({"enabled": True}, output_dir="outputs") as session:
    with span("custom.workflow"):
        result = run_workflow()
# astrolabe profile report <session.directory>
```

运行中断或写出失败时，事件可能不完整；报告会保留这一限制。profiling 产物只用于
观测，不能代替科学结果、durable receipt、恢复检查或未知请求的裁决。
