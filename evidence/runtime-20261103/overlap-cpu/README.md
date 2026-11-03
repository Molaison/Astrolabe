# CPU 跨 owner 重叠：原始证据核对

本目录来自 snapshot04 的真实 CLI / broker CPU 替身测试。核对只读取既有日志、scores 和 provenance，未重新执行工作流或测试。结论是**共享 worker、真实 CPU 工作重叠、控制器试增/退让和科学结果等价**的机制证据；当前仍没有完成 CPU 性能认证，也没有 GPU 性能结论。

## 1. 共享进程与计数

- owner PID 为 `2849413`、`2853403`；10 条 resident 事件（每 owner 5 条）均指向 worker PID `2851976`，reported `loads=1`。
- trace 只有 1 次 `model_loaded`，同一个 worker epoch；10 个完整请求的 `completed_requests` 连续为 1–10，10 个 transport ID 和 10 个 scientific request ID 均唯一，两个 trajectory 均完成。
- 此处 `load_count=1` 是 fixture daemon 声明的计数。trace 明确 `weights_loaded=false`、`gpu_resident=false`，不能写成真实模型权重只加载了一次。

## 2. 实际 CPU hash 重叠区间

时间均为同一主机日志中的 monotonic 秒；持续时间是区间墙钟时间。CPU 段在 `cpu_work_start/end` 之间实际执行 150,000 次串行 SHA-256；不是将等待 prepare 的时间当成 CPU 计算。10 段合计 1,500,000 次 hash，区间墙钟时间之和为 `1.055001168 s`。

| CPU owner | CPU hash 区间（秒） | peer owner / forward 区间（秒） | 交集（秒） | CPU 全段落入 forward |
|---|---|---|---:|---|
| 2849413 | 32442.874052825–32442.978834036 | 2853403 / 32442.868457720–32443.783522404 | 0.104781211 | 是 |
| 2853403 | 32443.841326561–32443.943622801 | 2849413 / 32443.841488139–32444.642652269 | 0.102134662 | 部分重叠 |
| 2849413 | 32444.689841350–32444.793923270 | 2853403 / 32444.698911372–32445.499762560 | 0.095011898 | 部分重叠 |
| 2853403 | 32445.553625625–32445.656868213 | 2849413 / 32445.558724020–32446.359466980 | 0.098144193 | 部分重叠 |

完整包含的 CPU 段持续 **0.104781211 s**，对应 `events.jsonl` 第 29–30 条；peer forward 对应第 28、33 条，持续 `0.915064684 s`。一共 4 对跨 owner 交集，合计 `0.400071964 s`，其中 1 段完全包含。

这里的 forward 是 CPU daemon 的同步点和约 0.8 秒 sleep。10 个 forward 区间仍严格串行；证明的是另一个 owner 的 CPU 工作可以在某次请求进行中执行，未证明同卡 GPU 并行。完整包含由测试 rendezvous 特意保证，不能外推自然工作负载的收益。

## 3. 控制器确实试增并退让

| monotonic 秒 | 上限变化 | reason |
|---:|---:|---|
| 32435.503868930 | 1 → 2 | `independent_supply_trial` |
| 32444.675898910 | 2 → 1 | `throughput_regression` |
| 32446.410212049 | 1 → 2 | `independent_supply_trial` |

共 508 条 controller 事件：2 次试增到 limit 2，中间 1 次因 throughput_regression 收缩到 1。该上限指独立 trajectory 的在途量；所用控制代理信号是 warm backend/shape completions。两次试增事件的 `gpu_activity=null`；第一次试增和后续退让时有效完成 trajectory 数仍是 0。不能将这些瞬时代理速率写成最终科研任务吞吐提升。

## 4. 科学结果逐字节相等

| Seed | 每模式数据行 / 列 | 每文件字节数 | 两模式相同 SHA-256 |
|---:|---:|---:|---|
| 17 | 5 / 35 | 1275 | `0119eb69c909283365958a4c659cf487e4b796854fa833468ee19487b273232b` |
| 18 | 5 / 35 | 1275 | `74432dbde7a19d25047eedb5b86935926b25a8190eefefbb4268aa87d24e4ca2` |

每个 seed 的 legacy/runtime `scores.csv` 完整字节相同，包括全部 35 列，而非仅分数或序列字段。每模式共 10 行、2,550 字节，step 均为 0–4；四份 provenance 均为 terminal/completed、error=null，且各自的 config SHA-256 与实际文件匹配。

## 证据与使用边界

- [summary.json](summary.json) 给出全部区间、源日志行号、controller 原始变更、计数、结果摘要和每个原始输入文件的 SHA-256。原始文件保持不变。
- [events.jsonl](events.jsonl)、[trace.jsonl](trace.jsonl) 提供时间线；`legacy/` 和 `runtime/` 保存科学输出、配置和 provenance。
- 本地核对的 `tests/runtime_hook/sitecustomize.py`、`tests/runtime_fixture.py`、`tests/test_runtime_execution.py` 摘要均与 `../source-snapshot-04.json` 完全匹配，可追溯 hash 循环、同步点和 sleep 的真实实现。
- legacy 分支没有运行 runtime fixture 额外的 hash / prepare sleep / forward sleep。两模式在这里用于验证科学结果，不具备公平耗时对照条件。不要据此计算加速比或资源成本收益。
- 独立 CPU 三模式性能基准及进程观察器仍由主任务处理。本整理不对它们做通过声明，不重新运行测试，也不改变实现。
