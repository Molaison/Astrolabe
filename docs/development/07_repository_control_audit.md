# 全仓入口与控制流复查

原复查日期为 2026-11-01。2026-11-06 按当前开发树修正 candidate batch
恢复说明；下方历史测试记录保持原范围，本次没有重跑。
版本边界见[开发者手册](../developer_guide.md#source-version)。
复查范围是本仓库维护的 Python 包、`run.py`、安装/checkout
入口、CLI/paper/preset 配置、脚本和 examples；第三方实现按外部工具边界检查。
本次未运行 GPU 推理。

## 复查后调用关系

| 调用方 | 实际执行路径 |
| --- | --- |
| console script、`python -m astrolabe`、checkout launcher | launcher → CLI → `Astrolabe(cfg)` |
| 普通 `run`、paper、design/refine/bench presets、旧 `run.py` | `run_repeats` → 集中配置校验和 factory；默认轨迹进入 `NativeFlow` |
| 默认轨迹 | `AcceptedSuccessorPolicy` → `coordinator.run` → `drive` |
| 配置选中的通用 native refinement | 既有组件注册、静态 stage、loss 表达式 → `NativeRefinementSpec` → `BranchingPolicy` → `ControlRun` → `drive` |
| 隔离 native jobs，包括 interface case | scientific owner → `NativeBranchTrial` → `BranchingPolicy` → journaled `ControlRun` → `drive` |
| 档案 refinement 公共兼容函数 | 科学协议适配 → 同一 serial native owner、`BranchingPolicy`、`ControlRun`、`drive` |
| candidate batch 和 collection helper | `CollectionPolicy`、`CollectionTrial` → `ControlRun` → `drive` |
| 直接 `DesignRun` / `ControlRun` API | 同一 coordinator；由调用者提供 policy、trial 和 runtime |

`drive` 是唯一反复执行 ask/perform/tell 的搜索循环。不同 policy 表达轨迹接受、
分支选择、异步收集等不同语义；不按 RNA/protein/task 名称选择控制分支。

## 本次发现和修复

- 原通用 `native_refinement` factory 实际只装配一个 RNA 档案协议。
  现通用 factory 直接组合既有组件和 loss；档案协议使用显式 factory 身份，
  不再冒充通用默认。
- 原入口缺少 `search` 或只提供一部分字段时，可能误选默认轨迹或 RNA。
  现完整、已注册的 `search.algorithm` / `search.owner` 才能选择搜索；
  未迁移的执行字段和冲突选择明确报错。
- 原旧 engine 校验只在 paper 入口生效。现所有配置执行入口在 repeat
  规划和模型加载前共享同一校验，物化后的 repeat 也再次校验。
- 分支搜索保存的配置补齐实际路由、单次 seed/output、执行 owner 参数；
  本次保存的配置不再携带会重新展开的 repeat 计划。旧专用输出缺少可判定的
  路由时应显式迁移，不自动猜测算法。
- collection helper 通过公开 `ControlRun` 生命周期执行；repeat 的 start
  数据不得自行合成一个没有实现选择的 search。
- 模型服务共用配置加载、原子 JSON 和进程寿命保护，不反向依赖专属搜索入口。
  native 会话的初始化、执行和清理使用同一个科学 owner 锁。

## 保留的循环及其边界

独立 repeats、interface case 和固定 refold manifest 是预先确定输入的批处理。
模型内部 recycle/sample、Pareto 数值计算、worker/Future 收集、服务接收和启动等待，
以及交互式配置提示属于各自计算或生命周期。它们不根据上一候选结果决定下一轮
搜索工作，不作为第二套优化控制器迁移。

外部 comparator adapters 保留原工具的优化方法。candidate batch 的一个 child
可以运行完整的 Astrolabe 搜索；外层拥有任务收集策略。
真实 `ProcessCandidateWorker` 路径使用持久 Runtime、journal 和 receipts；
兼容 worker 或 dry-run 仍不具备同样的恢复能力。
child 与 collection 的提交边界须由该路径的回执核对。
child 完成不自动证明外层 collection 已提交，无法确认时保持 unknown。

## 能力与验收边界

统一控制不等于后端能力一致。通用配置 refinement 当前支持静态 stage；
无法保持的 stage transition、接受门控和 live Pose 等配置须在加载模型前拒绝。
已存在的默认轨迹和档案科学 owner 保留自己的已支持能力与科学约束。

serial native / live Pose 的科学状态仍为 process lifetime；隔离 native jobs
使用 owner 核验的持久化 receipts。未知调用结果不能自动重试，重试继续计费。
保存配置的新实验重放与付费任务的断点恢复是不同动作。

验收使用实际配置路由、完整 Candidate/共享 segment、proposal/evaluation 顺序、
预算、失败和保存配置的 CPU 回归；静态入口扫描仅补充发现路径，不能单独证明
推理数值等价。

原复查最终冻结源码后，172 项 CPU 回归全部通过，日志
`/tmp/astrolabe-repo-audit-final.log`。覆盖配置 factory、通用静态 refinement、
公共兼容入口、interface/配额/恢复、默认 coordinator、原生生命周期和失败处理。
保存配置同时以普通 YAML loader 验证。

此前较大组检查为 386 passed / 2 failed：一次在并发修改源码时触发严格恢复
哈希拒绝，一次由前序测试的进程级信号 handler 污染了中断测试。
冻结后原预算恢复用例通过；中断测试显式设置并恢复自己的 handler，生产处理器、
超时和恢复校验均未放宽。两项及后续变更均由最终组覆盖。两组结果不相加计数。
