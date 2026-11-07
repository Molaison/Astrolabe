# Astrolabe 开发者手册

本文是 Astrolabe 在本仓库中的 canonical
开发者手册。它回答四个问题：行为应由谁负责、数据如何流动、公共参数怎样演进、
以及用什么最小证据证明修改完成。用户如何准备输入、运行、调参和验收，见
[用户手册](user_guide.md)。

第一次修改从[贡献流程](../CONTRIBUTING.md)开始。
本文用于查找当前代码的职责、数据流和扩展合同。
搜索与持久化控制的说明见[统一框架实现](development/06_unified_framework_implementation.md)。
外部仓库的借鉴及适用边界见[参考实践](#reference-practices)。

开发环境可使用 `python -m pip install '.[dev]'` 安装测试与打包工具。
模型相关测试另需对应的 vendored 运行时；验收命令见 `justfile`，当前覆盖范围见
[发布验收](release_acceptance.md)。公开源码包不依赖内部项目管理文件。

## 0. 名称、入口与基本原则

### 0.1 Astrolabe 与历史兼容名的边界

- **Astrolabe** 是当前公开的软件名、Python namespace 和唯一安装 CLI。
- **GuidedHallucination** / **GH** 只作为历史名称或环境兼容合同保留；不安装 `gh` alias，以避免与 GitHub CLI 冲突。
- 本开发树安装后的 console entry point 是 `astrolabe=astrolabe.launcher:main`。
  launcher 处理运行环境后调用 `astrolabe.cli.main`。
- 开发 checkout 使用 `./scripts/astrolabe`；`scripts/gh_dev` 仅委托到该入口。
- 内部 `GH_*`、环境组 `gh`、`~/.gh_envs`、`.gh_python_path` 不随公共 CLI 重命名；bootstrap 固定为 `scripts/install.py`。

本文中的“Astrolabe pipeline”指：

```text
输入/配置
  -> initializer
  -> initial evaluate -> post-eval -> initial score/loss snapshot
  -> 每个后续 step：mutator（可先经过 predocker）
       -> evaluate -> post-eval -> loss/context -> decision/gate
       -> accept/rollback -> stage transaction（如满足入口）
  -> 持久化输出
  -> posthoc collect/filter/refold/report
```

<a id="source-version"></a>

### 0.2 源码版本与能力边界

本手册随包含 `astrolabe/design/` 的开发树维护。
2026-11-06 核对时，发布 `main@2a8b72a`、本地开发树和 evaluator pipeline
源码副本尚未收敛。读取代码和文档时使用同一个版本。

| 源码范围 | 本文适用内容 |
| --- | --- |
| 发布 `main@2a8b72a` | CLI、组件、配置、默认轨迹、posthoc 和发布 CPU 检查；该提交没有本地 `design/` 统一控制模块 |
| 本开发树 | 包含 `DesignRun`、`ControlRun`、共享 coordinator、native registry、持久 candidate batch，以及下述 evaluator/Score 执行合同 |
| `generic-evaluator-pipeline`、`evaluator-stages` 开发副本 | 已按来源文件迁入通用阶段执行、AF3 native daemon adapter、Score worker 和回执合同；并行搜索 owner、runtime CLI/session/model pool 和动态供给仍未迁入本树，不能将整个副本视为已集成 |

Git HEAD 不标识 untracked 文件或未提交改动。
无 Git 的副本使用冻结源码清单及补丁标识。
验证记录还须包含实际配置和运行环境，详见[验证证据](#validation-evidence)。
`development/04_unified_framework_sketch.py` 和 `05_unified_framework_usage.md`
是目标草图，不能作为已实现 API 的调用示例。

### 0.3 开发的五条硬规则

1. **先找 owner，再写代码。** 不把领域行为塞进 CLI、`run.py` 或任务脚本。
2. **先找 consumer，再加 knob。** 没有已验证 consumer 和 acceptance criterion 的参数
   不进入公共配置。
3. **失败可见。** 缺配置、缺指标、缺工具和 backend 失败应在 owning path 明确报错；
   fallback 必须显式、可配置、可审计。
4. **输出是合同。** `config.yaml`、`scores.csv`、结构路径、manifest 和 posthoc DB/CSV
   都有下游 consumer，不能把“文件存在”当作内容正确。
5. **最小充分验证。** 先运行覆盖 owner 的现有测试和真实可观察边界；不同风险边界才
   增加验证层，不重复堆叠 lint、全套测试、reviewer 和人工复核证明同一件事。

### 0.4 功能增加前必须写清楚

每个非平凡修改先回答：

1. 用户可观察行为是什么？
2. 公共表面是 CLI、配置、Python API、运行输出还是 posthoc schema？
3. 哪个模块是单一 owner？
4. 当前哪个实现/registry/parser 可复用？
5. 失败应在哪一层暴露？是否真的需要 fallback？
6. 哪个最小命令能证明目标行为？哪个直接受影响的共享边界还需验证？

推荐形状：

```text
小而明确的用户表面
  -> 显式 config/data model
  -> 共享 builder / registry / helper
  -> backend 实现或 wrapper
  -> owner-boundary test / realistic smoke
```

避免形状：

```text
新 flag
  -> CLI special case
  -> compose special case
  -> run.py private runtime key
  -> posthoc 再猜一次语义
```

## 1. Repository map 与单一 owner

```text
setup.py                         astrolabe console entry point
astrolabe/launcher.py        安装入口、checkout 入口和运行环境选择
scripts/astrolabe                当前 checkout 的 canonical 开发入口
scripts/install.py               canonical server/container bootstrap
scripts/gh_dev                   唯一保留的 launcher 兼容入口
scripts/installation/            host env、asset 与 software catalog
scripts/containers/              image build、container kit 与 SLURM wrapper
scripts/release/                 export、license inventory 与 release audit
scripts/maintenance/             transient cleanup 与 vendor patch maintenance
scripts/validation/              显式 smoke/runtime validation
examples/benchmarks/             可重放性能 benchmark
astrolabe/cli.py             argparse command/handler 与 Hydra compose
astrolabe/cli_command_definitions.py
                                 共享命令定义
astrolabe/cli_interactive.py config compose/scaffold 等交互 builder
astrolabe/shell_completion.py Bash/Zsh 共用 completion engine
astrolabe/app.py             repeat lifecycle；延迟加载 run.py
run.py                           单次 run 的薄 orchestration
astrolabe/configuration.py   search factory 选择与共享配置加载
astrolabe/design/            typed 科学数据、操作、搜索控制、存储和恢复
astrolabe/model_execution.py 模型 owner、执行与 RNG 生命周期
astrolabe/argparser.py       runtime normalization、schema validation、run dir
astrolabe/run_config_lifecycle.py
                                 persisted config normalization/pruning
astrolabe/core/              输入、protomer/chain/structure 等 domain model
astrolabe/initializers/       第一个序列/结构状态
astrolabe/structure_start.py 输入结构的 reference/select/gate 语义
astrolabe/stage_manager.py   当前 stage 的解析和可见设置
astrolabe/stage_transaction.py
                                 stage entry 的 plan/attempt/decision/commit/rollback
astrolabe/stage_transition.py
                                 transition gate 数据、score context 与 artifact
astrolabe/evaluators/         evaluator registry、StandardScores、wrapper
astrolabe/post_eval_hooks/   evaluate 后、loss 前的结构处理
astrolabe/losses/            loss registry、expression engine、gate expression
astrolabe/decision_context.py
                                 所有 acceptance 路径共享的 score context
astrolabe/decision_maker.py  accept/reject、history、scores.csv
astrolabe/mutators/           proposal engine、registry、backend wrapper
astrolabe/selectors/         可复用位点选择语义
astrolabe/structure_processors/         mutator 前的结构变换
astrolabe/runtime_requirements.py
                                 task-aware env/asset requirement inference
astrolabe/posthoc/           collect/filter/refold/report 与 SQLite owner
astrolabe/analysis/          跨项目复用的分析逻辑
astrolabe/experimental/      尚未稳定的实验性公共流程
astrolabe/utils/             无独立领域语义的共享底层工具
config/                          Hydra templates、component aliases、env specs
examples/                        可复用例子和 example-specific helper
vendor/                          仓库拥有的第三方源码/submodule/wrapper
tests/                           行为与回归测试
```

### 1.1 Component 类名与配置身份

| Python package | 责任 | 公开类名示例 | 配置合同 |
|---|---|---|---|
| `astrolabe/evaluators/` | 运行 predictor/scorer 并产生标准分数 | `AlphaFold2Evaluator`、`AlphaFold3Evaluator`、`Chai1Evaluator` | 既有 `_type` key 保持稳定 |
| `astrolabe/mutators/` | 提出 sequence/structure proposal | `RandomSubstitutionMutator` | 既有 `_type` key 和运行时 `.name` 保持稳定 |
| `astrolabe/initializers/` | 构造初始 sequence/structure state | role-specific initializer classes | 既有 mode/`_type` key 保持稳定 |
| `astrolabe/structure_processors/` | mutator 前的通用结构处理 | `AutoDockVinaLigandPoseRescue`、`SymmetryRescueProcessor` | 既有 predocker/processor `_type` key 保持稳定 |

Python 类名现在直接表达角色，但配置 registry identity 不随类名迁移。`StructureProcessor` 是比 rescue 更广的通用抽象；rescue 是其中一种 role-specific implementation，并非所有 structure processor 都是 rescue。

### 1.2 Script 文件布局

| Path | Owner |
|---|---|
| `scripts/astrolabe` | source checkout 的 canonical launcher |
| `scripts/install.py` | canonical bootstrap |
| `scripts/gh_dev` | 唯一保留的真实兼容入口 |
| `scripts/installation/` | host environment、asset prefetch、software catalog |
| `scripts/containers/` | Apptainer/container build、kit 与 scheduler wrapper |
| `scripts/release/` | source export、license generation、release audit |
| `scripts/maintenance/` | repo cleanup 与 vendor patch maintenance |
| `scripts/validation/` | 独立 smoke/validation program |
| `examples/benchmarks/` | benchmark driver，不属于安装脚本 |

Post-hoc collect/filter/refold 与 experimental candidate-batch 由 canonical `astrolabe` subcommand 拥有，不再保留平行 legacy script alias。

判断 owner 的方法不是“哪个文件最方便改”，而是“哪个模块已经拥有该状态的
lifecycle”。例如：

- 新命令只在 CLI 解析 intent；设计语义进入 builder/config/core。
- stage switch 的副作用进入 transaction/transition，不进入 evaluator。
- gate 可见字段进入 shared decision context，不在 `run.py` 拼临时 dict。
- repeated post-run selection 进入 `posthoc/`，不新建任务分析脚本。
- mutator 前的结构变换进入 predocker，不把 docking 分支写进 mutator 主逻辑。

<a id="design-control"></a>

### 1.3 统一科学数据与控制

本开发树的 [design/types.py](../astrolabe/design/types.py) 定义
`System`、`Candidate`、`PredictRequest`、`Evidence` 等科学对象。
`Candidate` 表达完整赋值，不能只凭相同 sequence 字符串认定科学身份相同。
[design/store.py](../astrolabe/design/store.py) 保存可编码的数据及引用。
`ProcessResources` 中的 Pose、模型和 live score handle 属于当前进程，
不能随科学快照持久化。

[configuration.py](../astrolabe/configuration.py) 的
`SEARCH_FACTORY_CATALOG` 选择搜索实现。
evaluator 和 mutator 的 family registry 选择组件工厂。
两类 registry 分别拥有搜索与组件的配置职责。
[design/native_registry.py](../astrolabe/design/native_registry.py)
将选中组件接入 typed operation，并绑定配置、源码和资产身份。
构造和预检不加载模型。

[coordinator.drive](../astrolabe/design/coordinator.py) 执行
`ask`、`perform`、`tell`。policy 拥有选择、预算和停止条件，
trial 执行一次有界操作。policy transition 不消耗共享 RNG。
新增搜索策略复用该控制循环，新增 backend 复用组件 registry。
具体路径与限制见[控制流复查](development/07_repository_control_audit.md)。

## 2. 从 `astrolabe` 到输出的真实数据流

### 2.1 CLI 与配置组合

正式入口链是：

```text
./scripts/astrolabe 或已安装 astrolabe
  -> astrolabe.launcher 的 checkout_main 或 main
  -> astrolabe.cli:main
  -> argparse command handler
  -> _compose_hydra_config()
  -> _run_from_config()
  -> astrolabe.app.run_from_cfg()
```

`astrolabe.cli` 应保持轻量。只有真实运行被请求时，
`astrolabe.app._load_run_module()` 才动态加载仓库根 `run.py`。因此
`astrolabe --help`、completion、`astrolabe env doctor` 和 config inspection 不应被 AF3、
PyRosetta、Torch/JAX 或其他可选 backend 的 import 拖垮。

`astrolabe run` 先按固定次序收集 CLI overrides：`--overrides-file` 中的有效行、
`--overrides` 字符串、命令尾部 `hydra_overrides`。之后分两条路径：

**没有 `--standalone`：**

1. 从 `config_dir` 选择 `--task <name>`；未指定时以 `base` 为基底。
2. 将上述全部 overrides 交给 Hydra compose；config-group override（例如
   `env=env`）改变 defaults 选择，普通 dotlist 改字段，同一字段按收集次序后者胜出。
3. `--overlay` 最后由 `OmegaConf.merge` 合并。

**有 `--standalone`：**

1. 仍先选择 repo template；未指定 `--task` 时以 `base` 为基底。
2. 只把左侧名称对应 `config_dir` 子目录的 config-group overrides 交给 Hydra
   compose；其余 overrides 暂存。
3. 剥离 standalone 自己的 `defaults:`，再把其内容合并到 repo template 之上。
4. 将暂存的普通 dotlist overrides 按原收集次序合并到 standalone 之后。
5. `--overlay` 仍是最后一次 merge。

两条路径中，`--print-config --dry-run` 都打印 overlay 之后的最终组合视图，并在
进入 `app` 前退出。

不要从参数在命令行中的书写顺序猜优先级，也不要在 backend 中重新读取原始 YAML。
所有 runtime consumer 只读组合后的 config。

### 2.2 公共 Python API

配置运行传入已经组合完成的 Hydra `DictConfig`。
使用 `initialize_config_dir(config_dir=str(astrolabe.config_dir()), version_base=None)`
和 `compose(...)`，再调用 `Astrolabe(cfg).run()`。
`release_protein` 会执行 AF3 推理，需要模型资产和匹配的运行环境。
CPU 例子使用 `examples/01_cpu_reproduce.py`。

本开发树的 [app.py](../astrolabe/app.py) 也接收实际构造的 `DesignRun`
和 `ControlRun`，直接进入对应组合。
YAML 路径字符串不是以上任何一种输入。
typed 组合的可运行合同以 `tests/test_design_coordinator.py` 和
`tests/test_unified_control_surface.py` 为例；目标草图不代替这些接口。

### 2.3 `astrolabe config compose` 的边界

`astrolabe config compose` 是交互式 draft builder，不是第二套 schema：

- 默认 `:write` 只写 `<name>.yaml`。
- `:write-all` 同时写 `<name>.yaml` 与 `d.<name>.overlay.yaml`。
- `:write-overlay` 只写 overlay。
- builder 的 section 是 `basic/evaluator/mutator/loss/predocker/stages`；它必须生成
  正常 `astrolabe run` 能组合的 config，不能发明 builder-only runtime 字段。
- 非 TTY 必须早期返回清楚错误和 exit code `2`，不能抛交互库 traceback。

task-private overlay 不放 repo root；可复用 overlay/example 放 `examples/`，项目私有值
留在 owning project/task workspace。

### 2.4 App 与 repeat lifecycle

`astrolabe.app.run_from_cfg()`：

1. 将组合 config 序列化为未 resolve 的 YAML，作为每次 repeat 的干净基线。
2. 校验 `repeats > 0`。
3. 每个 repeat 重建 config、设置 `repeat_index` 并隔离显式
   `raw_model_results_dir` 子目录；当前 runtime 会在显式 `seed` 非零时逐 repeat
   复用它，只有 seed 缺失/为零时才为该 repeat 生成随机值。
4. `global_config.reset()` 后调用 `run._run_pipeline()`。
5. graceful stop 只在一个 repeat 完成后停止后续 repeat。

不要让 repeat 之间共享可变 singleton、backend handle 或未显式拥有的 cache。
每次 run 的资源由 `_shutdown_pipeline_resources()` 收尾。

### 2.5 单次 pipeline

`run._run_pipeline()` 的主路径是：

```text
parse_arguments(cfg)
  -> 创建 run dir，规范化配置，写 persisted config.yaml
initialize_pipeline_components(args)
  -> global config / protomers / cyclization
  -> initializer
  -> mutators / LossEngine / StageManager / evaluators / predockers
  -> structure_start state
load_or_evaluate_initial()
  -> StandardScores -> post-eval hooks -> final score snapshot
run_optimization_loop()
  -> mutate -> evaluate -> post-eval -> loss/context -> decide
  -> stage entry transaction（如满足入口）
  -> incremental scores.csv / structures / sequences
shutdown + owned raw-output cleanup
```

本开发树的 `run_optimization_loop()` 委托给
`design.native_flow.run_native_optimization()`，再进入共享 coordinator。
上图描述一次轨迹的科学顺序。stage transaction 和 acceptance 仍由原 owner
实现，不是 coordinator 中按模型名称分派的分支。
initializer、decision context、component registry 和 output helper
继续持有各自规则。

### 2.6 运行输出与 posthoc

典型 run contract：

```text
<out_dir>/<task_name>/<NNN_run...>/
  config.yaml
  scores.csv
  logs/
  structures/
  sequences/
  raw_model_results/   # 仅当显式/回退到 run dir 时
```

关键 consumer：

- `decision_maker.Decisions` 维护 accepted/rejected history，并增量重写
  `scores.csv`。
- `posthoc.collect` 读取 `scores.csv`、`structures/` 和 `config.yaml`，计算必要派生
  指标并 upsert 到 SQLite；文件 signature 支持增量跳过未变 run。
- `posthoc.filtering` 从 DB 选行，写 `filter_report.csv` 和筛选后的结构/DB。
- `posthoc.refold` 从 DB 的 sequence metrics 与 source `config.yaml` 生成或执行
  refold plan/manifest。
- `posthoc.report` 写 summary/per-step CSV 和报告。

改变 column、step/stage 语义、structure naming 或 persisted config 时，必须同时审查
这些 consumer。成功判定应检查行数、列、accepted/stage 关系、结构映射或 manifest
字段，而不只是路径存在。

## 3. 配置 schema、alias 与 lifecycle

### 3.1 配置是公共 API

可复用默认值放 `config/base.yaml`，task 工作流放 `config/<task>.yaml`，机器路径和
权重放 env spec 或环境变量。公共 schema 应表达领域意图，而不是 backend 调用栈：

```yaml
evaluator:
  evaluator_terms:
    Fold:
      _type: protenix
mutator:
  mutator_terms:
    Propose:
      _type: LigandMPNN
predocker:
  predocker_terms:
    Align:
      _type: hotspot_reference_align
stages:
  default:
    evaluator: Fold
    mutators: [Propose]
```

原则：

- `*_terms.<alias>._type` 定义实现；stage 和其他 component 引用 alias。
- alias 是任务级可读身份，`_type` 是 registry identity；不要互换。
- stage 引用不存在的 evaluator/mutator alias 必须在 runtime 初始化前失败，并列出
  available aliases。
- 只在跨 stage 不变时使用 global field；随 stage/oligo 改变的行为由
  `stages.<name>.*` 拥有。
- 新输入语法先进入 `astrolabe/core` parser/model，不在 evaluator/mutator 各解析
  一遍。
- 新 schema 默认严格拒绝 unknown/deprecated 字段；兼容读取必须有真实 consumer、
  明确 migration 和删除条件。

### 3.2 Stage-owned 字段

新的 runtime 配置应以 stage 为 owner：

- `evaluator`
- `oligo`
- `mutators`
- `mutator_weights`
- `mutation_num`
- `weighted_score_expression`
- `acceptance_threshold`
- `acceptance_gate`
- `early_stopping` / `prefix_early_stopping`
- 非 default stage 的 `_entrance`
- `transition_consistency_gate`

根级 `mutation_num` 和 `initial_temperature` 等历史字段仍可能存在于 base/config，
但不应成为新 feature 的 owner。`argparser._validate_schema_compatibility()` 已拒绝
`evaluator.name`、`mutator.mutation_num`、非默认 `mutator.mutation_rate` 和
`mutator.mutator_weights` 等过时写法。

`mutation_num` 可是整数、按 protomer mapping 或百分比字符串；它最终针对 mutable
length 解析并校验不超过可变位点数。不要在 mutator wrapper 中再次解释这个预算。

### 3.3 Enabled alias 与 persisted cleanup

`parse_arguments()` 根据所有 stage 计算真正使用的 aliases：

- mutator aliases 来自 `stages.*.mutators`；structure/RFD3 initializer 的
  `initializer.sequence_designer` 也强制加入。
- evaluator aliases 来自 stage evaluator、default fallback，以及 transition gate
  自己指定的 evaluator。
- predocker aliases 由实际 runtime usage 分析得到。

这些结果写入 derived `enabled_aliases`。component factory 与 persisted config 都应
复用它，不要各自推导一个近似集合。

写入 run 的 `config.yaml` 不是原始输入副本，而是
`run_config_lifecycle.build_persisted_run_config()` 生成的可重放、最小必要视图：

- 归一化兼容 initializer 字段，移除可由 owner 表达的 legacy root mirror。
- evaluator/mutator/predocker 只保留 enabled aliases 及必要 backend config。
- 本开发树保留所有显式配置且非 `None` 的 `loss_terms`，包括只用于 hook、
  条件或报告的项；递归保留所需的 `expression_terms`。
- env 只保留 task requirement 或 refold replay 需要的路径。
- AF3/Protenix refold replay 所需 backend config 例外保留。
- derived selectors、enabled aliases 等在 pruning 前注入，因此 persisted view 代表
  实际执行语义。

扩展 lifecycle 时必须同时回答：新字段是否需要 replay？谁推导它？是否应写入
`config.yaml`？posthoc/refold 是否消费？不能为了“完整”把整份 env 和所有未启用
backend 都持久化。

发布 `main@2a8b72a` 尚使用按 stage expression 裁剪 loss 的旧逻辑。
将本开发树的保存规则迁入其他分支时，须连同保存、重新加载和评分回归一起验证。

### 3.4 临时 raw outputs 的 ownership

- 用户显式设置 `raw_model_results_dir`：Astrolabe 使用它，绝不自动删除。
- 未设置：Astrolabe 尝试创建 `/tmp/gh_<run>`，写 `.gh_auto_managed` marker；不可用时回退
  到 `<run>/raw_model_results`。
- 仅成功完成且 marker 可验证时自动删除 owned 目录。
- run 失败时保留 raw outputs 用于诊断。
- marker 缺失或删除失败必须报错，不能扩大删除范围。

任何新 scratch/cache 都应同样定义 owner、marker、成功/失败 retention 和 replay
需求。

## 4. 核心 component 的 owner 与扩展清单

### 4.1 输入与 domain model

Owner：`astrolabe/core/`，包括 protomer statement、sequence parsing、folding
input、chain resolution 和 structure parsing。

扩展清单：

1. 定义显式数据字段或 typed model，不靠文件名/字符串位置暗示语义。
2. 在 core 一次解析，向 evaluator、initializer、mutator 提供已解析对象。
3. chain identity 需要无歧义时使用明确 chain/residue mapping，不猜 protomer 与
   structure chain 的对应。
4. PDB/mmCIF、非标准残基、covalent/cyclization 等边界沿用现有 domain helper。
5. invalid state 在 backend 之前失败，并说明字段路径与期待格式。

非标准残基与拓扑的稳定专题合同见
[非标准残基合同](nonstandard_residue_contracts.md)、
[环化拓扑示例](cyclization_topology_examples.md) 和
[D-n 对称性](d_n_symmetry.md)。

### 4.2 Initializer

Owner：`astrolabe/initializers/`；`run.build_initializer()` 只负责选择和 wiring。

当前 mode 包括 `random`、`dplm`、`aa_set`、`mask_mix`、`pssm`、`all_a`、
`x_mix`、`structure`、`rfd3`。每一种“第一条序列如何产生”的策略都是
`Initializer`，返回 `InitializerResult`。

结构型 initializer 的合同：

- `initializer.sequence_designer` 必须指向具备 structure-based design capability 的
  mutator alias，即使该 alias 不在 stage mutators 中也会被构建和持久化。
- `initializer.structure` / `initializer.rfd3` 持有 source、`chain_map`、必要时
  `motif_map` 与 partial mapping 语义。
- 只替换 mutable protein positions；fixed segment、locked position 和 covalent
  endpoint 不因初始化被偷偷解锁。
- RFD3 先生成结构，再复用同一个 structure-to-sequence 路径，不创建第二套 inverse
  folding pipeline。

扩展 initializer 时，在 `initializers/registry.py` 注册对应实现，
由既有 `build_initializer()` 路径装配。声明 runtime requirements，
确保 initializer-only mutator 不被 pruning，并验证 fixed/mapping/manifest/output。

### 4.3 `structure_start`

Owner：`astrolabe/structure_start.py`。它与 initializer 不同：initializer 决定
第一个设计 sequence/structure state；`structure_start` 决定现有结构怎样作为
reference、candidate selection 或 step-0/pre-accept gate。

其配置和结果必须通过 `StructureStartConfig/State` 与 `ReferenceGateResult` 流动，
reference score 字段由 shared decision context 注入。不要把 reference gate 偷塞进
initializer 或任务 CLI。

### 4.4 Stage manager、transaction 与 transition

Owner 分工：

- `stage_manager.py`：解析 stage/default inheritance，暴露 current evaluator、oligo、
  mutators、budget 和 expressions。
- `stage_transaction.py`：stage entry 的 plan、attempt、decision，以及 commit/rollback
  的有序副作用。
- `stage_transition.py`：transition consistency config/context/result、reference
  comparison、artifact/manifest 和 gate score fields。

进入新 stage 可能需要换 evaluator/oligo 并重新评估，所以它不是把
`current_stage` 字符串直接改掉。修改时必须覆盖：source snapshot、target
re-evaluation、seed-pass、active attempt、失败 rollback、成功 commit，以及
`scores.csv` 中 stage-change 行的语义。

**强制合同：** transition-consistency fields 必须经
`stage_transition.transition_consistency_gate_score_context` 产生。seed-pass 与
active-gate acceptance 必须看到同一组字段；不得在其中一个路径手写近似 dict。
default stage 不能定义 transition-consistency gate。

### 4.5 Evaluator 与 `StandardScores`

Owner：`astrolabe/evaluators/registry.py`、具体 evaluator 和需要外部进程的
`astrolabe/evaluators/wrappers/`。

Evaluator 输入标准 folding inputs，输出 `StandardScores`：

- `names` 与 oligomer-keyed `plddt/ptm/iptm/pae/pde/complex_paths`；
- protomer-keyed per-residue scores；
- chain-pair metrics；
- optional `ComplexPoses`；
- opt-in `ScoreContribution`。

公共 normalized score 在这里统一尺度与 shape。例如 pLDDT 的 `[0,100]` backend
输出会归一到下游假设的 `[0,1]`。不要让每个 loss 猜 backend 单位。

增加 evaluator：

1. 注册轻量 factory；registry 模块不能 import 重 backend。
2. wrapper 负责 subprocess、模型调用、raw parse 和 backend-specific output。
3. 返回 normalized `StandardScores`；定义每个 metric 的 scope、shape、单位和方向。
4. raw artifacts 留在 owned raw output；稳定结构路径进入 `complex_paths`。
5. 缺 env/weights/tool 在该 evaluator 被实际选择时清楚失败。
6. 验证 registry/alias、score parse、结构映射、seed determinism（若承诺）和真实
   backend smoke（若行为依赖 backend）。

逐项声明 backend 实际提供的指标。没有输出的 confidence、PAE 或 chain-pair
metric 按缺失合同处理，不能填零伪装成功。输入体系、支持的 chain 组合、
模型采样和输出归约也属于 adapter 合同。
延迟注册、加载成功和数值一致分别验证。

<a id="execution-boundary"></a>

#### 4.5.1 evaluator 执行与并行的开发合同

本树以 `model_execution.EvaluatorStages` 声明实际模型边界。
`kernel_handler` 在同一 owner 中解释阶段；AF3 native daemon 已接入该路径。
未迁移的 callable 仍按完整调用处理，不能用 evaluator 外层的
`prepare_input/run_function/prepare_output` 名称推断纯模型边界。

`EvaluatorStages` 的 prepare 包含特征处理和输入转移，
execute 包含完整模型算法，finalize 保持原生归约和输出物化。
推理槽持有到实际设备完成。请求内存从 prepare 前持有到 finalize 完成。
prepare/finalize 使用 GPU 时仍需容量，不占模型推理槽。
声明 stages 不代表该后端已证明可以并发推理。

`design/native_refinement_runtime.py` 的串行持久路径分开保存 raw prediction、
ScoreReceipt 和 member commit。ScoreReceipt 绑定请求、raw scores、评分 protocol、
测量结果与 after-RNG；提交前核对身份，已完成评分不重复执行。
`design/refinement_worker.py` 从已有副本迁入，以 candidate 为单位执行预测和 Score，
使用私有目录、评分上下文、Pose 和 RNG。当前配置 CLI 尚未接入并行 worker pool；
worker 的进程合同已有独立回归，不能据此声称搜索已经并行化。
父控制流程拥有 proposal、搜索状态和提交。
独立 loss 可以在状态隔离后并行，不能让多个线程共用可变 Pose 或 loss cache。

对应阶段、AF3 工厂 doubles 与资源准入回归已进入发布 CPU 清单。
worker、持久恢复、kernel 和实际 AF3 wrapper 的 CPU 回归由同一脚本的
`--suite execution` / `just test-execution` 执行，需配置科学运行时。
CPU 测试使用受控设备完成信号或 vendor doubles；它们不证明真实模型数值一致或性能。
原生归约设备、dtype、计算顺序和随机流的变化需要独立科学验收。
逐文件职责、验收标准和恢复边界见
[evaluator/Score 执行合同](development/08_evaluator_score_execution.md)。

### 4.6 Score snapshot 与 contribution

Owner：`astrolabe/evaluators/score_snapshot.py`、
`astrolabe/score_contributions.py` 和 evaluator score helpers。

`ScoreContribution` 是 runtime component 向统一 snapshot 增加指标的显式合同，包含
`source`、`metric_name`、`value`、`scope`、`entity`、`shape`、`aggregation`、
`flatten_as` 和 provenance。它不能覆盖已有 key；非 global scope 必须指明 entity。

稳定 score field 必须先进入 snapshot/contribution owner，再被 loss、gate、decision
或 posthoc 引用。不要先在 expression 中依赖一个只在某 backend detail dict 偶然存在
的字段。

### 4.7 Post-evaluation hooks

Owner：`astrolabe/post_eval_hooks/`；`run.py` 只调用现有 hook point。Hook 位于
evaluator 输出后、需要结构的 loss 和 final score snapshot 前。

扩展时明确：

- 是否改变 structure、pose 或只增加 report；
- 哪些 loss 需要它，未配置时是否允许自动启用；
- hook 输出怎样合并多个 oligomer；
- backend/env 需求和失败是否影响主输出合同；
- hook 字段不能泄漏到未启用的决策路径。

### 4.8 Loss 与 expression

Owner：`astrolabe/losses/base.py`、loss registry 和具体 loss module。

`LossEngine` 将 evaluator snapshot、registered loss terms 和 `expression_terms` 组合成
`weighted_score` 及 detail environment。Stage 使用 expression 定义 objective、
acceptance gate/threshold、early stopping 和入口条件。

增加 loss/metric：

1. 注册 term 及参数 metadata；不要用 custom gate dict 代替 loss/expression。
2. 定义缺失输入是 error、N/A 还是显式 exploratory opt-in。
3. 说明 metric 的 scope、聚合方式、单位和优化方向。
4. oligomer/protomer-specific term 必须使用共享 chain/protomer resolution。
5. explicit loss detail 优先于 default snapshot；禁止无声覆盖同名 key。
6. 验证 expression variable、依赖 term、missing input、flattened output 和 persisted
   pruning。

### 4.9 Decision context 与 decision maker

Owner 分工：

- `astrolabe.decision_context` 构造一个共享 `DecisionScoreContext`：
  weighted/current/previous/delta scores、stage-entry、reference gate、transition gate、
  early stop、soft acceptance 与 hard-gate result。
- `astrolabe.decision_maker` 拥有 accept/reject policy、Metropolis compatibility、
  sequence/score/stage history、mutation apply/clear 和 `scores.csv`。

**强制合同：** `run.py` 和 `decision_maker` 的 acceptance-gate shared context 必须经
`astrolabe.decision_context`。stage-entry acceptance、普通 mutation decision、
reference gate 和 transition gate 不能各自维护私有字段语义。

`force_accept` 不能绕过 active reference 或 transition-consistency gate。Hard gate
失败必须留下 rejected row；mutation state 必须 clear，不能只改 log 文案。

### 4.10 Mutator 与 selector

Owner：`astrolabe/mutators/registry.py`、具体 mutator、wrapper，以及
`astrolabe/selectors/`。

Mutator 提出下一 sequence/structure-backed proposal；它不决定最终是否接受。
`build_mutator_runtime_config()` 将当前 step/stage/oligo/selector 等共享 runtime
上下文传入 mutator。不要增加只能从 `global_config` 猜出的隐式条件。

增加 mutator：

1. 通过 `_type` 注册并声明 capability；backend runner 放 wrapper。
2. 复用 selector expression/normalized selector，不建第二套位点语言。
3. 尊重 stage-owned `mutation_num`、alias weights、locked/fixed/covalent positions。
4. 若组合多个 state/oligo，显式定义 ignored/negative/shared protomer 语义和权重
   归一化，不从列表顺序猜。
5. 记录 proposal telemetry；proposal 失败不能伪装为“无变化成功”。
6. 验证 alias wiring、selector、fixed positions、seed contract、missing resources
   和 mutation apply/rollback 边界。

### 4.11 Predocker

Owner：`astrolabe/structure_processors/base.py`、具体 predocker 和 wrappers。配置通过
`predocker.predocker_terms.<alias>` 定义，再由 mutator alias 的 `predocker` 列表
选择。详细稳定合同见 [Predocker reference](predockers.md)。

扩展时：

- 定义输入/输出结构、cache key、manifest 和是否允许覆盖 input；
- 外部 tool 调用放 wrapper；
- fallback policy 必须是显式 enum/field，并记录 backend、reason 和实际输出来源；
- geometry 改变需用小型真实结构验证，而不只 mock subprocess；
- 部分失败不能留下看似 final 的结构。

### 4.12 Runtime requirements 与 env

Owner：`astrolabe/runtime_requirements.py`、`config/third_party_envs.yaml`、
`scripts/installation/env_setup.py` 和 env configs。

- `astrolabe env doctor` / wizard 必须 task-aware，说明哪个 evaluator/mutator/predocker/loss
  需要哪个 key。
- executable、weight、repo path 和 subprocess Python 进入 env config，不硬编码在
  component。
- 可选 backend 的 Python import 保持 lazy；只有路径实际执行时才因缺依赖失败。
- 如果 backend 可在独立 subprocess env 运行，保持 CLI 主环境轻量，仅传显式
  path/env contract。
- 运行第三方依赖位于 `vendor/`（vendored source/submodule），不要用文档中的
  `pip install` 或 runtime 自动安装替代仓库声明。
- `config/third_party_sources.yaml`、`.gitmodules`、patch helper 和 env spec 的 source
  ownership 必须一致；已有 submodule path 不能被普通 clone 覆盖。

环境与资产细节见 [第三方环境与模型资产](third_party_envs.md)；调度行为见
[Slurm reference](slurm.md)。

### 4.13 Posthoc

Owner：`astrolabe/posthoc/` 与对应 `astrolabe posthoc ...` handler。稳定链路是：

```text
run dirs
  -> collect: scores/config/structures -> SQLite (+ optional CSV)
  -> filter: expression/preset -> selected rows + report + structures/DB
  -> refold: selector + source config + sequences -> manifest/new run outputs
       -> collect back into the same DB
  -> filter/report
```

普通工作流使用高层 `astrolabe posthoc refold`；`refold-export` / `refold-run` 是已有
manifest 的低层调试与恢复入口，不是第二套规范流程。

扩展 posthoc：

1. 优先扩展 DB/schema/command，而不是新增一次性分析脚本。
2. collect 必须增量、安全处理 unchanged run，并区分 missing 与 invalid metric。
3. preset-required metric 缺失默认失败；仅显式 exploratory option 可放宽。
4. file writes 用临时文件/原子替换，结构 materialization 的 copy/hardlink 语义需可见。
5. refold 必须保存 source run/config/sequence/backend 与 manifest input hash，保证 replay
   可追溯。
6. CLI、completion、DB/CSV output、smoke acceptance 同步更新。

## 5. Score 与参数的公共合同

### 5.1 每个新 knob 的七项声明

任何进入 YAML、CLI、persisted config 或 stable Python API 的参数，都必须在实现和
面向使用者的文档中能回答：

| 项 | 必须说明 |
|---|---|
| Consumer | 哪个具体 owner/function 读取；没有 consumer 就不加 |
| Meaning | 改变搜索、采样、评估、接受、停止还是资源行为 |
| Units | Å、比例、概率、步数、样本数、无量纲等 |
| Direction | 更大更好、更小更好、仅类别选择，或不是质量指标 |
| Valid range | 允许区间、边界是否包含、`null` 的含义 |
| Default/owner | 默认放哪里，是否 stage/backend/env-specific |
| Acceptance | 什么观测证明该值生效且未破坏共享边界 |

同一个物理概念只保留一个 canonical 名称。兼容 alias 必须路由同一 consumer，不能
形成两套实现。

### 5.2 实际调参流程

历史项目中可靠的调参不是“看到一个低分就改权重”，而是以下闭环：

1. **冻结比较条件。** 固定 code revision、template/overlay、backend/model、input、
   seeds、steps/repeats 和 output root；先保存最终 dry-run config。
2. **从最小真实样本建立观测。** 当执行不确定性高时用 disposable canary 缩小数量，
   不修改 production config；执行路径已知则直接 full run，并尽早检查第一批相互独立
   的完整输出。
3. **按 stage/oligo/accepted 状态看分布。** 从 `scores.csv` 或 posthoc DB 比较 count、
   missing rate、median/quantile、极端值、accepted/rejected delta，而不是只看 best row。
4. **检查贡献而非只看总分。** 对 objective 的每一项计算典型 `weight * value`，确认
   它在代表性分布上达到预期相对贡献；单位或尺度不同先规范化/变换，再调 coefficient。
5. **一次只改一个决策维度。** 搜索预算、proposal diversity、objective ranking、
   hard validity gate、early stop 分开修改；保持 paired seeds/input。
6. **用下游 acceptance 回验。** Astrolabe 内部分数改善不等于科学结果改善；必要时通过
   refold、结构检查或目标 benchmark 检验，同时保留 denominator 与失败行。
7. **冻结采用值和证据。** 记录 overlay、run dirs、summary/DB、选择理由和未解决风险；
   不从聊天记忆重建参数。

参数按 owner 分为四类：

| 目的 | 典型参数 | 观察量 |
|---|---|---|
| 搜索覆盖 | `steps`、`repeats`、stage `mutation_num`、mutator weights/selectors | unique proposals、acceptance rate、coverage、runtime |
| Backend proposal/sampling | mutator/evaluator 自己的 `temperature`、`num_samples`、recycles 等 | diversity、determinism、backend quality/runtime |
| 排序与接受 | score transforms/weights、`acceptance_threshold`、Metropolis decision setting | contribution、delta、accepted/rejected 分布 |
| 停止/有效性 | hard `acceptance_gate`、transition/reference gate、early stopping | invalid rejection、false reject、completed steps |

### 5.3 Contribution calibration

权重只在 metric 的单位、方向和有效范围已确定后才有意义。推荐在代表性固定样本上
保存以下表，而不是凭量级猜系数：

```text
term | scope | direction | missing% | q10 | median | q90 | transform | weight
     | median(weight * transformed_value) | intended role
```

校准准则：

- 先检查 score snapshot 是否已统一尺度；不要用 weight 修补 backend 单位 bug。
- 以 median/IQR 或具有明确物理意义的 reference delta 估算贡献，不被单个 extreme
  candidate 支配。
- 分 stage、oligo 和 dataset family 看分布；不同 denominator 不混成一个命中率。
- 若某项只为打破 tie，其典型贡献应明显小于主目标；若是主目标，应在常见变动范围内
  能改变排序。
- 输出 raw term 与 contribution，确保总分变化可解释。
- 在采用前做 paired comparison，报告改善、退化、tie、missing 和失败，而非只报告
  选中结果。

### 5.4 Hard gate、soft objective 与 temperature

Hard gate 只用于明确的无效态或不可继续态，例如缺 required metric、违反不可协商
几何/拓扑合同、reference/transition consistency 不合格。它不应用来把连续质量指标
粗暴二值化，以掩盖 objective 校准不足。连续偏好优先进入 score/loss；探索性阈值
使用 soft acceptance 或 posthoc filter，并保留失败分布。

必须区分两类 temperature：

- **Backend sampling temperature**：mutator/evaluator 生成 proposal 的随机性，由该
  backend config 拥有。
- **Decision temperature**：`decision_maker` 在 Metropolis acceptance 中接受较差
  proposal 的概率；只有 `use_metropolis_acceptance` 启用才生效，历史
  `initial_temperature` 属于该兼容路径。

新参数必须用能看出 owner 的名字，不能用一个全局 `temperature` 同时影响 proposal
与 accept/reject。比较实验中也不能一边改 sampling、一边改 decision temperature。

## 6. CLI 与 shell completion 合同

所有新增/重命名 `astrolabe` subcommand、alias、option、positional 或 fixed-choice value 都
必须遵守 [CLI 设计规则](command_design_rules.md)。最低要求：

1. parser/handler 只表达 intent，并复用 shared option helper。
2. `--help` 说明选择规则、path 语义、layering、derived default 和 failure mode。
3. 空调用不能意外进入 pipeline；交互命令非 TTY 返回 `2`。
4. preview/`--dry-run` 是真实 CLI 路径，不要求 execution-only assets。
5. completion 优先从 argparse parser introspection 派生。
6. parser 不足时在 `astrolabe/shell_completion.py` 增加 shared dynamic provider；
   Bash/Zsh 不维护分叉 command tree。
7. dynamic completion 必须 lazy 且 parameter-gated；只有当前 argument 需要时才访问
   SQLite、filesystem 或 config。
8. 修改 provider 前后，用同一环境和输入测量
   `./scripts/astrolabe completion complete ...` 的响应时间，记录冷、热启动条件；
   性能判定使用该变更预先确定的基线和标准。
9. completion 修改必须更新 `tests/test_cli_completion.py`，并运行真实 Bash、Zsh
   source round-trip。

真实 CLI check 至少包含：

```bash
./scripts/astrolabe <command> --help
./scripts/astrolabe <command> ... --dry-run   # 若该命令支持
```

只 import parser、只调用 handler function 或只 snapshot help 都不足以证明 shell
contract。

## 7. Dependency、failure 与 output discipline

### 7.1 Vendor 和 environment

- runtime 第三方源码归 `vendor/`；不要新增系统/global `pip install` 步骤。
- 新 source/submodule 先确认 path owner，更新 license/source inventory 与 env spec。
- backend 权重、二进制和独立 Python 环境由 env config + runtime requirement 管理。
- optional backend import 必须 late；CLI startup 不为未选择 backend 付费或失败。
- 可复用 setup flow 进入 `scripts/installation/env_setup.py`/`just env-setup`，不新增平行 bootstrap。

### 7.2 Visible failure 与 fallback

可接受：

```text
primary backend failed
  -> explicit fallback_policy=input
  -> manifest/log records backend, policy, reason, actual output
```

不可接受：

```text
primary backend failed
  -> catch Exception
  -> return input/empty score
  -> downstream looks successful
```

修 bug 时先复现 observed failure，再在 owner 修。只处理会影响输出合同或 repair/rerun
决策的失败；不要借机增加 speculative fallback、feature flag 或第二实现。

### 7.3 Output contract checklist

改变输出前逐项检查：

- 谁写、何时写、失败时是否有 partial artifact？
- schema/version、units、row identity、stage/step/accepted semantics 是什么？
- write 是否 atomic，重跑是否 idempotent/incremental？
- `config.yaml` 是否保留 replay 所需 consumer fields？
- posthoc collect/filter/refold/report 是否仍能读？
- 用户能否区分 primary success、fallback、skipped、partial 和 failure？
- 大型/临时 artifact 的 owner 和 cleanup marker 是否明确？

## 8. 文档治理

仓库入口按阅读目的分工：

1. `README.md`：名称、最短路径、文档索引。
2. `CONTRIBUTING.md`：完成一次修改的操作流程。
3. `docs/user_guide.md`：用户流程、配置、调参、运行、验收和 posthoc。
4. `docs/developer_guide.md`：owner、数据流、扩展与验证合同（本文）。

根目录 [CONTRIBUTING.md](../CONTRIBUTING.md) 是贡献流程入口，
按定位代码、实现、验证和提交组织步骤，链接本文的接口细节。
它不复制一份组件参考手册。

独立文档只保留范围窄、跨任务稳定且会长期独立维护的 reference：

- [CLI 设计规则](command_design_rules.md)
- [第三方环境与模型资产](third_party_envs.md)
- [Slurm](slurm.md)
- [Predocker](predockers.md)
- [非标准残基合同](nonstandard_residue_contracts.md)
- [环化拓扑示例](cyclization_topology_examples.md)
- [D-n 对称性](d_n_symmetry.md)
- [发布检查](release_checklist.md)
- [变更记录](changelog.md)

新功能的用户说明进入 user guide，owner/extension 说明进入本文；不要为每个 feature
再建 quickstart、tutorial、notes 或 utility inventory。一次性计划、评审、状态、
执行证据和交接保存在独立的项目记录中，不混入公开文档。
项目私有科学分析和 run evidence 留在项目 workspace，
不是 repo docs。

文档修改必须：

- 以当前 CLI/config/runtime 为真，不复制已过时的命令；
- 保留 unresolved field、backend 限制与 blocker，不用流畅文字掩盖；
- 不把测试名或 commit message 当作唯一用户说明；
- 删除文档时同步修正 canonical 入口和 repo 内链接，不留下替代索引表维护负担。

代码、配置例子和当前实现说明在同一个 PR 中同步。
运行证据保留执行时的源码身份和覆盖范围。
提案、历史审查与当前接口分别标注，不以修改文档的日期更新历史验收结论。

<a id="validation"></a>

## 9. 最小验证阶梯

### 配置本项目的验收

验收按改动涉及的行为选取。开发者先确定本次适用行，交付时逐行给出结果。
以下命令均在目标 checkout 根目录、同一已配置的开发环境执行。

| 触发条件 | 必须执行 | 通过条件 |
| --- | --- | --- |
| 每个 PR 的公共软件接口与分发 | `python scripts/validation/release_cpu.py -q`，沿用现有 CI | 命令正常退出且退出码为 0；所声明能力没有因缺依赖跳过 |
| 修改 `Runtime.call`、日志写入、预算，或形式验证覆盖记录中的依赖 | `python scripts/validation/verify_runtime_recovery.py --self-test` | 源码同步检查通过；正常模型与真实代码通过；故障变体被检出；恢复后的原代码再次通过 |
| 修改 evaluator 阶段、Score、daemon 或恢复 wiring | `python scripts/validation/release_cpu.py --suite execution -q` | 生命周期、错误传播和恢复合同通过；适用后端缺失或跳过仍是未验收 |
| 修改核心 evaluator、mutator、decision、loss 或 `StandardScores` 共享抽象 | `python -m pytest -q` | 完整回归正常结束；新增失败修复，既有失败由同环境基线对照证明并保留，不把它们记为通过 |
| 修改实际模型算法、设备放置、RNG 或声称性能改善 | 复用该后端现有 acceptance 脚本，经公共 `astrolabe run` 执行固定输入与共同配置 | 在运行前固定比较器、科学容差或严格一致要求；检查预测、评分、选择和恢复，性能另测完整流程 |

纯文档本地检查仍用 V0。CI 对所有 PR 运行发布 CPU 和恢复形式验证，
本地无需为未涉及的后端运行真实模型。execution、全仓回归与 GPU 验收仍由对应
开发环境执行，不能用公共 CI 的 CPU 结果替代。

形式验证工具、范围、源码 SHA256 和同步提交在
[`verification/runtime_recovery/coverage.json`](../verification/runtime_recovery/coverage.json)
配置；覆盖源码变更会让检查失败，要求重新确认模型与代码的对应关系。
[`verify-runtime-recovery` 技能](../.agents/skills/verify-runtime-recovery/SKILL.md)
说明运行与维护方式。该检查证明有界模型中的性质；真实代码的进程中断测试和
故障变体负责检查模型对应关系，不等于证明整个 Python 程序。

一次验收记录包含适用检查、源码内容身份、环境、实际命令、退出码和产物位置。
结果区分通过、失败、阻塞和不适用。超时或被终止的命令没有通过结论，即使用例
已打印通过摘要。缺依赖不是不适用；只有改动不涉及该边界时才记不适用并说明原因。
日志和 JUnit 作为证据保留，不能用测试总数替代行为标准。
新增 workflow 随分支推送后才在远端生效；仓库的分支保护另须选择对应成功检查，
本地新增配置不表示已经启用远端合并限制。

### V0：静态/文档边界

适用于纯文档、小型 schema 文案或机械路径修正：

```bash
git diff --check
git diff -- docs/developer_guide.md
```

再直接检查：保留链接存在、无待删除文档引用、命令/字段与当前 parser/config 一致。

### V1：Owner-focused behavior

可复用代码修改先运行最窄的现有测试：

```bash
python -m pytest -q <existing-focused-test.py>
```

同时运行请求的可观察边界，例如 CLI help/dry-run、一个 score row、一个结构变换或
一个 posthoc report。一次性脚本不创建测试文件、fixture、framework 或 validation
helper；直接检查目标输出。

### V2：直接受影响的 shared boundary

只有修改跨模块 wiring 时增加一层，例如：

- CLI surface：completion test + Bash/Zsh round-trip。
- config/lifecycle：representative `--print-config --dry-run` + persisted config test。
- evaluator/score/loss：score snapshot/expression + `scores.csv` row contract。
- stage/decision：transaction + stage-entry/active-gate + row history。
- posthoc：对应 smoke，检查 DB/CSV/structure contract。
- backend wrapper：真实最小 backend smoke/canary，而非只 mock。

### V3：Repository/专项 acceptance

核心 evaluator/mutator/decision/loss/`StandardScores` 共享抽象发生改变时，在
具有相应模型依赖的开发环境运行：

```bash
python -m pytest -q
# 等价快捷入口
just test
```

专项命令见 `justfile`，例如 AF3 determinism/profile/inference benchmark、
posthoc collect/filter/refold smoke。只有修改覆盖该边界时才运行；不要把所有昂贵
验收都堆到无关修改上。

如果 full suite 因未改动的明确缺依赖被阻塞，可以报告 blocker，但必须已有 focused
验证覆盖本次行为。不能因为 suite 慢、预算不足或接近结束而把未验证说成通过。

### 按行为选择现有检查

| 修改 | 本开发树中的验证起点 | 要检查的结果 |
| --- | --- | --- |
| 配置和运行身份 | `tests/test_run_provenance.py`、`tests/test_configuration_routing.py` | 实际路由、配置保存、拒绝行为、终态和身份范围 |
| 组件接入 | `tests/test_design_native_registry.py`、`tests/test_native_registry_seams.py` | 选中配置、延迟加载、alias、typed 请求和未知类型错误 |
| score 与 loss | `tests/test_standard_score_snapshot.py`、`tests/test_sequence_losses.py`、对应公式测试 | scope、单位、shape、缺失值、缓存失效及独立数值参考 |
| 搜索与预算 | `tests/test_design_coordinator.py`、`tests/test_default_coordinator.py`、`tests/test_design_branching.py` | 候选、次序、选择、预算、RNG 与失败传播 |
| 持久恢复 | `tests/test_native_refinement_recovery.py`、`tests/test_candidate_recovery.py` | 已完成工作不重复、unknown 不重发、回执与控制状态对齐 |
| 包与公共 CLI | `tests/test_distribution.py`、`tests/test_release_cli.py`、`tests/test_cli_run_dry_configs.py` | 安装产物、仓库外入口、配置解析和错误输出 |
| 验证入口与依赖锁定 | `tests/test_release_validation.py`、`tests/test_install.py` | 实际 pytest 选择与退出码、声明依赖和关键运行时版本、逐项哈希 |
| CPU 结果流转 | `tests/test_cpu_release_example.py` | 独立坐标计算、CSV、SQLite、筛选与报告 |

共享行为优先在已有参数化测试中增加 case。
后端特有输入处理和数值语义使用针对性的测试。
上述文件存在不代表发布 CI 已执行它们。
实际发布 CPU 清单由
[release_cpu.py](../scripts/validation/release_cpu.py)维护。
[release-cpu.yml](../.github/workflows/release-cpu.yml) 与本地
`python scripts/validation/release_cpu.py -q` 使用同一入口。
`--collect-only -q` 可列出实际选择；其他 pytest 参数和退出码原样传递。
`just test-cpu` 调用发布清单；`just test-execution` 使用
`--suite execution`，执行发布清单加 evaluator/Score 执行层回归，
需要 Torch 和 AF3 等科学依赖，但不进行 GPU 或真实模型推理。
`just test` 仍执行完整 pytest。

公式正确性使用独立 oracle。执行方式对照使用同一科学配置下的原生与候选路径。
模型可以构造、结构文件存在、总分相等，均不能单独证明完整数值一致。
静态断言可以检查静态架构合同；输入、输出和恢复行为需要实际调用验证。
实现从 shell 迁到 Python 时，保留解释器选择、参数、环境传递和错误输出的行为检查，
不继续要求旧 shell 语句出现。依赖锁文件按声明的依赖、版本和哈希检查，
不以包的总数或哈希总数作为完整性标准。

<a id="validation-evidence"></a>

### 验证证据与复用

每份验收记录都说明以下内容。各脚本目前有自己的结果格式，
本表是交付要求，不表示仓库已经实现统一结果 schema。

| 内容 | 必须能确定的事实 |
| --- | --- |
| 验证声明 | CLI、公式、执行一致性、恢复或性能中的哪项行为 |
| 源码 | baseline/candidate 的提交与相关内容清单，包括实际使用的未提交代码和 vendor 修改 |
| 科学配置 | 生效的输入、选中组件配置、权重、seed、采样预算、归约及选择规则 |
| 运行边界 | 实际命令、解释器或容器、设备、并发、缓存、服务与资源约束 |
| 比较 | 比较器版本、预先规定的指标和标准、允许忽略的非科学字段 |
| 结果 | 实际调用量、产物、失败与未完成范围 |

`run_config_lifecycle.py` 在 `provenance.json` 中记录两种配置身份：

- `config.sha256`：落盘 `config.yaml` 的原始字节，保持已有读取合同。
- `config.resolved.sha256`：复用 `build_persisted_run_config(..., resolve=True)`，
  在原配置上下文中解析裁剪后保留的字段，以键排序、紧凑分隔符、UTF-8、
  非转义 Unicode 且拒绝非有限数值的 JSON 计算哈希。路径按字符串编码，
  字典键须为字符串，避免 JSON 将不同类型的键编码为同一身份。

解析只访问保留字段及其插值依赖；未启用后端的无关默认值保持未解析。
原参数和落盘 YAML 不被修改。保留字段不可解析、缺少必填值或无法编码时，
新增哈希为 `null`，`reason` 说明失败，其他 provenance 和真实运行终态仍保留。
该字段是 schema 1 的可选增量；旧记录没有它不代表旧运行失败。

解析哈希通常在 run 开始时采集，表示当时的保留视图，包含输出目录等运行路径。
它不保存完整配置内容，不代表后续动态状态，也不作为跨目录科学等价或缓存复用键。
Git dirty 范围仍不含 untracked 文件和 submodule 工作树。
显式输入记录也不覆盖后端随后读取的所有资产。
验收须补齐相关身份，可复用 native registry 的组件源码和资产记录。
不能仅凭相同 HEAD 或相同权重路径复用结论。

每项声明分别记录通过、失败、阻塞、未运行或不确定。
只有完整且有效的成功证据才能复用。
使用 `codex-check` 的环境中，显式声明源码、测试、配置、工具和运行依赖。
依赖清单完整且未变的可重复本地检查可以复用其成功回执。
动态资源、真实模型和部署状态不能由旧本地回执代替。

### 恢复与性能的专项标准

| 持久状态 | 允许的动作 |
| --- | --- |
| 请求尚未发出 | 按原身份提交 |
| 已 issued，缺少完成证据 | 保持 unknown，由原 owner 核对；不能自动重发 |
| 原始预测完整，Score 未完成 | 支持该路径的 owner 核验身份和存活任务后，只恢复 Score |
| Score 完整，控制提交未确认 | 核对 journal 与回执；无法确认时保持 unknown |
| 控制提交完整 | 从提交后的状态继续，不重复选择或扣预算 |
| 输入身份冲突、引用缺失或回执损坏 | 明确拒绝该次恢复 |

故障注入验证受影响的状态转换，同时核对输出、RNG、预算和模型调用量。
某一种 native owner 的可恢复范围不能推广到任意 external child。
缺少原子 collection 提交时，应遵守该 owner 已实现的 unknown 处理。

性能比较通过公共 `astrolabe run` 执行。
对照使用共同的 scoring、mutator、steps、模型预算和输入配置。
区分 cold/warm，记录完整输出完成时间、实际工作量、GPU 活动、
内存和等待原因。同工作量的吞吐变化可写为
`T_baseline / T_candidate - 1`，并给出原始耗时。
受干扰、失败和恢复运行单列，不只保留最快的结果。

确定性配置继续按其精确标准比较。
原生本身不确定时，先定义重复性实验、比较指标和判定方法。
不能在看到差异后放宽容差，也不能用严格模式认证默认模式。
性能证据不足的并发优化保留其明确的实验状态，不据此改变默认值。

### 停止规则

当请求的 observable behavior 和一个直接受影响的 shared boundary 已证明，就停止
验证。只有证据冲突或新增独立风险边界时才增加下一层。真实运行则负责到下一有意义
决策点：早期完整输出不正确就停、修、重跑；输出正确且 durable supervisor/磁盘
handoff 已接管，才可交接。上述停止规则不免除已触发的 V3 检查，
也不免除用户要求的真实端到端验收。

<a id="branch-integration"></a>

## 10. Dirty worktree 与提交保护

本仓库常同时存在 runtime artifacts、vendor/submodule drift 和其他人的修改。开始和
结束都执行只读检查：

```bash
git status --short
git diff -- <owned-paths>
git diff --check
```

规则：

- 不 reset、checkout、clean 或覆盖无关修改。
- 只编辑明确 owned files；发现同一文件并发修改时先整合，不以 HEAD 覆盖工作树。
- stage 时列出具体路径；有 runtime artifacts 时不用 `git add -A`。
- 本地任务管理记录不进入公开源码包；科学运行保留独立的可恢复执行记录。
- logs、outputs、datasets、raw model results、生成 overlay/PSE 不混入 feature commit，
  除非任务明确要求这些 artifact。
- reusable examples 放 `examples/`；task-private overlay 不放 repo root。
- 新 linked worktree 放在主 checkout 外；删除前保留其独有修改和运行记录。

集成使用 `main` 加短期功能分支。PR 声明实际 base；
依赖 PR 按依赖顺序合入，每个单元包含相应代码、配置、测试和文档。
不以分支名认定能力，也不把整个开发现场快照当作单个 backend 的补丁。

收敛历史分支时，先记录原提交或源码 manifest 到集成提交和验证证据的对应关系。
已覆盖的修改避免重复引入。evidence 分支保存历史结果，不作为另一套运行实现。
确认独有改动与活动 owner 后，再清理不再使用的分支或 worktree。

当前几类增量的收敛顺序如下；每一步都在目标 checkout 上验证：

1. 固定各来源的提交或源码清单、独有改动、依赖和活动运行 owner。
2. 先集成 `design/` 数据与控制、组件 registry、配置保存等基础合同，
   连同对应测试和当前实现说明合入。
3. 在基础合同上迁入 evaluator stages、原始预测与 Score 回执等执行边界。
   先证明科学顺序、数值与恢复，再迁入依赖它们的并发和自适应调度。
4. 在集成结果上检查发布入口、配置例子、CPU 检查及受影响后端。
   同步 README 和文档索引，并更新源码版本表。
5. 记录来源到集成结果的对应关系，再处理已无独有代码和活动 owner 的历史分支。

文档随每个代码单元同步；最后的索引检查不能代替前面的接口说明和验收。

提交前逐项确认：

1. diff 只含目标行为所需文件；
2. public surface、completion、docs、config 和 test 按实际影响同步；
3. 新参数有 consumer/units/direction/range/default/owner/acceptance；
4. failure/fallback/output/replay contract 可见；
5. 选择的验证覆盖真实路径，不只是 import；
6. 没有 task-private/runtime/generated artifact；
7. 没有未经请求的依赖、抽象、flag、兼容分支或第二实现。

## 11. 常见扩展模板

### 11.1 新 CLI command/option

1. 在 parser/handler 定义 intent，复用 shared option。
2. 遵守 command design rules，写清 layering/path/default/failure。
3. 让 completion 从 parser 派生；必要 dynamic provider 保持 lazy。
4. 更新 `tests/test_cli_completion.py` 和最窄 command test。
5. 跑真实 help/dry-run 与 Bash/Zsh round-trip。
6. 用户行为写入 user guide；owner 变化才更新本文。

### 11.2 新 config template 或 knob

1. 先标 consumer 和 owner；决定 global、stage、backend 或 env scope。
2. 使用 alias，不把 class name 写进 stage。
3. 声明 units/direction/range/default/null semantics。
4. 在 `--print-config --dry-run` 检查最终值。
5. 若需 replay，扩展 persisted lifecycle；否则证明会被安全 pruning。
6. 用代表性 output 证明 knob 生效，不只 assert 配置里有 key。

<a id="extension-evaluator"></a>

### 11.3 新 evaluator/mutator/predocker

1. 在对应 registry 注册轻量 factory；heavy import 留到实际调用。
2. wrapper 拥有 subprocess/backend parse；component 拥有领域合同。
3. 接入 runtime requirements/env/vendor ownership。
4. 定义 output/score/manifest、failure/fallback 和 deterministic seed contract。
5. 运行 registry/alias/缺资源测试及一个最小真实 smoke。
6. 若改变 CLI/config public choice，同步 completion 和 user guide。

以 evaluator 为例，先检查 `evaluators/registry.py` 和既有 family factory。
本开发树还通过 `design/native_registry.py` 构造 typed operation。
在这些现有入口中接入目标组件，不新增平行的 backend 名称列表。
先验证未选中后端不会被加载，再验证选中配置、输入转换和标准输出。
实际权重的数值验收与加载测试分开记录。

<a id="extension-score"></a>

### 11.4 新 score/loss/gate

1. 先把 stable metric 加入 `StandardScores` snapshot 或 `ScoreContribution`。
2. 明确 scope、aggregation、units、direction、missing policy。
3. 注册 loss/term，验证 expression 与 flattening。
4. 用观测分布校准 transform/weight，记录贡献而不只总分。
5. hard gate 仅用于明确无效态；连续质量留在 objective/soft policy。
6. shared acceptance fields 进入 `decision_context`；transition fields 经
   `transition_consistency_gate_score_context`。
7. 验证普通 decision、stage entry、reference/transition path 与 `scores.csv`。

### 11.5 新 stage/transition behavior

1. 在 `StageUnit/StageManager` 表达 config/state。
2. 在 `stage_transaction` 定义 plan/attempt/decision/commit/rollback。
3. 在 `stage_transition` 定义 gate result、artifact 和 shared score fields。
4. 不在 evaluator/mutator 或 `run.py` 私有 key 中复制 stage semantics。
5. 覆盖 default inheritance、entrance、target re-evaluation、seed/active gate、失败
   rollback、成功 commit 和 row-history。

<a id="posthoc-extension"></a>

### 11.6 新 posthoc behavior

1. 先定义 DB/row/metric identity 和 missing policy。
2. 扩展 collect/filter/refold/report 的 owning module 与 CLI。
3. 保持 incremental/idempotent/atomic write。
4. 更新 completion 与 command help。
5. 跑对应 `just posthoc-...-smoke`，检查 DB/CSV/结构/manifest 内容。

## 12. Definition of done

一个修改只有同时满足以下条件才完成：

- 请求行为通过公共 `astrolabe` CLI、配置、Python API 或输出真正可用；
- 复杂度位于正确 owner，`run.py`/CLI 没有新增可复用领域 special case；
- 参数、score、stage 和 output schema 的公共语义已说明并可审计；
- invalid state 和 backend failure 清楚可见，fallback 不伪装 primary success；
- persisted config 和 posthoc/replay consumer 没被遗漏；
- 最小充分验证覆盖 owner 和直接受影响的共享边界；
- canonical 用户/开发者文档同步，未新增一次性 docs；
- dirty worktree 中无关改动和 runtime artifacts 得到保护。

测试通过只是证据，不是定义本身；只有测试覆盖真实 requirement，输出合同也满足，
修改才算完成。

<a id="reference-practices"></a>

## 13. 参考仓库与适用边界

以下来源于 2026-11-06 核对的官方文档和源码。
GitHub 链接固定到核对的提交。采用的是维护方法，
不要求 Astrolabe 依赖这些框架。

| 仓库及出处 | 用于 Astrolabe 的做法 | 适用边界 |
| --- | --- | --- |
| [scikit-learn 的 estimator 合同](https://github.com/scikit-learn/scikit-learn/blob/6a9d938e58168c55cd2bd70b031273c5169c32d9/doc/developers/develop.rst)和[共同检查](https://github.com/scikit-learn/scikit-learn/blob/6a9d938e58168c55cd2bd70b031273c5169c32d9/sklearn/tests/test_common.py) | 将扩展合同写成共同测试，注册新实现时增加 case | 不把 `fit/predict` 或其构造器规则照搬到科学工作流；Astrolabe 保留配置边界的提前校验 |
| [ASE calculator 指南](https://docs.ase-lib.org/development/calculators.html)和[实现](https://docs.ase-lib.org/_modules/ase/calculators/calculator.html) | 声明支持的输出，输入变化使缓存失效，不支持的属性显式处理 | `StandardScores` 的缺失合同和 loss cache 仍由本仓库实现；不要求所有后端提供同一组 confidence |
| [AiiDA 执行概念](https://github.com/aiidateam/aiida-core/blob/8c24cbc94dd58f2bb36e70402b51f4124fc90edb/docs/source/topics/processes/concepts.rst)和[缓存合同](https://github.com/aiidateam/aiida-core/blob/8c24cbc94dd58f2bb36e70402b51f4124fc90edb/docs/source/topics/provenance/caching.rst) | 区分执行对象与持久记录，明确身份与缓存有效性；raw 预测和派生 Score 分别记录 | 沿用现有 journal/store/receipt；Astrolabe 的源码和权重内容身份、unknown 不重发须独立保证 |
| [vLLM 模型注册](https://github.com/vllm-project/vllm/blob/6c13a380b1b0f071d3522181fb321c5fcd27bfae/docs/contributing/model/registration.md)和[模型测试](https://github.com/vllm-project/vllm/blob/6c13a380b1b0f071d3522181fb321c5fcd27bfae/docs/contributing/model/tests.md) | 注册、延迟加载、公共输入处理和后端特例分开组织；区分加载与数值检查 | vLLM 该指南将部分 correctness 检查列为可选；Astrolabe 的科学等价声明必须有相应数值证据 |

[scikit-learn 的贡献流程](https://github.com/scikit-learn/scikit-learn/blob/6a9d938e58168c55cd2bd70b031273c5169c32d9/doc/developers/contributing.rst)
提供功能分支与 PR 工作流的参考。
[AiiDA 文档入口](https://github.com/aiidateam/aiida-core/blob/8c24cbc94dd58f2bb36e70402b51f4124fc90edb/docs/source/index.rst)
按教程、操作、概念和参考区分阅读目的。
Astrolabe 对应保留贡献流程、用户手册、组件参考与明确标注的架构说明，
避免让一次性实验状态成为公共开发规则。
