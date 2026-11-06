# 完成一次 Astrolabe 修改

本文说明如何定位代码、验证修改并提交可审查的变更。组件接口和职责见
[开发者手册](docs/developer_guide.md)，设计任务的运行方法见
[用户手册](docs/user_guide.md)。

## 确定要修改的源码

先在目标 checkout 中查看版本和已有改动。

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

查看[源码版本边界](docs/developer_guide.md#source-version)。
本地开发树、发布分支和无 Git 的源码副本可能包含不同接口。
有未提交文件时，提交号不能完整标识正在运行的代码。
使用已有任务的源码清单或保存本次拥有文件的差异。

新工作使用基于目标分支的独立功能分支。已有科学运行使用的 checkout 保留原 owner。
在独立 checkout 中集成代码，避免改变正在运行的源码。
分支规则见[提交与集成](docs/developer_guide.md#branch-integration)。

## 准备开发环境并获得第一份输出

优先使用已匹配目标源码的开发环境。新建开发环境时，在该环境中安装开发依赖。

```bash
python -m pip install -e '.[dev]'
python -c "import sys, astrolabe; print(sys.executable); print(astrolabe.__file__)"
```

确认打印的包路径属于要修改的 checkout。`.[dev]` 不安装所有模型运行时。
模型和 PyRosetta 的依赖按[第三方环境](docs/third_party_envs.md)准备。
CPU 例子的报告还需要 `setup.py` 已声明的 `matplotlib` 等基础依赖。
只有模型依赖的环境可能不完整。缺包时在独立开发环境补齐项目依赖，
或使用已有完整环境，不修改正在服务科学任务的环境。

在仓库根目录，用同一个解释器检查公共 CLI 和配置。

```bash
python -m astrolabe --help
python -m astrolabe run --task release_protein --print-config --dry-run
```

预期结果是命令帮助和组合后的配置。`--dry-run` 在模型运行前退出。
`release_protein` 的真实运行需要 AF3 环境和资产。
使用 checkout launcher `./scripts/astrolabe` 时，也要核对其选择的运行环境。

接着运行仓库自带的 CPU 坐标例子。输出目录必须尚不存在。

```bash
demo_root="$(mktemp -d)"
python examples/01_cpu_reproduce.py --out-dir "$demo_root/result"
```

检查输出中的 `scores.csv`、`scores.sqlite`、`selected/` 和 `report/`。
该例子从已有结构计算接触几何，再执行收集、筛选和报告。
它验证 CPU 分析与结果流转，不执行模型推理。
数值参考见[CPU 复现说明](docs/reproducibility.md)。

## 按修改目标定位代码

| 要完成的修改 | 首先阅读 | 验证起点 |
| --- | --- | --- |
| 接入 evaluator、mutator 或 predocker | [组件扩展步骤](docs/developer_guide.md#extension-evaluator)和对应 registry | 注册、延迟加载、配置选择、输出解析，再运行相应真实后端 |
| 新增或修改 loss | [评分扩展步骤](docs/developer_guide.md#extension-score)和 `astrolabe/losses/` | 独立数值参考、缺失值、表达式、缓存失效、保存配置 |
| 修改搜索、预算或恢复 | [统一控制](docs/developer_guide.md#design-control)和 `astrolabe/design/` | proposal/evaluation 次序、预算、RNG、回执与控制提交 |
| 修改模型执行或 Score 并行 | [执行边界](docs/developer_guide.md#execution-boundary) | 阶段寿命、私有状态、科学一致性，再测完整运行耗时 |
| 修改 CLI 或配置 | `astrolabe/cli.py`、`configuration.py`、`argparser.py` | help、dry-run、配置拒绝、保存和重新加载 |
| 修改 posthoc | `astrolabe/posthoc/` 和[扩展步骤](docs/developer_guide.md#posthoc-extension) | 实际 DB、CSV 和结构输出，重复执行的结果 |

类型、配置、registry、实现和测试沿用现有目录。
新增 backend 通常不需要在 coordinator 中添加模型名称判断。
共享协议测试中已有相同输入形状时，增加 case；只有后端独有的行为才单列测试。

## 先定义行为，再实现

在修改说明中写清触发条件、预期输出、失败行为和责任模块。
新增指标还要说明单位、方向、归约方法和缺失值。

修复缺陷时，让回归测试能区分修复前后的行为。
例如保持 evaluator result 不变而修改工作序列，序列 loss 必须重新计算。
对应现有用例是 `tests/test_sequence_losses.py` 中的
`test_loss_engine_recomputes_when_working_sequence_changes`。

新增科学公式时，使用独立计算或已说明来源的参考值。
串行与并行调用同一个公式得到相同结果，只能证明执行一致。
修改公共接口时，同步实际调用者，并处理输出或持久化格式的已有 consumer。

## 选择验证并保存证据

按[验证阶梯](docs/developer_guide.md#validation)选择检查。
先运行相关文件中的具体用例，再检查直接受影响的公共行为。
共享核心修改仍须满足该节的完整测试要求。

```bash
python -m pytest -q tests/test_run_provenance.py
```

上面是修改 provenance 时的实际入口，不是所有改动都必须运行的固定清单。
其他修改使用对应测试。发布 CPU 检查的当前选择以
[release-cpu.yml](.github/workflows/release-cpu.yml)为准。
`just test` 运行完整 pytest，不是轻量检查别名。

后端加载、真实输出正确、恢复正确和性能改善分别记录。
需要真实模型时，先固定共同科学配置、输入、权重、seed 和预算，
再通过公共 `astrolabe run` 入口执行。
每项结论附实际命令、源码身份、环境和产物位置。
缺依赖、未运行及结果不确定分别说明。

## 同步文档并提交

1. 用户可见用法写入用户手册。组件职责与接口写入开发者手册。
2. 在同一变更中更新配置例子、受影响测试和必要的文档链接。
3. 架构草图保留为提案。只有实际接入的接口才写成当前能力。
4. PR 说明写明问题、最终行为、必要验证和实际限制。
5. 明确 PR 的 base 和依赖。每个合入单元都应可以从其 base 重建并验证。

提交前检查本次拥有的文件。

```bash
git diff --check
git diff --stat
```

逐个指定暂存路径。实验输出、模型权重、私有环境路径和其他任务的修改保留在各自位置。
文档变更检查文件目标、标题锚点和命令。科学验收记录保留其对应版本，
不因为文档更新而改写旧结果。

这套流程采用了 scikit-learn 的功能分支和共同接口检查、
ASE 的科学后端合同、AiiDA 的执行记录边界及 vLLM 的模型接入检查。
原始出处和在 Astrolabe 中的适用范围见
[参考实践](docs/developer_guide.md#reference-practices)。
