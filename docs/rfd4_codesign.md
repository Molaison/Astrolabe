# RFdiffusion4 联合生成与界面反馈

## 生成结构与独立预测

`rfd4` mutator 提出一套共同生成的序列和原生蛋白质原子结构。
Astrolabe 仍把候选序列交给配置的 evaluator。
默认的独立预测路线中，生成结构保留 `generated_unrefolded` 来源标记，不覆盖预测的 `StandardScores.complex_paths`。
下面的 `generated_structure` evaluator 是显式选择的另一条路线；两者都不伪造 pLDDT 或 ipTM。

联合提案不能按独立位点概率处理。
RFD4 在推理前获得允许改变的位置，输出后提交完整生成序列。
Astrolabe 不截断生成后的突变，也不把部分新序列拼到旧结构上。
默认等长路线中，`mutation_num` 限制释放的独立序列位置数，并不保证这些位置全部发生替换。
显式变长重构使用整条序列替换，由 `feedback.reconstruction` 的长度范围控制。
相同序列的不同生成构象可以分别保存。

## 配置与使用

[`config/rfd4_codesign.yaml`](../config/rfd4_codesign.yaml) 是共享、可组合的最小运行配置。
配置运行一次初始 AF3 评价和最多两轮联合提案，使用单 seed、单 sample、单 recycle 的独立 AF3。
这些缩减设置用于集成和有界参数比较，不是最终设计质量评估设置。
示例允许释放 `100%` 的可变 binder 位点，但仍排除反馈中应保留的有利相互作用、固定位置和 selector 未选位置。
优化指标是形状互补 `interface_sc` 和跨界面氢键数 `interface_hbonds`，不是极性原子满足比例。
排序标量为 `interface_sc + interface_hbonds`，同时要求
`delta_interface_sc >= 0 and delta_interface_hbonds >= 0`，且标量严格提高才接受。
因此一个指标的增加不能补偿另一个指标的下降；这不是经过权重拟合的亲和力模型。
带符号的 Rosetta 短程残基对能量单独记录，并用于位点排序，不直接混入目标。
具体阈值和预算都是显式协议参数，不是生物学最优值。

主 Astrolabe 环境需要已有的 PyRosetta 和配置的 evaluator 依赖。
RFD4 使用自己的独立 Python、源码和 Hugging Face 模型目录：

```bash
export ASTROLABE_PY=/path/to/astrolabe/python
export ASTROLABE_RFD4_PYTHON=/path/to/rfd4/python
export ASTROLABE_RFD4_SOURCE=/path/to/RFD4-Proteina
export ASTROLABE_RFD4_MODEL=/path/to/model-snapshot
export ASTROLABE_RFD4_CACHE=/path/to/persistent/rfd4-cache
export GH_WEIGHT_AF3=/path/to/alphafold3/models

./scripts/astrolabe run --task rfd4_codesign --overlay case.yaml --dry-run
./scripts/astrolabe run --task rfd4_codesign --overlay case.yaml
```

原生 Torch/Triton JIT 还需要可用编译器。从精简容器启动时，应让编译器及其 sysroot 在容器内可见，
并按需要显式设置 `CC`、`CXX`；宿主机上存在 `gcc` 不代表容器内能调用它。

`case.yaml` 只放当前任务的输入和输出身份；固定 target 序列须有 `+` 前缀：

```yaml
task_name: my_complex
out_dir: /path/to/results
seed: 17
seqs:
  A: BINDER_SEQUENCE
  B: +TARGET_SEQUENCE
```

物理链 A 是 binder、B 是 target。非 AB 体系须同时调整共享配置中的 assembly、链分组和 stage。
修改共享配置中的 `steps`、`stages.default.mutation_num` 和 generation 参数可扩大实验，
不要把共同方法设置复制进各个输入 overlay。
关闭 `mutator.mutator_terms.rfd4.feedback.enabled` 可单独使用第一阶段的联合生成，
仍然遵守固定序列、释放位点与完整结构缓存约束。
若要按接口选择位置，可设置现有 `position_selector` 表达式。

### 可选：直接评价生成坐标，不做预测或 repack/minimize

在共同的方法 overlay 中显式选择 evaluator，并提供 step0 原生生成复合物：

```yaml
evaluator:
  enabled_aliases: [generated]
  evaluator_terms:
    generated:
      _type: generated_structure
      initial_structures:
        AB: /absolute/path/to/native_generated_complex.cif
stages:
  default:
    evaluator: generated
```

输入路径可由各个 case overlay 提供。使用同一组 overlay 执行
`./scripts/astrolabe run --task rfd4_codesign --overlay generated.yaml --overlay case.yaml --dry-run`
及去掉 `--dry-run` 的实际运行；不修改默认共享实验配置。
`initial_structures` 是 assembly→path 映射，物理链序列必须与 step0 输入完全一致。
后续每步只读取该次 mutation 返回并已发布的 `rfd4.generated.cif`，校验步骤、assembly、请求/结构摘要及实际序列。
不会扫描文件、按序列合并构象，或在提案缺失、过期（包括零释放位点的空提案）时回退到初始/已接受结构。
同序列的不同构象仍分别评分；发布文件不依赖可清理的 backend cache。

返回值为 `StandardScores(names=..., plddt={}, complex_paths=...)`。
现有 SC、HB loss 和逐残基评分直接使用这些坐标；此路线绕过运行器的 post-eval repack/minimize 和旧 DEBUG 初评 pickle，
不执行独立预测，不补造置信度。其目标是生成坐标上的优化，不等同于独立 refold 验证。
本次仅支持普通 `run`，配置 `recovery` 会在恢复构建前明确拒绝；不提供 portable/recovery 扩展。

### 固定小分子靶标

普通 CLI 的联合生成和原始坐标评价支持标准蛋白质 binder 加单个非共价 CCD 配体靶标：

```yaml
types:
  A: p
  B: c
seqs:
  A: BINDER_SEQUENCE
  B: +CFF
extra_residue_params:
  - /absolute/path/to/verified/CFF.params
```

每条配体链恰好一个 CCD 残基。配体保持 CCD 身份、全部重原子、电荷和键拓扑，
全部配体坐标均作为原生坐标条件，不参与蛋白质突变预算。
坐标条件不是输出坐标的精确钳制：输出另经 `max_fixed_atom_rmsd` 工程容差检查，
仍须独立核对配体几何。原子引用使用物理链和残基序号，如 `B:1:O11`。
缓存区分链类型和 CCD 身份。显式 `.params` 原样复制到运行目录，
供评分和动态反馈共同使用，不覆盖冲突参数，也不自动重新参数化。
须验证实际供受体定义，不能假定 Rosetta 内置残基类型必然正确。
`exclude_AA=[]` 时不添加序列概率条件；均匀 20-AA 概率是显式软条件，
不能代替“不限制突变”。输出仍须是规范蛋白质氨基酸。

`rfd4_native_ligand_reward` loss 调用原生 RFD4 interaction metric，
计算 `min(hbond_atom_pair_count_coi, 6) + 0.5 * hydrophobic_atom_pair_count_coi`，
需要 `python_bin` 和 `source_dir`。它不加载生成模型，也不做预测。
这只是原生搜索的代理奖励，不是 Rosetta 氢键数：当前原生 AtomWorks 的配体
供受体启发式可能把不应接受氢键的氮计入，因此须用准确化学定义独立审计。

此配体扩展不包含 SMILES 输入、共价配体、可变配体或类型化 durable `Proposal` 配体支持。

## 读取生成结构

普通 CLI 在输出目录的生成结构子目录中保留每次提案的
`*.rfd4.generated.json` 索引、`*.generated.cif`、`*.annotated.cif` 和 `*.backend.json`。
索引包含请求与结构摘要；backend 清单记录实际条件、输入映射及模型身份。
无论该提案后来接受还是拒绝，这些文件都不会随 raw model 临时结果清理。

类型化 `Proposal` 可以直接从内容存储读取或导出，不依赖原模型缓存目录：

```python
from astrolabe.design.generated_structure import (
    read_generated_structures,
    materialize_generated_structure,
)

records = read_generated_structures(proposal, data_store)
for record in records:
    assert record.validation == "generated_unrefolded"
    print(record.assembly, record.sequences, record.structure_digest)
paths = materialize_generated_structure(proposal, data_store, "exported_proposal")
```

通过 registry 构建的 RFD4 operation 自动在外层 Runtime 查询缓存前解析生成依赖。
直接用 `make_mutator_operation` 包装 RFD4 时必须传入 `prepare_native=True`。

## 动态反馈的含义

每轮反馈来自当前已接受的 evaluator 结构。
被拒绝提案的结构仍可检查，但不成为下一轮反馈来源。
冻结集合每轮重新计算，不永久累计。

| 观测 | 定义 |
| --- | --- |
| 界面氢键 | Rosetta `HBondSet` 识别的跨组氢键。记录供体重原子、氢、受体和能量。默认 raw energy < 0 才计入，每条 donor-H/acceptor 观测只计一次，不将逐残基计数的两端重复相加。重原子距离邻域只用于定义界面，不代替氢键识别。 |
| 短程界面残基对能量 | Rosetta `EnergyGraph` 跨组残基对的带符号加权能量。包含不利正项。排除长程及不可分解项，不是结合自由能或突变结合 ddG。 |
| 未满足界面极性原子 | 在指定界面邻域内，按当前 Rosetta 残基的实际供氢/受氢位点可形成氢键，但未参与符合阈值氢键的重原子。供体是 `Hpos_polar()` 中极性氢的 `atom_base()`，受体来自 `accpt_pos()`；埋藏状态由单独的 SASA 条件决定。 |
| 极性原子满足比例 | 参与至少一个符合阈值氢键的界面极性重原子数，除以该集合总数。链内氢键也能满足原子。它不表示所有供氢或受氢价态都被完全占据。 |
| 形状互补 | Rosetta 的结构级形状互补标量。没有把它伪装为逐位点指标。 |

反馈标量优先读取 `StandardScores.complex_paths` 中的实际文件，与下一轮 mutator 使用同一原子标准化入口；仅有 pose 的调用仍可使用。
独立后处理必须匹配运行时的 PyRosetta 初始化：当前 Astrolabe CLI 使用
`-ignore_zero_occupancy false -mute all -beta`（非标准残基还可能带额外参数），反馈评分函数为 `ref2015`。
仅使用同名 scorefunction 而遗漏初始化选项，仍可能改变氢键判定。比较时需记录初始化参数及 PyRosetta 版本。

选择策略保留有利氢键参与位置，并可保留短程界面能量最有利的位置。
其余可设计位置与 `position_selector`、固定序列掩码和位点预算相交。
空选择不自动扩大到所有可变位置。
固定残基使用序列条件 `C_SEQ` 和模型可表示的重原子坐标条件 `C_CRD`。

未满足极性原子的氢键条件默认只写到身份稳定的 target 原子。
`C_HB=ACCEPTOR` 表示这个原子本身是受体，不是要求对方成为受体。
供受体身份与 `HBondSet` 使用相同的残基级化学列表，不仅看通用原子类型标记。
因此内部 Pro 主链 N 不会被选为供体，而带可供氢的 N 端 Pro 仍可作为供体；
对于 N 端 Pro 等使用虚拟影子原子的残基，供氢的虚拟基原子按
`ResidueType.atom_being_shadowed()` 映射回真实重原子（例如 `NV→N`），
极性列表、氢键供体身份及满足度计数使用相同映射，不改变原生氢键能量。
His 互变异构、质子化及末端/二硫键变体按实际 Rosetta 残基类型判定，不按残基名称硬编码。
这是当前结构在指定 Rosetta 化学表示下的定义，不是独立的 pH 或质子化态预测。
同时具有供受体身份的原子使用 `BOTH`，在原生 RFD4 中同时开启两个角色通道，不表示二选一。
原子满足度仍沿用“任意一个符合阈值的氢键”，不会要求两个角色都满足，也不表示价态完全饱和。
候选条件按复合物原子 SASA 升序取至多 `max_hbond_conditions` 个；主链和侧链均可入选。
修正化学判定会改变反馈记录和请求摘要，旧生成缓存不能冒充修正条件下的新采样；
旧结构、旧请求及已有验收证据应保留，影响须在相同坐标上另行审计。
`BOTH` 只用于兼有供体和受体化学角色的原子。
这些条件是模型的采样输入，不保证输出一定形成指定氢键。

## Partial diffusion 参数

RFD4 的 `partial_t_backbone` 和 `partial_t_latent` 分别控制骨架与序列表示的起始时间。
时间越接近 1，保留输入越多；越接近 0，扰动越强。
释放位置范围与这些时间参数是两个独立控制量：放开更多位置不等于降低 partial 时间。
`partial_trajectory_fraction_steps` 控制使用完整采样轨迹末尾的步数比例。
它不是扩散时间，不能和前两个时间参数混用。
当前原生 recycle preset 的分段总长是 60，因此接口要求 `nsteps: 60`，
不擅自缩放原生 schedule。Partial 设置可以缩短实际采用的采样轨迹。

本接口把序列固定和坐标固定分别保存。
输出还会核对序列条件，并报告固定、模型可表示的重原子坐标偏差。
默认固定原子 RMSD 界限为 0.5 Å，这是可配置的工程验收参数，不是生物学最优阈值，也不是零位移承诺。

示例默认使用 `partial_t_backbone: 0.7`、`partial_t_latent: 0.7`。
这是单一复合物有界比较后的工作参数，不是通用最优值：

- 同一 seed17、相同释放预算下，t=0.7 的独立 AF3 SC/氢键轨迹为
  `0.338820/0 → 0.406653/1 → 0.461577/1`，两轮接受。
- seed29 的初始 AF3 构象不同，SC/氢键为 `0.408889/0`；
  第一轮 `0.457696/1` 被接受，第二轮 `0.357730/0` 被拒绝。
  不能把这个 seed 的改善都相对 seed17 的初始值计算。
- t=0.8、完整释放预算在 seed17 得到 `0.411263/2`，相较 t=0.7 氢键更多、SC较低；
  两者互不支配，默认选择0.7是为了更充分的局部重采样，不宣称它在每个指标上优于0.8。
  t=0.8 的 seed29 输出违反禁C条件并终止，不能视作成功复现。
- 更强的 t=0.5/0.3 在 seed17 的四个生成结构均改善SC和氢键，
  但各自独立 refold 为 `0/0`，全部拒绝。因此不能根据生成结构的高分继续加大噪声。

生成与 refold 的改善不保证同时出现：t=0.7、seed29 第一轮生成SC下降，但独立refold双指标提高。
上述两条t=0.7轨迹的最终接受结构AF3 ipTM约0.10/0.16、归一化平均pLDDT约0.43/0.45，
仍然低置信度；只证明当前指标上的小样本改善，不证明可靠折叠、结合能力或普遍成功率。

## 缓存和恢复

生成缓存绑定实际输入结构内容、链映射、位点选择、原子条件、seed、采样设置、软件身份和模型权重。
同一路径内容变化会产生不同请求身份。
不同输出路径本身不应改变科学请求。
源码、适配器、配置和模型文件使用内容摘要；安装依赖记录还结合文件的设备、inode、大小和纳秒级修改/变更时间。
缓存也区分所记录的 GPU 映射与运行设置，不承诺跨软件或硬件身份复用。

缓存命中使用经过校验的结果，不启动模型。
未命中时，后端在独立的 Python 环境中加载原生 RFD4，复用当前任务拥有的驻留模型。
Astrolabe 的 Python 环境不需要安装 RFD4 的依赖。
RFD4 的 Triton/Inductor 编译缓存另存于该缓存目录下的私有子目录，
不继承 AF3 或容器留下的空编译缓存路径，也不把编译缓存当作已完成的生成结果。

每个内容键只有一个写入者。
进程退出会释放文件锁。
临时目录和失败结果不被视为已完成条目。
通过完整性与语义检查后，结果才原子发布。
损坏的缓存明确报错，不悄悄把不完整结构交给评估器。

普通运行保存独立的生成结构索引和结构副本。
类型化运行把结构与来源写入 `Proposal.artifacts`，由现有内容存储保存其引用闭包。
这两种访问方式都不要求把生成结构混进预测证据。
生成缓存的可恢复性不代表任意旧 Astrolabe 搜索流程都支持进程重启。搜索恢复仍遵守所选运行模式的既有契约。

## 支持边界

当前实现针对长度不变的标准蛋白质复合物；普通 CLI 另支持上述固定单 CCD 配体靶标。
链位置使用物理链内的残基顺序，并保存与原始编号的映射。
动态 Rosetta 反馈目前要求单字符物理链标识。
AF3 常规输出含链末端 OXT，但当前原生 RFD4 不表示该原子。
接口只在模型输入投影中省略已确认的链末端 OXT，完整保留原始输入，并记录被省略原子的身份、坐标、原因及原始/模型输入原子数。
输出不会补造 OXT；坐标固定与 RMSD 门槛仅针对模型可表示的重原子。
这些 OXT 仍参与独立 evaluator 结构上的真实化学观测，但不会成为 RFD4 氢键条件，也不会被列为可由模型固定的氢键。
其他缺失、非标准或矛盾的原子身份不按此例外处理。
不能忠实表达的可变共享序列副本、非标准化学、矛盾的序列结构来源或概率操作会明确拒绝。
不会无声换用 RFdiffusion1、RFdiffusion3 或一个不同的优化模型。
原生条件不保证每个样本满足序列排除要求；输出校验会拒绝违例。
当前普通CLI可能因此终止该轨迹，不自动换seed、修补序列或重试成一个“成功”提案。

动态条件的作用是改变提案分布。
是否改善结构合理性、形状互补或结合，需要在独立评价和适当规模的比较中观察。
一次运行通过只证明集成链路可执行，不能证明亲和力提升。

## 科学来源

RFD4 条件语义以 [annotations reference](https://rfd4-proteina.org/latest/reference/annotations.html) 为准。
时间语义见 [partial diffusion and inverse folding](https://rfd4-proteina.org/latest/how-to/partial-diffusion.html)。
[PPIFlow](https://github.com/Mingchenchen/PPIFlow) 的界面优化和 partial flow 为反馈流程提供参考，但本实现不是 PPIFlow 的复制或等价实现。

## Opt-in dynamic local repair

`feedback.retention.local_repair: true` restricts proposals to measured binder
heavy-atom neighborhoods of the target. The local policy defaults are
`repair_radius: 6.0` Å, `hbond_retention_cutoff: -0.5` REU,
`energy_retention_cutoff: -1.0` REU, `energy_top_k: 2`, and
`max_hbond_conditions: 1`; explicit existing values override these defaults.
Set `stages.default.mutation_num` to a bounded residue budget (for example 12).
The ordinary nonlocal policy and shared default configuration are unchanged.

Each accepted parent is diagnosed again. Source-present favorable crossgroup
HBs and top favorable short-range pair interactions protect whole binder
residues. A reachable unsatisfied target polar atom is chosen by burial, SASA,
and stable atom identity. Release candidates lie within the repair radius of
that atom, ranked by proximity, unsatisfied polar count, and worse pair energy.
Without a reachable polar goal, interface residues are ranked by the latter two
proxies. The complement stays sequence/coordinate conditioned. These are not
local shape-complementarity measurements, and the signed pair energy is not a
binding free energy. Identical parents may legitimately produce identical plans.

The target atom's own donor/acceptor role is passed to the native one-body
`C_HB` condition, at most one goal in local mode even if a larger cap is supplied.
This does not specify a partner atom or enforce HB distance/angle. Fixed
coordinates are conditioning, not exact output clamps. Diagnostics in
`ConditionPlan.feedback_json` include measured geometry, protection reasons,
local/goal-local candidate sets, and the selected goal.

For a controlled static-condition ablation, `feedback.fixed_plan_path` can
reference a JSON record with `source_sha256` and a serialized `plan` from the
initial request. The complete fixed/released partition, current eligible budget,
AA exclusions and target-only HB conditions are checked. Condition identities
then remain static, but coordinates still come from the current accepted
parent; the initial-plan hash and mode are retained in request/cache provenance.
A chemically appropriate initial plan must be generated from the same target.
This option does not enable durable generated-evaluator recovery.

For proposal-aware feedback, additionally set
`feedback.repair_unsatisfied_attempts: true` (requires enabled local repair and
no fixed-plan override). Each returned generated structure is chemically
analyzed for its actual requested target atoms. An unsatisfied attempted target
increments a per-trajectory failure count even if the proposal is subsequently
rejected. The next parent-based plan prioritizes reachable unsatisfied atoms
with fewer failed attempts, before burial/SASA/identity. It never substitutes a
rejected structure for the accepted parent, or rotates targets by step number.
Counts and source-SHA-linked outcomes enter each subsequent serialized plan and
cache identity. Duplicate delivery of the same request is counted once. This
controller is process-local to the ordinary CLI trajectory; durable recovery
remains unsupported. Cached generated results are still diagnosed normally.

### Local inpainting and contiguous repair segments

`generation.mode: inpaint` selects native **full-noise generation**, conditioned
on the retained structure, instead of corrupting the accepted parent as a
partial-diffusion initial state. Remove inherited `partial_t_backbone`,
`partial_t_latent`, and `partial_trajectory_fraction_steps` keys: supplying any
of them, even as null, with inpaint mode is an error. Missing mode or explicit
`mode: partial` retains the historical partial defaults. Inpaint mode is part
of the generation fingerprint and cache identity; ordinary full-generation
steps/recycles still count toward the actual model work budget.

With local repair enabled, opt in to `feedback.retention.repair_segment_length`
to release one **complete, contiguous, fixed-length, indexed** binder segment.
Its length must be at least two and no greater than the mutation budget.
Windows cannot cross chains, excluded positions, or protected good-interaction
residues. At least one member must be within `repair_radius` of the chosen goal
(or target interface if no goal is feasible); connecting flanks may lie outside
that radius. Among feasible windows the planner uses closest goal distance,
summed unsatisfied polar counts, descending summed pair energy, then stable
identity. No complete feasible window means no release, not a sparse fallback.
The exact window, protected positions, goal and outside-radius flanks are
serialized in the condition plan and therefore in the request/cache identity.

Both partial and inpaint modes can use this same segment policy and
proposal-aware failure memory. The released segment has neither sequence nor
coordinate conditions, while the complement retains both. This first interface
does not insert/delete residues, introduce backbone-only transition shells, or
hard-clamp output coordinates. Retained-coordinate drift, chain connections,
ligand geometry and actual chemically qualified hydrogen bonds must be measured
on raw outputs; an HB role condition is not proof of a formed interaction.

### 变长重构与 unindexed 功能 motif

显式设置 `feedback.reconstruction` 后，重新生成整个 binder 骨架，仅保留已有强氢键所需的
功能几何。首版支持一个完全可变、无 PTM／locked positions／homology、单 segment、
单物理副本的蛋白与一个固定单 CCD 配体；不支持混合 mutator、pre-update 或 CLI recovery。
普通等长路线不受此选项影响。

```yaml
generation:
  mode: inpaint
  nsteps: 60
feedback:
  enabled: true
  reconstruction:
    length_min: 90
    length_max: 110
  retention:
    freeze_hbonds: true
    energy_top_k: 0
    max_hbond_conditions: 1
    hbond_retention_cutoff: -0.5
  repair_unsatisfied_attempts: true
```

此段替换对应设置；移除继承的 partial 参数和 `repair_segment_length`，并使用
`position_selector: null`。长度上下限包含 motif 残基，均为闭区间。此模式的预算是完整生成
的轮数／diffusion 步数，`mutation_num` 不再限制重构位点数。若只统计模型直接生成结构，
选择 `generated_structure` evaluator，继续保留 `generated_unrefolded` 标记。

每轮从已接受结构重新测量跨界面氢键。保留能量小于设置阈值、D–A ≤3.5 Å、
D–H–A ≥120°且重原子在源结构中存在的相互作用。例如 Asn 仅条件化 CG／OD1／ND2；
不固定其主链序列位置。整颗 motif 残基保留身份（C_SEQ），指定原子保留软坐标条件
（C_CRD），C_IDX=False，并使用独立 C_IDXGRP，释放 motif 间的原始序列间距。
没有合格氢键时不增加 packing 锚点，该轮是配体条件下从头生成骨架。
最多向一个未满足的配体极性原子指定其自身供受体角色。

生成结果以完整 `pending_sequence` 暂存；接受时提交并刷新长度、切片和 mask，拒绝时丢弃。
System 的长度许可显式参与身份；默认仍拒绝变长。逻辑 occurrence 的实际范围通过
`resolve_legacy_index_map(occurrence, candidate, data)` 按候选计算，不能把原始长度写死。
使用 typed operation 时，须在建立 System 前调用 mutator 的 `declare_variable_topology()`，
并关闭仅表示点突变的 mutation-context codec。

`rfd4.backend.json` 的 `validation.motif_mapping` 保存逻辑 motif ID、
源残基位置、输出残基位置及原子名；`sampled_lengths` 保存本轮实际长度。
映射由原生 SA_UII／SA_CRDSRC／SA_IDXGRPSRC 联合验证，缺失、重复或身份不符即失败。
下一轮主动把原生已重新编号的 motif 再设为 unindexed。其余新骨架不具有逐位点 ancestry，
不能凭相同残基编号声称氢键保留或新增。

长度范围、motif 原子、谱系、氢键目标及反馈记录均进入请求缓存身份。命中仍重新验证原生
映射及输出序列。若设置氨基酸排除且包含 motif，排除在生成后的非 motif 位置验收；
不覆盖受保护 motif 的原始身份。软坐标条件、完整插入映射或更高目标分数均不能代替
肽键、配体化学几何和整体折叠检查。
