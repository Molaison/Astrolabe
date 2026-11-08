# Mol-1 独立保留分支与后续验收合同

**状态（2026-11-08）：原始实现已保留；旧 GPU canary 有证据；当前 main 未完成 Mol-1 验收。** 本文落实已定边界，不新增架构或科学取舍，也不执行模型。

## 1. 分支用途：档案与窄移植起点，不整树合并

- 保留分支：`research/mol1-qualification-20261108`。
- 保留基线：旧 PR9 [`1bdaf3a53227c9197b930dc00a01b752acd90067`](https://github.com/Molaison/Astrolabe/commit/1bdaf3a53227c9197b930dc00a01b752acd90067)。原始 Mol-1 实现及其依赖均留在该历史中；本文提交不修改它们。
- 本合同的 main 参照：[`887fee093f5790da57dca72a07caf229a0c6c76b`](https://github.com/Molaison/Astrolabe/commit/887fee093f5790da57dca72a07caf229a0c6c76b)。这是固定比较对象，不宣称 main 此后不会变化。
- **candidate 身份校验单独合并 main；Mol-1 独立保留。** candidate 修复不在本文提交范围内，也不是 Mol-1 获得资格的证据。

**本分支不是可合并到 main 的整树。** 后续只窄移植科学 adapter，接入目标 main 的 family registry、[`NativeModelClient`](https://github.com/Molaison/Astrolabe/blob/887fee093f5790da57dca72a07caf229a0c6c76b/astrolabe/model_execution.py#L1305) 以及现有唯一资源/恢复 owner；不搬旧 broker、DaemonClient、锁域、调度器或旧恢复框架，不覆盖 main 的安装、启动和打包合同。保留旧依赖是为了追溯，不是批准将它们带入 main。

依据：[运行比较 §3.1](/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/analysis/001_execution_architecture/01_shared_contracts/45_proton_runtime_review/02_runtime_comparison.md#31-molence-mol-1独立后端候选不是-protenix-别名)。该节的 frozen matrix 缺口须与后续独立所有者的 canary 分开理解。

## 2. 已保留的实现与能力边界

以下源码链接全部固定在 P9 基线，而非浮动分支。

| 保留对象 | 可核对的行为与限制 |
| --- | --- |
| [`mol1.py`](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/astrolabe/evaluators/mol1.py) | 独立 Molence Mol-1 adapter，不是 Protenix/OpenDDE 权重别名。当前仅支持蛋白，写原生 job、显式 seed 和已有 MSA 文件；拒绝非蛋白、PTM、templates 及所检查的共价/环化 topology。 |
| [`mol1_resident.py`](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/astrolabe/evaluators/mol1_resident.py) | 独立解释器、source、checkpoint、CCD 等身份检查；单本地设备、native composite resident worker。配置初始化后不允许静默改变。 |
| [`mol1_assets.py`](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/astrolabe/evaluators/mol1_assets.py) 与 [`mol1_daemon.py`](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/astrolabe/evaluators/wrappers/mol1_daemon.py) | 读取本地 checkpoint、CCD、cluster、obsolete 资产；调用 Molence 原生 API，检查每个 seed 的样本和输出完整性。原生 composite 包含数据处理、推理、dump、CUDA cache cleanup，其墙钟不是 GPU kernel 时间。 |
| [`utils.py` factory](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/astrolabe/evaluators/utils.py#L395) | `mol1` / `mol-1` 是同一 factory 的两个 key；旧入口只接受 daemon execution。注册不等于安装或验收。 |
| [`test_mol1_resident.py`](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/tests/test_mol1_resident.py) | fake native API/权重及生命周期合同：输入 guard、分数提取、单次加载复用、配置改变拒绝、缺失输出报错和关闭。不是 pretrained 数值或 GPU 认证；本文未重跑。 |

输出语义也必须保留并在窄移植时核对：whole-complex pLDDT 原始量纲为 0–100，旧 adapter 除以 100 进入标准分数；链均值量纲为 0–1。逐残基置信度来自 `atom_plddt → atom_to_token_idx → token_asym_id`，不能把链均值重复成残基值。旧代码按 `token_asym_id` 首次出现顺序与输入链顺序配对；这是一项待真实 native 输出验证的假设，不能仅凭数组长度相同认证链身份。`atom_plddt` 自身量纲也须按固定版本核验，不能由 whole-complex 的量纲推定。PDE 保留为 PDE，`pae=None`，不得伪造 PAE。

`mol-1-base`、**20 cycles / 200 steps / 5 samples / bf16** 只是旧 adapter 默认值，不是推荐参数、性能最优值或未来验收配置。旧 adapter **不自动继承 uplift 的配体、PTM、template 或 constraint 能力**；uplift 也未以 base 证明 constraint checkpoint。base 不能冒充 constraint 模型。

## 3. 旧证据：可追溯复用，但不能转移为 main 验收

### 3.1 P9 frozen matrix：当时 pretrained/GPU 为 BLOCKED

[P9 frozen matrix](https://github.com/Molaison/Astrolabe/blob/1bdaf3a53227c9197b930dc00a01b752acd90067/evidence/runtime-next/final/backend-certification.json#L681) 中，Mol-1 两个别名对应同一 factory，`pretrained_gpu.measured=false`、状态 `BLOCKED`；配置/环境/权重未冻结，真实数值、失败恢复、吞吐 gate 未完成。不能用两个注册记录计作两个模型验收，也不能改写旧矩阵。

**这仅描述该冻结记录，不表示 Mol-1 此后从未跑过 GPU。** 后续证据如下，源码时期、入口和输入均不同。

### 3.2 Astrolabe 通用 evaluator 流水线：真实短蛋白 canary

来源：[06_live_acceptance.json](/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/analysis/001_execution_architecture/01_shared_contracts/13_generic_evaluator_pipeline/06_live_acceptance.json) 的 `backends[backend=Mol1]`，以及 [01_delivery.md](/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/analysis/001_execution_architecture/01_shared_contracts/13_generic_evaluator_pipeline/01_delivery.md)。

- 固定 `source_epoch`：`09449a9335eb8ad518d16bde16ade9e6767a6ebb40468408f16ea3389ec96da6`，是该交付的源码身份，不是 P9/main Git commit。
- 该交付通过真实 `astrolabe run` 执行；Mol-1 完成 **3 个候选 × 每候选 1 个样本 = 3 个原生结构**，每个结构 20 个残基。一个模型 owner；delivery 报告只加载一次；Score 回执来自两个 PID。
- 记录包含结构 SHA256，delivery 报告真实解析、有限坐标、残基数，以及初始/最终候选序列核对。这是该输入的功能 canary，不是折叠质量、亲和力或通用数值等价证明。
- 该实现由现场基线副本与补丁/manifest 交付，不是任意 main/P9 的可直接应用补丁。其阶段拆分和修复也不能反推已存在于本档案树。

可复用的是原始产物、输入/身份核验入口和对应修复线索；**没有证实持续就绪推理队列的能力上限、GPU 持续满载或生产吞吐提升。** 两个 CPU Score PID 不代表两路 GPU 推理。

### 3.3 uplift：TY3 H800 的 base 复合物 canary

来源：[REPORT.md](/xcf26/zpzeng/custom_scripts/data/outputs/uplift-gpu-20261106/continue/live-features/mol1-acceptance/REPORT.md) 与 [result.json](/xcf26/zpzeng/custom_scripts/data/outputs/uplift-gpu-20261106/continue/live-features/mol1-acceptance/result.json)。

- task：`3386bc35593ae0c3a7df268a87a8acaee27f2e35e0f47ae8dc51748d40dce271`；attempt：`085e55b065d34cd6921155cc93849a78`。
- TY3 `pm-ad990001`、H800 GPU 2；UUID `GPU-c7d20d2e-8621-f86a-71fc-7173f7d6cf64`。入口为 `/mnt/sfs/zengzhengpeng/scientific-tool-runtimes/uplift-suite/bin/mol1`；报告记录严格加载 `/mol_data/checkpoint/mol-1-base.pt`。
- **1 个请求、seed 0、2 个样本**，cycle 20 / step 200；不是两次独立任务。输入为 28+18 残基蛋白、已有合成 MSA、SEP、CCD/SMILES/SDF 配体及命名共价键。报告记录原生退出 0、结构序列/有限坐标等检查通过。
- best 回收 1 个 CIF、all 回收 2 个；补收和同命令 resume 新增 attempt 均为 0。该证据仅覆盖该次回收/恢复，不等于完整故障注入或 main 的 unknown 恢复验收。
- 这是 **uplift 安装版本和入口**的能力，不是本分支蛋白 adapter 的扩展证明。报告明确 base 移除了 constraint embedders，未验收 constraint checkpoint；SDF 参考构象或一次输出距离也不是硬 pose/距离约束证明。

不把这个短输入推成通用显存预算、吞吐结论、三主机矩阵或其他输入类型的 GPU 资格；报告列出的 RNA/DNA、templates 等能力不能冒充本次 GPU 覆盖。

### 3.4 本次只读核对的证据文件身份

以下 SHA256 由本次 `sha256sum` 读取文件得到；用于固定本文引用的记录，**不是本次复跑模型或重新核验原始结构**。本文仅用系统 Bash/Git/jq/sha256sum，不激活或使用 Mol-1/Python/科学运行环境，因此没有新增该类环境使用记录，也没有修改旧证据。

| 本地证据 | SHA256 |
| --- | --- |
| 通用流水线 `06_live_acceptance.json` | `903f6d33de6420feee72552a6050edfa354afd291fbc9d017d2a1d9f7afe8a6e` |
| 通用流水线 `01_delivery.md` | `eb36de8ea42776af675b66dc7726fc56de1a82729d92bd1a9dfc4da4ac50d321` |
| uplift `REPORT.md` | `eb9441793a1065c817a688934f02278c604a0485108169b58408bdc86b7360e9` |
| uplift `result.json` | `d50c4f2b09d2fa481d6e9c81057d29a40671f1d6a06e49e9dc5c98ab9de5f109` |

旧证据只有在相关源码、入口、输入、配置、模型/资产、seed、环境、资源和服务边界匹配时才可用于相应检查；相关变化需补对应证据。以上记录都不能直接授予 `887fee0…` 或未来 main 的 Mol-1 资格。

## 4. 后续验收门槛：当前均未在目标 main 上签收

下表是后续提取的完成条件，不是本次执行计划或已完成声明。数值容差、工作负载、并发方案和通过阈值须由主助手在运行前确定并记录；本文不代定数值，不用结果倒推阈值。

| 门槛 | 必须交付的可核验证据 |
| --- | --- |
| **A. main 接线与唯一 owner** | 固定目标 main SHA 和最小提取 diff；科学 adapter 接 family registry / NativeModelClient，资源准入、lease、取消、会计及恢复仍由 main 的唯一 owner 管理。不引入旧 broker/锁域/第二套恢复；注册、安装和启动 fail-fast 各自有证据。 |
| **B. 固定完整运行身份** | Molence source commit、实际文件 hash/本地补丁；base 或另行合格的 constraint checkpoint 名称与权重内容 hash；CCD CIF/RDKit、cluster、obsolete 等实际资产 hash；解释器绝对路径/版本、依赖与原生库、安装 catalog 条目、容器身份（如使用）、许可证/资产可用条件。冻结 dtype、cycles、steps、samples、MSA、kernel/fusion/TF32、设备 UUID/型号及资源预算。路径存在、模型名称或 registry key 不能代替内容身份。 |
| **C. 输入与科学输出合同** | 固定蛋白序列/MSA/链 ID 与顺序；逐一核对输入→native token→残基→原子→输出结构映射，覆盖重复序列链/重排风险，而非仅核对长度。核验各层 pLDDT 量纲、真实逐残基置信度、PDE/PAE 区分；逐 seed/sample 核对结构、分数、文件和回执身份，缺失或重复必须失败。未支持输入明确拒绝；新增配体/PTM/template/constraint 要独立授权和验收。 |
| **D. 真实 pretrained native 对照** | 在相同冻结输入/资产/seed/sample/科学参数下，比较原生 Molence 入口与 main adapter。运行前登记结构对齐/原子匹配、坐标和分数比较方法及容差、随机性处理、通过阈值。交付原始结构、分数、日志、hash 和逐项对照；fake API、dummy/synthetic 权重或仅能解析 CIF 不替代数值与结构一致性。 |
| **E. 错误、cleanup、会计与恢复** | 验证初始化/推理/输出缺失/评分失败、超时取消及进程退出；观察 worker、VRAM、lease、准备 RAM/输出资源与计数的实际回收，不提前释放仍活跃后代资源。原始预测回执与 Score 回执分离；Score 失败只恢复 Score；已发出但结果 unknown 的请求不得自动重放。交付重启/恢复后无重复预测、无身份串线和无会计泄漏的记录。 |
| **F. 第一层：推理服务能力** | 使用**预先准备的真实输入、持续就绪队列**，先通过 D 的科学一致性，再记录真实推理边界、完成请求/样本分母、测量窗口、加载/预热、吞吐、GPU 活动和资源峰值。按预定方案比较并发，证明所声明的推理服务能力，不用 CPU worker 数推断 GPU 并发。 |
| **G. 第二层：端到端供给** | **F 通过后**，以该已测服务能力为基准运行真实 `astrolabe run`，验证 Beam / Mutate / CPU Score 是否持续供给、背压、代际 barrier、CPU/GPU 重叠、总吞吐和恢复；同时核对候选预算、seed/sample、评分及选择语义。交付命令/配置、身份 manifest、回执、遥测和各 gate 的 PASS/FAIL/BLOCKED。 |

**F → G 的顺序不能倒置。** 包含启动、评分或代际等待的整体活动率，以及仅在推理区间计算的条件活动率，都不能单独证明推理并发已最优。旧 composite wall time、单次 canary 或 CPU Score worker 数也不能替代第一层测量。

## 5. 本次交付的验收范围

本次将原实现及本文保留并推送到独立研究分支，不合并到 main，不改动 Mol-1 实现或重跑模型。**分支交付只表示原实现与后续验收条件已保留，不表示 A–G 已通过。** 后续任一门槛缺少匹配证据时，应保留其未验收状态，不能由 P9 注册、其他所有者的 canary 或 candidate 修复替代。
