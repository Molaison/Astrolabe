# Evaluator 生命周期、执行边界与认证矩阵

本文件记录实际代码实现和已有证据；**没有把 CPU 替身模型、JSONL 通信测试或空闲 GPU 盘点写成真实模型/GPU 认证，也没有宣称尚未测量的加速。**

## 1. 实施基线与范围

输入方案和审查证据固定在 `b537b8a81ff0738922e80a997ffeb9700cdafe09`。此次改造继承并保留 f102 后续实现至 `834402daade9f62e9dec0b08041dade163a89369`；不是从旧方案提交覆盖重做。完整基线说明、保留提交与快照位于 `docs/runtime_implementation_baseline.md` 和 `evidence/runtime-20261103/`。

实际入口是 `astrolabe/evaluators/utils.py` 的 registry/factory → evaluator `prepare_input/run_function/prepare_output` → 下列 resident/wrapper。当前共有 **13 个 evaluator family、14 个 registry key**，其中 `mol1` 与 `mol-1` 是同一 factory 的别名。与原方案相比，Mol-1 已由保留的后续提交接入，因此本轮继续改造这一真实 adapter；**Boltz2 未在 registry 或 loader 中出现，不能用 Boltz1 自定义 checkpoint 冒充 Boltz2**。

| Registry / 精确模型范围 | 当前实际代码和固定参考 | 本轮状态及边界 |
|---|---|---|
| `alphafold3` | vendor `4b154d922759c918dd4b5a4d5a779853100fd20e`；实际 `model_dir` 的完整文件摘要；native model config | pilot：JAX 参数加载、一次 device transfer、CPU prepare / 连续 forward / CPU finalize；没有变更样本/recycle/seed/precision |
| `alphafold2` | vendor `c77e5d2a8961d1a353632c462914ff0a32a950f6`；`params_model_{model_id}_{suffix}.npz`，默认 `model_3_multimer_v3`，`ptm` 等由实际配置区分 | owner `prepare_only`；worker 固定参数和显式 seed。共享模式保留既有 audited source 与 `RunModel.predict` 摘要检查 |
| `rf3` | 外部 Foundry `repo_path` 的 `src/` 与 `models/rf3/src/` 实际摘要、`ckpt_path` 摘要、实际 Python 环境 | 新增 common worker；exact effective config（含 seed）分别缓存；原生 post-init RNG 每请求回放；保留 legacy 直连/旧桥接作对照 |
| `esmfold` | ESMFold1；`esmfold_3B_v1.pt` 或显式本地 `.pt`；实际 esm/esm2/openfold 源码与 torch hub checkpoint 文件摘要；仓库 ESM pin `43c67d952da30d2471f609848931d20ec8c98532`、OpenFold pin `8396098268a5675b0ff776ec274f15d380a4ccf5` | 模型构造时 RNG round-trip；同轨迹复用；native infer 与 output_to_pdb 的连续设备相关段保留 |
| `esmfold2` | `biohub/ESMFold2`；ESM pin `ba4d7124864eed323a93bf3cfefcd958f573b75a`，Transformers pin `ef32577f55da19a4989cd7b22e004dc43a4998cb`；两个显式本地 snapshot | folding checkpoint 与嵌套 ESMC checkpoint 分别计数；显式 seed、local_files_only；CPU job/文件I/O与 native builder.fold + 完整结果materialization分层；后者保留复合段 |
| `protenix` v0.5 default | `protenix_base_default_v0.5.0` + 实际 checkpoint/source/env | 独立 model identity；CPU feature/dumper、worker predict；版本是否可运行取决于实际 source/checkpoint 配对 |
| `protenix` v0.5 constraint | `protenix_base_constraint_v0.5.0` + 专用 checkpoint | 单独认证；保留 strict/compat 的 requested/effective model 规则，不把切换后的结果记为 v2 |
| `protenix` v1 | `protenix_base_default_v1.0.0` + 实际 source/checkpoint/env | 工厂身份保留测试有覆盖；真实 v1 运行尚无证据 |
| `protenix` v2 | `protenix-v2` + 实际 source/checkpoint/env；仓库 reference pin `2475421477ab414b571149ad4a875c390ff8a35d` | 同一通用 adapter 不等于所有源版本认证；保留 dtype、动态 skip_amp、原始 mc_dropout/RNG 消耗 |
| `opendde` | vendor `266ce4c49d492ad1077866000d83704999985f46`；实际 model_name、checkpoint、CCD、可选 template 资产 | 本地公共 Python API 常驻；single-device/single FoldCP；CPU preprocess 独立，原生 infer loop 的 GPU/RNG/cleanup 全部在同一 nested lease |
| `mol1` / `mol-1` | Molence Mol-1 的实际 `mol/` source 摘要；默认 `mol-1-base`；checkpoint、CCD、cluster/obsolete 数据摘要 | 独立 adapter，不以 Protenix 兼容性代证；本次代码支持原 factory 的 protein 输入合同；未新增或认证 constraint variant |
| `boltz` | **Boltz1** `Boltz1.load_from_checkpoint` / `boltz1_conf.ckpt`；vendor `cb04aeccdd480fd4db707f0bbafde538397fa2ac` | lazy loader 实计数；CPU YAML/MSA prepare；Lightning predict/dataloader/transfer/writer 保留复合段；不是 Boltz2 |
| `chai1` | vendor `c544fb183e865c4950909444db860a9d50604f66` 及已有受审 local variant；六个 component checkpoint、可选 ESM2 JIT checkpoint、conformers | CPU feature context、GPU ESM、完整 folding 连续子图分开准入；组件默认 CPU offload，显式 `component_residency=device` 可保留 GPU；两者均待真实 GPU 认证 |
| `rhofold` | vendor `6bdfbda720184409eb682ce08c05d258162ddc48`；RhoFold checkpoint 与本地 RNA-FM hub 依赖 | 新增 common worker；CPU tokens、RhoFold/RNA-FM + 原位置 sigmoid/softmax、CPU NPZ/PDB；可选 OpenMM 独立请求并验证实际物理绑定 |
| `dummy` | 仓库内 deterministic adapter；可选 PyRosetta 写结构 | 无推理权重、无 GPU；纳入 owner CPU 时间与关闭协议；写结构时 PyRosetta 是进程级环境 |
| `rosetta_crystal` | PyRosetta 进程环境；实际 residue `.params`、参考结构和计算配置 | CPU backend；参数变更要求新进程；参考 pose/参考评分缓存按真实文件和计算参数失效；不修改 scoring/replicates/selector |

这里的 Git pin 是仓库参考，不替代实际运行 manifest 的文件 SHA-256。不同安装 checkout、权重文件或 Python 环境必须产生不同身份。调用者必须冻结源代码和模型资产直到本次 session 结束；不能在进行中的 worker 内热替换模块或权重文件。

环境身份由实际 worker 启动环境产生：owner经broker准入的metadata probe，使用同一环境overrides、设备mask与线程限制；worker独立重算。身份包含Python binary/prefix/version、框架和Triton/NVIDIA包版本、Torch CUDA编译版本/cuDNN运行时版本/build config摘要，以及JAX/XLA、CUDA/CUBLAS/cuDNN、Torch allocator/precision、OpenMM、线程和动态链接相关有效环境。GPU-hidden owner的 `JAX_PLATFORMS=cpu` 不直接作为GPU worker身份。`test_worker_environment_identity.py`验证不同有效初始化环境不能复用同一身份，并验证外部wrapper的已知unset行为。

`execution_code`还包含公共codec、artifact预算、RNG/lifecycle/session helper，以及worker实际使用的runtime server/contracts/admission/client/broker_client/telemetry源码摘要。不能只因checkpoint未改变，就让修改过公共执行代码的owner复用旧worker。

## 2. 生命周期 owner 与状态含义

### 公共状态

`lifecycle.py` 检查已经存在的 PyTorch tensor storage、module parameters/buffers 或 JAX array；查询状态不会为了获取状态再导入或初始化 GPU 框架。`process_alive`、`weights_loaded`、`gpu_resident`、`ready` 分开记录：

- 有 hello 只证明进程存活且接收协议。Boltz 和 Chai 初始 `load_count=0`，权重可在首请求合法准入后加载。
- `load_count` 在具体成功 loader 边界递增；失败 loader 不记成成功。Chai 分组件报告；ESMFold2 分 folding/ESMC；RhoFold 分主 checkpoint / hub；RF3 分 checkpoint deserialization / engine initialization；ESM1另报 checkpoint 读取计数。
- Chai `weights_loaded` 表示六个 folding component 已齐备，单个 component 的加载/驻留独立报告。ESM 是可选依赖，有自己的计数。CPU offload 的“已加载”不能写成“GPU 常驻”。
- 可观察的 tensor storage 去重计算 `resident_bytes`；没有可观察 tensor 的已加载对象报告 `None`，不能用 `0` 冒充空闲显存。驱动中的 CUDA/JAX allocator、workspace、上下文和临时显存由 broker 实际进程/设备遥测补充。
- `ready=True` 允许 lazy initialization，不意味着无需后续显存准入。
- `release_requires_exit=True` 明确框架 allocator 可能保留显存。清掉 Python 变量、`.to('cpu')`、`empty_cache()` 都不单独构成显存已归还的证据。最终释放由 worker 进程树退出、broker 账本和后续设备遥测确认。

### 具体所有权

| 对象 | Owner / 复用条件 | 释放条件 |
|---|---|---|
| trajectory、科研状态、候选选择、stage transaction | 原 trajectory 进程；不把 legacy 全局可变循环放入线程并发 | trajectory 结束；父进程统一关闭 evaluator/mutator/predocker |
| 物理 GPU、GPU worker 子进程树、准入/驻留/临时预算 | 当前 session 单一 broker；索引/UUID归一到物理设备；所有 worker 的可见设备限制在当前分配 | 释放引用且 idle 可回收；不强行重建仍活跃的私有 RNG 模型；session 最后清理全部已登记进程 |
| AF3 / AF2 参数 | 对应 JAX worker；代码、权重、环境、设备和初始化 config 相同；显式 request seed | client release / broker idle reclamation / session exit；AF3参数只搬一次，host cache仍计入RAM |
| ESMFold2 | 显式 seed + 固定两个 snapshot/两个源码树 + immutable builder/model settings | 同上；`.close()` 从 evaluator → runner → resident → client 传播 |
| ESM1 / Protenix / RhoFold / Boltz / Chai / OpenDDE / Mol1 | 同轨迹 worker；保留 trajectory nonce，避免跳过 constructor RNG 消耗；同一模型连续请求复用 | 轨迹结束显式 `close(release=True)` 并要求broker确认进程树释放；OOM/不确定执行/失效协议由 runtime fence；未确认释放会保留artifact并报错 |
| RF3 | 一个 exact effective config 的 resident engine；原 adapter 的 post-init RNG快照在每请求开始回放 | service关闭所有 engine；新增配置含seed会创建新兼容身份；缓存不继续返回已关闭service |
| Rosetta references | evaluator owner；结构内容、params内容、repack/minimize/seed/replicate/链映射配置都匹配 | 内容/config变化同时失效pose和reference metrics；params变化因进程全局初始化而拒绝原地复用 |

## 3. 真实执行阶段

通用方法是 `DaemonClient.call(... runtime={'phase':'managed', ...})`，worker `serve` 声明允许的 managed methods。CPU feature/result codec 位于外层 CPU slot；只有 `operation(identity, phase='device')` 内获取同一物理 GPU 的 lease。资源/RAM/output仍属于原始科学请求，没有因 nested lease 创建额外样本或科学请求。

| Backend | CPU prepare | 设备准入覆盖的真实工作 | CPU finalize / 未继续拆分的原因 |
|---|---|---|---|
| AF3 pilot | owner featurisation/cache lookup；worker feature decode | 设置当前 topology、显式 PRNGKey、完整 forward、同步 D2H；不拆 native graph | worker result encode；owner extract_structures、score/输出。owner feature/cache、decode、encode、extract分别有时间事件 |
| AF2 | owner native process_features；worker decode | 显式seed predict、JAX output转NumPy | result encode、owner原结构/评分路径 |
| Protenix | owner原input feature/dumper；worker decode和request校验 | RNG restore、原动态skip_amp更新、predict、保持dtype的D2H、RNG capture | codec、owner原dumper；未修改sample数量或默认dropout |
| ESM1 | owner序列合并；worker decode | native infer、native output_to_pdb、必要confidence D2H；该native调用仍含小段CPU工作 | result codec/owner score包装；不是将整个admitted wall time当kernel时间 |
| ESMFold2 | 原owner输入；worker job、ProteinInput/StructurePredictionInput构造 | native builder.fold含嵌套ESMC、模型计算及nativeCPU步骤；CIF字符串及所有可能device-backed confidence的materialization保留在同lease | CIF/JSON写文件独立CPU；具体库版本的内部细分仍需真实版本审查与设备测试 |
| OpenDDE / Mol1 | worker native preprocess_input；CPU MSA/template工具在当前作业进程树中 | native infer_predict含显式seed、dataloader、模型、dumper和CUDA cleanup；保持原生连续执行，不把dumper的设备操作遗漏 | 完整性和output预算校验、owner复制/score；native loop的CPU段仍可优化，当前保守占slot |
| Chai1 | native feature context；GPU ESM调用前后的CPU阶段有时间事件 | ESM embedding单独lease且验证返回CPU；整段run_folding_on_context保留trunk/diffusion/confidence及GPU中间量；不按单组件释放活跃scratch | 原native CPU ranking/文件尾部仍包含在连续fold调用范围；codec独立。没有改变joint diffusion、num_trunk_samples、X masking或low_memory |
| Boltz1 | owner check_inputs/process_inputs、MSA/YAML | lazy model load、Lightning predict及其dataloader/transfer/writer；native复合段整体准入 | RNG codec、owner复制结果；未将Lightning writer误称GPU kernel |
| RF3 | owner准备chains/templates/ligands/MSA路径；worker私有目录和run参数 | native engine.run含pipeline/validation_step/confidence/dump；保守复合nested lease，保留原post-init RNG reset | worker RNG codec及output预算校验、owner原score路径；不宣称Foundry内部pipeline完全CPU或已做GPU细分认证 |
| RhoFold | owner MSA token / RNA-FM token；worker decode | 模型及RNA-FM；原GPU上的sigmoid/softmax保留在GPU且在D2H前完成 | CPU NPZ、ss.ct、RNAConverter PDB；relax作为单独composite请求，绝不在GPU-hidden owner隐式跑OpenCL |
| Rosetta / Dummy | owner CPU | 无GPU推理 | 全部CPU阶段；PyRosetta进程级环境不可“卸载”成未初始化 |

`admitted_seconds`、`handler_seconds` 和 composite stage 是墙钟范围，不是GPU kernel耗时。GPU activity只取实际设备样本；没有采样时标 unavailable。原始full-input树的实际文件字节计入 prepared，shape key只包括形状/dtype/字符串长度，不将序列写入遥测。cold未被adapter强写false：broker按同身份、shape和phase的首次观测标记；这不自动证明某次JIT编译已发生。

## 4. 科研语义、RNG与提交安全

1. `torch_rng.py` 保持 Python random、NumPy、torch CPU和所用CUDA generator的完整状态。GPU-hidden owner只在原有 `set_random_seeds` 调用点登记真实seed；未知已有CUDA消耗时不使用 `torch.initial_seed()`猜测。
2. 第一次下放时worker使用该明确seed；随后返回真实消耗后的CUDA RNG bytes，owner按物理device存储，下一evaluator或mutator收到同一stream。marker/device/完整状态先验证再修改owner RNG。设备不在allocation内立即拒绝。
3. 需要保留constructor RNG的模型使用trajectory nonce；不声称跨轨迹消除全部重复加载。AF2、AF3及符合显式seed/固定snapshot条件的ESMFold2允许兼容共享。
4. RF3保留其原代码的特殊语义：每次run都恢复模型initialize完成后的快照；不把它改成连续推进的随机流。
5. 不更改candidate数量、seed、precision、recycles、diffusion step、acceptance、stage transaction、fixed/shared segment或mutator联合解码。Protenix requested/effective variant仍分别记录。
6. 每次请求使用私有request目录，结果必须匹配该请求。只有完整结果才复制到原有科学输出路径。`OutcomeUnknown`时保留目录直到session确认退出/隔离，避免删掉仍在运行worker的数据；不自动重试为新科学请求。
7. 标准tree codec在分配的output额度内先检查payload；外部vendor直接写出的文件在返回前另查总大小，broker也采样中途文件/RSS。**这是有采样窗口的检测/终止，不是文件系统hard quota。**
8. 原同进程Torch后端还交换进程全局数值设置：default dtype、deterministic/warn-only、cuDNN deterministic/benchmark/TF32与float32 matmul precision。这样保留 `set_random_seeds` 的cuDNN设置，以及Boltz原生 `set_float32_matmul_precision('highest')` 对后续backend的影响。初始设置进入模型身份；请求设置摘要进入shape观察。原外部解释器后端不新增owner设置副作用；RF3仅原同进程分支往返设置，post-init回放仍只重置原来的RNG。
9. `WorkerArtifacts.start_client/close_client`在启动或释放未知时调用 `retain()`，解除私有TemporaryDirectory的自动GC删除；普通close也不能删除retained root，只有明确释放确认后可以清理。shared root始终归session。重复关闭仍保留并传播未确认释放；absence of client handle不是bootstrap可删除的证据。该行为有真实临时目录和GC契约测试。

## 5. RhoFold OpenMM 的额外设备依赖

固定RhoFold的 `AmberRelaxation(use_gpu=False)` 并不保证CPU：它先选OpenCL，失败时交由OpenMM自动选platform。只隐藏CUDA不能证明OpenCL绑定，因此shared模式在 `relax_steps>0` 时必须明确 `openmm_platform`；该选择必须与欲复现的旧运行一致。

- `CPU` / `Reference` 明确无GPU relaxation；CPU线程数受当前worker CPU预算约束。
- `CUDA` 必须使用该worker唯一可见的获配GPU，`DeviceIndex=0`。
- `OpenCL` 必须给 `openmm_properties.OpenCLPlatformIndex` 和 `DeviceIndex`；`openmm_binding.py`通过Khronos PCI bus信息，与仅查询获配token的NVIDIA PCI/UUID比较；缺能力或不匹配时拒绝。不能仅凭OpenCL设备序号声称在allocation内。
- `Modeller.addHydrogens`也会单独选择platform，故在请求内临时绑定到同一平台，finally恢复原方法。没有修改relax_steps、力场、采样预算或默认数值precision。

依据：固定RhoFold [`relax.py`](https://github.com/ml4bio/RhoFold/blob/6bdfbda720184409eb682ce08c05d258162ddc48/rhofold/relax/relax.py)，[OpenMM platform properties](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html)，[Khronos PCI bus query](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetDeviceInfo.html)。该路径尚缺真实OpenMM/GPU数值与释放验收。

## 6. 认证矩阵与原始证据

`P` = production adapter + 小型CPU替身模型/真实transport契约；`U` = 无真实模型/GPU运行证据；`N/A` = 无该GPU推理能力。**P不是GPU数值认证。**

下表完整后端机制测试的远端证据为snapshot04：所有收集到的evaluator、RNG、Torch数值设置、AF2共享worker和GC保留测试通过。随后snapshot06对公共进程释放、CLI、AF2共享和生命周期等受影响范围完成148项复验，全部通过且无skip。测试运行于已有科学容器，使用真实CPU框架/进程传输和小型替身模型；真实权重、GPU数值等价及性能结论仍按最后一列的U或N/A记录。最终Dummy CPU三模式固定工作量27/27完成且评分/决策一致，但该配置没有加速；完整口径见[实现报告](runtime_implementation_report.md)，不能用它升级任何预训练GPU后端的认证等级。

| Backend / 配置范围 | 单请求、连续复用、隔离、失败清理的已有测试入口 | 缓存/RNG证据 | 真实模型 / GPU / 加速结论 |
|---|---|---|---|
| AF3 pilot | `test_backend_daemons.py` production handler；managed边界已随snapshot02通过 | feature cache命中、序列/权重身份变化失效、constraint未编译拒绝 | U；尚无真实AF3固定任务三组性能数据 |
| AF2 exact test suffix=ptm | `test_af2_resident.py`与共享`test_af2_broker_resident.py`随snapshot04通过；两owner共用worker、initial guess隔离和worker死亡fence均覆盖 | CPU prepare不载权重，worker一次load，显式seed；fixture显式CPU绑定，生产GPU准入未放宽 | U；CPU JAX替身不认证真实multimer/ptm GPU |
| RF3 CPU fake Foundry | `test_rf3_resident.py`、原`test_rf3_evaluator.py` | 相同effective config复用；同seed原post-init RNG reset，不同seed新engine；close等进程退出 | U；需安装的Foundry源码摘要+权重+GPU验收 |
| RhoFold CPU tiny nn.Module | `test_rhofold_resident.py`、`test_rhofold_evaluator_compat.py` | 原随机draw与sigmoid/softmax对照；2次请求一次加载；物理OpenCL匹配/拒绝的CPU契约测试 | U；RNA-FM/RhoFold实权重和OpenMM均未认证 |
| ESMFold1 CPU fake model | `test_esmfold1_resident.py` | constructor/request RNG round-trip、bfloat16 transport、复用和错误fence | U |
| ESMFold2 CPU fake session | `test_backend_daemons.py`、`test_esmfold2_resident_settings.py`；新`test_evaluator_lifecycle_contract.py`分相位契约 | immutable builder settings；两个模型loader计数；请求输出隔离；CPU写文件不占device lease | U；没有以fake builder认定嵌套ESMC数值通过 |
| Protenix 四个名称 | `test_protenix_resident.py` | CPU真实torch/Python/NumPy RNG；动态skip_amp；四variant factory身份保留 | U；factory参数测试不认证四套真实模型 |
| OpenDDE CPU fake API | `test_opendde_resident.py` | 成功checkpoint loader一次；多个显式seed/job输出隔离；不支持分布式拒绝 | U |
| Mol-1 base CPU fake API | `test_mol1_resident.py` | 多request一次loader；sample计数/完整输出检查；不支持输入提前拒绝 | U；constraint variant未接入/未认证 |
| Boltz1 CPU fake model/trainer | `test_boltz1_resident.py` | hello0 → 首请求真实loader1 → 重复仍1；native参数保留；失败fence | U；Boltz2 N/A |
| Chai1 CPU fake components | `test_chai1_resident.py`、`test_evaluator_lifecycle_contract.py` | lazy0、单component1；异常finally offload；两种residency策略契约；managed hook已随snapshot02通过 | U；无GPU常驻模式收益认证 |
| Rosetta crystal | `test_rosetta_crystal_evaluator_contract.py`、新增reference dependency契约 | 内容/参数/计算config变化失效；legacy PyRosetta已初始化不虚报owned init | 真PyRosetta固定协议待运行；GPU N/A |
| Dummy | 原pipeline fixture与runtime workflow benchmark | 固定输出/完整trajectory由公共runtime验证 | GPU N/A；仅CPU机制基线 |

已有原始日志：`evidence/runtime-20261103/snapshot01-command.json`（实际命令、GPU disabled）、`source-snapshot-01.json`（源快照）、`snapshot01-backends.log`。该混合evaluator/mutator测试批次是112 passed、1 failed、1 skipped；唯一失败在mutator的ProDy Path兼容，所有收集到的evaluator测试通过。日志还出现旧RF3 fake process缺wait的atexit噪音，后续fixture已补wait/kill。快照之后的managed边界、OpenCL binding与新增cache测试不能借用snapshot01冒称已经远端通过，需使用后续快照日志。

第二次远端日志为 `snapshot02-command.json`、`source-snapshot-02.json`、`snapshot02-integration.log`：321 passed、4 failed、1 skipped。所有收集到的evaluator backend测试通过；失败包含3个真实CPU CLI运行的stage phase名称不一致，以及1个旧unsupported-backend测试期望。这些失败保留在原始日志中。

第三次远端 `snapshot03-integration.log`：377 passed、2 failed、1 skipped，408.65秒；前一轮CPU CLI phase问题已通过。环境身份、RF3/ESMFold2 managed边界与Torch数值设置往返测试通过。两个剩余失败是CPU CLI重叠断言，以及AF2共享fixture误继承另一测试遗留的 `CUDA_VISIBLE_DEVICES=sentinel`，被真实UUID准入正确拒绝。runtime修复污染源，AF2 fixture另显式绑定CPU；生产GPU准入没有放宽。

第四次远端原始证据为 `snapshot04-command.json`、`source-snapshot-04.json`、`snapshot04-integration.log`：**403 passed、1 failed、1 skipped，469.72秒**。全部收集到的evaluator测试通过，包括AF2共享worker、RNG/数值设置、启动/关闭未知结果与GC保留，以及上述公共执行代码身份。唯一失败位于新增benchmark进程测试的fixture：`child.pid`刚创建尚未写入时读取空串；该失败不属于evaluator数值或生命周期测试，原始失败仍完整保留。唯一skip为已由统一broker测试覆盖的旧admission检查。

第五次远端定向验证的原始证据为 `snapshot05-command.json`、`snapshot05-validation.log`：进程观察和benchmark两个测试文件 **21 passed、1 warning、无skip，2.92秒**。PID文件fixture竞争已修复并通过重跑，关闭了snapshot04唯一剩余失败；evaluator生产实现未改变。此定向结果与snapshot04完整批次分开报告，不合并成不存在的全套测试批次，也不作为GPU性能证据。

本地轻量环境无torch/JAX/GPU；环境身份、生命周期、Rosetta和ESMFold2 immutable settings原批次32 passed；后续环境身份/lifecycle/mutator具体GC复现联合31 passed。真实CPU torch子进程回归 `test_worker_preserves_constructor_dtype_cudnn_and_native_precision_side_effect` 随snapshot03及snapshot04通过，验证构造设置、Boltz已知precision副作用和连续RNG。准确结果以对应快照的命令与日志为准。

## 7. 复现和下一轮真实验收入口

CPU机制测试（装有对应vendor依赖的现有环境，明确隐藏GPU）：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  python -m pytest -q \
  tests/test_evaluator_lifecycle_contract.py tests/test_evaluator_rng_lifecycle.py \
  tests/test_worker_environment_identity.py tests/test_esmfold2_biohub_wrapper.py \
  tests/test_rhofold_resident.py tests/test_rf3_resident.py \
  tests/test_backend_daemons.py tests/test_af2_resident.py \
  tests/test_protenix_resident.py tests/test_esmfold1_resident.py \
  tests/test_opendde_resident.py tests/test_mol1_resident.py \
  tests/test_boltz1_resident.py tests/test_chai1_resident.py \
  tests/test_rf3_evaluator.py tests/test_rosetta_crystal_evaluator_contract.py
```

公共实工作流入口为 `run.py`，各factory使用 `execution=daemon`，统一调度由session runtime配置选择；对照入口与固定有效工作量报告由 `scripts/validation/runtime_workflow_benchmark.py` 提供。真实后端验收必须先固定：实际source/checkpoint/env摘要、获配GPU UUID/CPU/RAM、输入、seed、样本数、precision、协议、时间/GPU小时预算，然后交错执行旧直连、常驻串行、常驻自适应。

每个variant都需要：真实single、至少两次连续请求且load count不重复、缓存开关等价、跨请求隔离、故障/超时后无错误提交且实际进程/显存释放；数值容差从该后端direct重复误差取得，score/selection/acceptance差异独立报告。性能必须按固定有效完成任务数计量，并分别报告冷启动、稳态、墙钟有效吞吐、总资源成本、GPU活动/等待原因、CPU/RAM/VRAM峰值和失败；profiler仅解释原因。

当前尚未获配用于真实模型测试的GPU和实验预算，不能推出任何真实GPU加速百分比。已知收益不足/仍保守的范围包括：需要constructor RNG兼容的后端尚不跨轨迹共享；RF3/ESMFold2/Lightning部分CPU尾部仍在native复合范围；Chai nativefold的CPU ranking尾部仍保留连续调用；外部输出预算没有严格文件系统hard quota。这些限制必须随运行manifest和报告保留。
