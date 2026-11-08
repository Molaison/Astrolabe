# Proton-PottsMPNN 与 pH-sensitive binder 评分

安装、来源和运行资产说明见 [Proton 安装与来源](proton_installation.md)。
当前产品源码包含固定上游源码和已记录的本地补丁；v6 checkpoint 与 EV6 labeller
的 pkl 是独立 runtime assets，不进入 Git 或源码包。保留
`GH_PROTON_POTTSMPNN_PYTHON`、`GH_PROTON_POTTSMPNN_REPO`、
`GH_PROTON_POTTSMPNN_HBPLUS` 和 `GH_WEIGHT_PROTON_POTTSMPNN` 配置入口。

内部网络无需预安装、无需固定 His 的模式见
[自动 His 网络与两态评分](ph_core_automatic.md)，使用
`examples/02_ph_core_automatic.overlay.yaml`。下文的 01 示例保留原接口设计用途。

将 `examples/01_ph_sensitive_design.overlay.yaml` 用作已有 **A=binder、B=固定蛋白 target**
任务的最后一个 `--overlay`。示例替换 `ligand` 阶段的 mutator，并在
`benchmark_binder_v1_4` 的结构质量目标上加入有界 pH 静电差优化项；不是独立 target 输入。
若阶段、链、alias 或原结构质量目标不同，按实际任务修改对应字段。

当前 refinement 配置使用 `Q + 0.2 * (2 * sigmoid(ph_release / 5.0) - 1)`，
其中 `Q` 是原结构质量目标，`ph_release` 是 `ph_dddg_elec`，尺度 5 的单位是 Rosetta units。
pH 项限制在 −0.2..0.2，正的静电差提高得分。这些是明确的工程起点，不是论文给出的权重，
不等于实验自由能标定。`ph_release >= 0` 且 `ph_acceptors >= 1` 用于最终筛选，
**不是逐步硬 gate**；`ph_score` 作诊断，不直接据其最大值判断实验 pH-sensitivity。
提供参考结构时可额外记录 `interface_backbone_rmsd`，不擅自增加 cutoff。
该可选指标已按原分支实现接入 `losses/structure.py`，保持原接口与几何定义；
当前 main 候选的统一 SIF 验证由发布 owner 执行，本次补齐未新增科学验收结论。

## 两个独立来源

- **Proton-PottsMPNN**：[christian-creator/ProtonPottsMPNN](https://github.com/christian-creator/ProtonPottsMPNN)，
  固定 revision `09682abfa7d20e0abcdeea0490b7a4b1c190aee3`。
  上游称其为 *pH-sensitive binder design with Proton-PottsMPNN* (Jacobsen et al., 2026)。
  它不是下面 Ahn 论文的 ProteinMPNN 方法。
- **指标来源**：Ahn et al., *Computational design of pH-sensitive binders*，
  [bioRxiv v1, 2025-09-29](https://doi.org/10.1101/2025.09.29.678932)，
  [原文镜像与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC12621740/)。
  定义来自 Methods 的 “Interface histidine design with electrostatic repulsion filtering”
  和 “Buried charged residue network”，以及 Fig. S4、S6。

## Mutator 的用法与边界

```yaml
mutator:
  mutator_terms:
    ph_design:
      _type: proton_pottsmpnn
      binder_chain: A
      position_selector: interface(interface_distance=8.0)
      criteria:
        method: block_descent
        backend: potts
        combined_lambda: 0.3
        block_size: 2
        temperature: 0.0
        center_types: [HIS-S]
        dep_map:
          HIS-S: [HIS-P]
          HIS-P: [HIS-S]
          ASP-P: [ASP-D]
          GLU-P: [GLU-D]
```

原生 `PHDesignCriteria` 参数通过 `criteria` 传递。这里的 `combined_lambda`
调节上游联合目标中的稳定性与选择性；不是 Astrolabe 的 loss 权重，也不是 pH 值。
v6 的中性 His token 是 `HIS-S`，不是旧词表的 HID/HIE；`dep_map` 必须与 v6 一致。

**方向不能混用。** 上游优化最小化 `E(target state)-E(contrast state)`。模型默认以
`HIS-P` 对比 `HIS-S`，偏好质子化态；这不自动等于 Ahn 论文的酸性释放方向。
上面的论文配套示例固定中性 `HIS-S` 为 target、`HIS-P` 为 contrast，从而在该原生
选择性项中偏好中性状态。它仍不是实验 affinity 或 binding free-energy 差，须由
`ph_dddg_elec` 及结构质量评估补充检查。通用 mutator 的默认中心类型保持上游语义。
Proton 也支持 Asp/Glu 中心；Ahn 这组 His-state loss 不会为 Asp/Glu 做连续 pH 滴定。

模型优化完整的联合序列；adapter 选取一条真实设计并提交其实际编辑，**不把多个
Potts 解平均成独立位点概率**。固定位必须在模型的设计掩码中约束，超出 mutation
budget 的解报错而非截断。canonical 序列用于后续预测，原生 protonation tokens、
中心和能量保留在 raw model outputs 中。

snapshot 包含明确记录的本地补丁。原兼容补丁
`patches/vendor/proton_pottsmpnn/01_preserve_design_mask.patch`：在 AtomWorks 补齐缺失原子时，
按 chain/residue/insertion-code 保留设计掩码。它在优化前约束固定位，不事后改回序列；
上游 revision 与本地改动分别记录于 `SOURCE_PROVENANCE.json`。另一个补丁
`02_automatic_histidine_bias.patch` 增加可选的无中心 bias 和联合突变预算，
不修改模型权重或原始能量。

当前接入只接受一个 active complex，且恰有一条可变蛋白 binder chain；不支持 tied/symmetry、
多可变链或多结构联合设计。其他正向 context 不会各自重复使用完整的 mutation budget。
不能用模型不支持的上下文静默替代。PDB 链的序列与
当前 Astrolabe 序列必须一致。

### 运行环境

Astrolabe 的 Python 3.11 进程通过持久子进程调用原生 **Python 3.12** 模型。
Proton daemon 首次实际设计时才加载，后续请求复用同一模型，不逐步重载。
现有 `evaluator.scheduling.overlap_cpu_postprocess` 用于多个独立评估 work item；
单个 AB refinement 的下一步依赖本步评分及接受/拒绝，开启该选项不会消除这条依赖。
性能应比较完整轨迹耗时，不能仅以平均 GPU 利用率判断。
跨轨迹并行应使用独立进程，并隔离输出、raw 和临时目录，按显存与总 CPU 预算限制并发。
只提高每卡进程数不保证 CPU/GPU 重叠：副本可能锁步，或重复加载模型、争抢同一 GPU。
AF3 可使用下述共享推理 owner，让独立轨迹的 CPU 评分与另一轨推理重叠。
部分轨迹或更高的 GPU 活跃度不构成性能验收。

配置项是 `mutator.ProtonPottsMPNN`，term 可覆盖同名项：

| 项 | 含义 |
| --- | --- |
| `python_bin` | 外部 Python 3.12 可执行文件 |
| `repo_path` | 本仓库的 `vendor/proton_pottsmpnn` |
| `checkpoint_path` | v6 `potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt` |
| `hbplus_bin` | 外部 HBPLUS 可执行文件 |
| `extended_vocab` | 固定 `v6`，与 checkpoint 一致 |
| `device` | `auto`、`cpu`、`cuda` 或 `cuda:N` |
| `seed`, `n_jobs` | 模型 seed；原生 CPU solver 并行度，默认 1 |

外部环境须具备上游 foundry/mpnn、torch、atomworks、biotite，以及 EV6 labeller 的
FLAML/xgboost/lightgbm/scikit-learn/pandas 等依赖。源码固定并保留在 `vendor/`。
**这个 revision 的 v6 featurisation 会调用 HBPLUS**，不能仅凭上游 README 中
“design 不需要 HBPLUS”判定环境就绪。HBPLUS 不由默认安装器下载，也不随源码或 SIF 分发。
请从 [EBI 官方入口](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/)
指定的 RomanLas/HBPLUS 获取源码，确认许可后在私有目录运行其原生 `make`。
官方标称 v3.06；固定源码 `0bd4df6aab49054322c23c4dcfcc07075a84bc05` 的二进制横幅为 3.2，
不要仅按横幅推断下载了另一版本。公开下载不替代源码内的使用许可条款。
本集成不自动执行上游 `install.sh`、清空 venv 或安装系统/runtime 依赖。
使用容器时，完整的外部 Python 3.12 runtime、HBPLUS、源码和 checkpoint 路径还须在
容器内可见，不能只绑定 Python 可执行文件。本次接入不等于现有 SIF 已重建或包含这些依赖。

## 跨轨迹共享 AF3 GPU 模型

默认 `service_socket: null` 保留原来的本地推理。需要同卡多轨迹时，复用本地
`inference_service.call/serve`：一个 GPU owner 加载一次 AF3 模型并串行完成同步推理，
各轨迹进程独立进行 featurisation、结构提取、Proton 设计、CPU 评分和接受/拒绝。
它不缓存预测或设计结果，也不合并不同轨迹的随机状态。

在同一分配中启动 owner；`owner.yaml` 是已解析的 AF3 参数映射，不是整个任务配置：

```bash
rtk proxy python -m astrolabe.evaluators.af3_service \
  --socket /af3-service/model.sock --config /af3-service/owner.yaml
```

owner 配置不选远端 `service_socket`，且必须有真实 JAX GPU；无 GPU 就失败，不回退 CPU。
每条轨迹给 `evaluator.evaluator_terms.Alphafold3.service_socket` 指定同一个 socket，
并在启动轨迹进程前设 `JAX_PLATFORMS=cpu`。**不要清空 `CUDA_VISIBLE_DEVICES`**：
Proton 的独立 Torch daemon 仍需使用分配到的 GPU。
客户端和 owner 的 `model_dir`、`flash_attention_implementation`、`num_samples`、
`num_recycles`、`save_embeddings` 必须一致，不一致的请求明确报错。

socket 必须位于同 UID 的私有短路径（父目录 0700、socket 0600）；仅供可信本地进程，
不监听网络，不能接入不可信用户。容器必须把同一私有目录绑定到所有进程可见的路径。
分别隔离各轨迹的输出、raw 和临时目录，并把 owner 与全部 worker 纳入同一 CPU/内存预算。

全部轨迹结束后，由分配的原 owner 发出 shutdown 并等待已接受请求排空及进程退出：

```python
from astrolabe.evaluators.inference_service import call
call("/af3-service/model.sock", {"type": "shutdown"})
```

模型启动、首次编译、每步反馈、落盘和收尾都应计入完整吞吐比较；较低 GPU-util
不必然代表这条路线更慢。性能结论只适用于实测输入、硬件和资源配置。

### 最新完整轨迹验收：TY A800（2026-10-22 记录）

同一张 A800、1 GPU／4 CPU 亲和性／64 GiB 内存上限，固定五对比较
“共享两轨、两波”与“修复持久缓存＋共享四轨、一波”。每组均为四条完整 11 步轨迹，
科学配置不变；配对几何吞吐提升 **1.647953×**，95% 区间 **[1.576621, 1.782523]**，
通过预声明的下界 >1.05 门槛；计时窗内 GPU 秒比为 **0.605370**。
40 条技术复跑逐步通过同卡原版 JAX 0.4.34 参考检查，最终 pH 筛选通过数仍为 0。
同 seed 的技术复跑不是 40 个独立科学设计。

缓存修复仅将固定 XLA 的 `dnn_compiled_graphs` proto value 从 `string` 改为 `bytes`，
见 `patches/vendor/jax_cudnn_cache/01_cudnn_graph_bytes.patch` 和
[上游修复](https://github.com/openxla/xla/pull/21149)。实测使用私有、按 SM80 构建的
PJRT wheel；不升级 JAX、不修改共享环境，也不由默认安装器自动启用此补丁。
共享 owner 本身不保证旧 JAX 的 cuDNN 持久缓存可加载。

收益属于缓存、owner 生命周期与四轨供料的组合，不能单独归因缓存或并发。
硬件参考变更、旧 3090 差异、五对原始耗时、利用率窗口与验证边界见
[验收记录](01_ph_sensitive_acceptance.md)。

### 历史本地完整轨迹验收：RTX 3090（2026-10-21）

9TP A85/B87、seed `20261020`、每轨迹 11 步；同一 RTX 3090、4 CPU、64 GiB。
五对随机顺序比较，每批两条轨迹，包含冷启动、JIT、全部反馈、落盘、收尾和独立验证：

| 实测项 | 两轨迹串行 | 共享 owner＋两轨迹并行 |
| --- | ---: | ---: |
| 批次完整耗时中位数 | 537 秒 | 279 秒 |
| 有效完整轨迹／小时 | 13.36 | 25.51 |
| 整窗口平均 GPU 利用率 | 14.9% | 29.2% |

配对吞吐改善 **1.910×，bootstrap 95% 区间 [1.873, 1.948]**；分配 GPU 时间估计降低
47.6%。20 条完整技术复跑的逐步序列、决策、约束、分数和坐标均通过参考等价检查。
每个共享批次至少约 39–46 秒 CPU 评分与另一轨迹 AF3 forward 重叠；forward 也含 host/JIT
工作，不等于 kernel-only 活跃时间。CPU 评分及 Proton 解析／求解仍有空档，不声称 GPU 满载。

原分配在第五对遭 SIGINT，当时来源未知；后续构建中断定位到个人 watchdog 对 Slurm
后代的误判并另行修复，历史证据保留，不冒充每次信号的 siginfo 收据。
经用户事先同意，在同一物理 GPU 的新分配中
只重跑第五对，前四对保留，未增样。结论适用于这份**修订后的恢复合同**，不是原单分配合同
通过；中断 attempt 保留在配对估计之外，额外分配消耗另计，不并入上述配对成本比。
这批同 seed 技术复跑没有通过最终 pH 筛选，新突变均被原目标拒绝，不是合格 binder 或
实验结合证据。完整本地记录见 `logs/proton_pottsmpnn_20261020/17_acceptance.md`。

## 新增 loss terms

所有返回值均是原始可审计指标；Astrolabe 的 `weighted_score_expression` 是越大越好，
因此要最小化的项应加负号，或直接放入 threshold/gate。

| `_type` | 定义、单位与方向 |
| --- | --- |
| `ph_score` | 论文 20 项 His 界面氢键模式加权和；较高符合论文的筛选倾向，无已发表通用 cutoff |
| `ph_dddg_elec` | `ΔG_elec(low) - ΔG_elec(high)`；正值表示低 pH 静电结合较不利，Rosetta units |
| `ph_histidine_cation_acceptors` | 接受 target His/Lys/Arg 侧链氢键的不同 binder His 数；论文筛选至少 1 |
| `ph_histidine_cation_contacts` | all-library rotamer 最短距离满足指定 cutoff 的不同 His-cation residue pairs 数 |
| `ph_positive_charge_count` | 指定链上 Lys/Arg 与双质子化 His 的数量；`phase: high` 或 `low`，不是净电荷或一般 pKa 模型 |
| `ph_histidine_hbond_energy` | 指定 His-(His/Lys/Arg) pair 最有利的侧链氢键能；更负更有利，无氢键返回 0 |
| `ph_histidine_hbond_depth` | 同一主氢键两端重原子 AtomicDepth 的平均值，Å；无氢键返回 0 |
| `ph_histidine_pair_energy` | 固定骨架四点 glycine baseline：`E(H,P)-E(G,P)-E(H,G)+E(G,G)`，ref2015 units |
| `interface_backbone_rmsd` | reference-contact 界面两侧 N/CA/C/O 联合拟合 RMSD，Å；越小越好 |

界面项需要 `binder_chain`、`target_chain`，支持 `chain:A`、`protomer:A` 等已有 chain
selector 语义。`ph_positive_charge_count` 需要 `chain`。
His pair 的 `histidine_residue`、`partner_residue` 是**整个预测结构的 1-based pose index**，
不是每条链独立的 PDB 编号；第一个必须是 His，第二个必须是 His/Lys/Arg。

`ph_histidine_cation_contacts.distance_cutoff` 必须显式给出，单位 Å。
默认 `atom_mode: cation_nitrogens` 使用 Fig. S6C 的原子：His ND1/NE2、Arg NE/NH1/NH2、
Lys NZ；`atom_mode: sidechain_heavy` 则使用全部侧链重原子。
这是 rotamer library 中可能接近的 residue pair 数，**不代表这些 rotamers 能同时存在**。

`interface_backbone_rmsd` 需要 `reference_complex`；参考界面由两侧重原子距离
`contact_cutoff` 定义，默认 10 Å。链内按 sequence position 对应，允许不同 PDB 编号，
但链长度、界面 backbone coverage 必须一致；没有界面或缺少原子会报错。
它不是 target-aligned binder RMSD。

## 论文公式与实现约定

`ph_score` 的 20 个系数逐项保留在 `PH_SCORE_WEIGHTS`。也可以将完整的 20 项计数传给
`ph_score_from_counts()` 单独核验公式。计数单位是 **His residue**，不是氢键总数：
同一 His 的同一供体氮接触多个 acceptor 不会被误算成双供体。

以下细节在论文/补充材料中不充分，当前实现明确约定如下，**不声称逐值复现作者未公开脚本**：

1. 低 pH：按 Methods 的“所有 His 双质子化”状态约束构造；高 pH：中性 His，允许
   HIS/HIS_D tautomer。两态在 ref2015 下 repack。论文通过 pH=0、e_pH=100 使低态 His
   全部质子化，并用该状态代表低 pH；这里按 pose 加载 HIS_P，避免修改进程全局 pH 选项。
   这不是对任意 pH 的连续滴定。其他可滴定残基的状态不变。
2. core 使用 Rosetta `LayerSelector` 默认 core 分类，boundary 归入 surf；默认只计能量
   不大于 0 的氢键，可用 `hbond_max_energy` 收紧。含任一跨链 backbone 氢键的 His 进入
   单独的 `_bb` 箱，不同时累计到非 `_bb` 箱。论文未公开这些边界与混合模式的判定代码。
3. `ΔG_elec = E_elec(bound) - E_elec(binder) - E_elec(target)`，评分函数仅开启
   `fa_elec=1`。两态均先 repack；默认分离后不再 repack，`pack_separated: true` 可以开启。
   这是静电结合能差，不能替换成复合物总静电能差。正号方向来自论文的低态减高态定义。
4. 深度使用最有利 pair 氢键，取 donor heavy atom 与 acceptor 的平均 AtomicDepth；
   默认 probe 1.4 Å、resolution 0.25 Å。四点 glycine cycle、rotamer 库及距离 cutoff
   是明确的实现配置，不冒充论文未提供的精确参数。
5. Fig. S4 没有说明 positive charge count 的统计链与细节；本实现要求选链并注明两态。
   不能把它当作作者该图的逐点复现。
6. 论文没有给出 interface RMSD 的配准/选原子协议。本实现采用上面明确的参考界面
   backbone 定义，不能与另一种 RMSD 定义的阈值直接互换。

Rosetta energy units 不是经过标定的实验结合自由能。论文使用主网络氢键能 `< -0.5`
作为筛选条件，但这不意味着模型能预测实验 `Kd`。

## 复用的指标与不能从结构得出的测量

- `ipae_mean`：两侧 PAE 矩阵均值，Å。论文早期用 `<20`、最终用 `<8`；仅在 evaluator
  确实输出有效 PAE 时适用，不将缺失 PAE 当作通过。
- `plddt_mean_norm`：Astrolabe 已规范到 0..1；论文 `pLDDT >88` 对应 `>0.88`，不是 `>88`。
- `bindcraft_interface_dG` 与 `sap_score`：复用已有能量/aggregation-propensity loss，
  但原文没有公布通用通过 cutoff，不能补造。
- 原文的 SPR `Kd`、BLI `koff`、yeast enrichment、降解 MFI、信号/内吞实验必须有实验数据，
  不新增伪结构代理 loss，也不将计算筛选称作实验 pH-sensitivity 已验证。

## 验证层次

公式、注册、LossEngine、真实 PyRosetta 氢键/深度/两态与 mutation adapter 的契约测试，
和 **原生 Proton-PottsMPNN 模型推理** 是不同验收层。只有外部 Python 3.12、其依赖、
checkpoint、HBPLUS 均就绪并实际运行后，才能声称该环境完成模型推理验收。

## Parity 源码身份

`astrolabe posthoc proton-parity --arm pristine|clean` 会在重型导入、资产访问和模型调用前，
校验 `potts_mpnn_ph.py` 与 `utils/inference.py` 的原始 Git blob，固定上游 commit
`09682abfa7d20e0abcdeea0490b7a4b1c190aee3`。发行默认 vendor 含五个补丁，不能作为
这两个 arm 的原始源码；请显式准备独立原始源码树并用 `--source-root /path/to/original`
选择它。程序不会逆补丁或修改共享树。`clean` 只在验证原始源码后安装进程内掩码 adapter。
`legacy` 仍使用独立 `repo/vendor/proton_pottsmpnn_legacy`，报告记录其实际文件身份，
不将其标为上游原文。此门禁覆盖两个关键文件，不代表对整个源码树或科学结果的验收。
