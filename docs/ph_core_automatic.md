# 自动生成内部 His 网络

使用 [02_ph_core_automatic.overlay.yaml](../examples/02_ph_core_automatic.overlay.yaml)
作为已有 A=binder、B=固定蛋白 target、含 `ligand` 阶段任务的最后一个 `--overlay`。
其他链命名、完整三聚体或不同质量目标，须在实际任务中调整 selector、oligomer 和 Q。
这份 overlay 不含 target 输入，不会自动启动 parent 批次。

本模式从原 parent 序列开始，允许零 His；不生成预安装 motif，也不设置 His 位点。
每次提案使用当前结构的可变掩码，`greedy_energy_block` 联合优化允许位点。
`center_count: 0`、空 `center_types` 与 `explicit_centers` 禁用中心安装和 pin。
His、R/K 及支持位点均可在后续优化中变化。

## 突变 bias

原生 Potts 稳定性目标加入以下可选项，优化时最小化：

\[
J(S)=H_{stab}(S)/\sigma_H-\sum_a b_a\min(n_a(S,M),c_a).
\]

`aa_bias` 使用三字母 canonical AA 名；正值提高该 AA 的提案偏好。
`aa_bias_region` 与当前可变掩码相交形成 M，默认示例只偏好当前 core 的 His。
`aa_bias_caps` 限制奖励增长，不强迫某个 His 数量，也不固定位置。
未设置 bias 时保留原生稳定性目标。sigma 是该提案内冻结的单点突变能量尺度，
不随每次 block 更新重新标定。原始模型能量及后续两态能量均不包含 bias。

示例 `HIS: 2.0`、奖励饱和于 2 个 His；这些是待校准的工程起点。
首个原始零 His parent 的真实 CPU canary 中，bias 0/0.5 均未引入 His，
bias 2.0 引入两个 His，但局部 pack 后未形成合格网络、释放奖励仍为零。
示例因此采用 2.0 作为能触发 His 提案的起点；一个 parent 的结果不是普适标定。
bias 不能保证每条提案都生成 His，更不能替代网络或结构评分。
示例限定提案 His 为 HIS-S、Asp/Glu 为去质子化 token，输出仍为 canonical 序列。

固定 target、用户掩码与 selector 始终生效。示例另外保护已成键的二硫键 Cys 和 G/P。
`mutation_num: 8` 是单次提案变化上限；如需相对原 parent 的累计上限，必须同时设置
`reference_sequence` 和 `max_total_mutations`。参考序列须明确保存并在恢复时复用。
两个预算都在联合求解内部限制，输出再次核验，不截断联合解。

## 每轮打分

`_type: ph_core_network` 的三个 alias 使用同一当前预测结构的 binder-only apo 坐标。
从 target 分离后重编号为 A/1..N，并保存原编号映射；不移动主链、不重折叠。
当前协议保留 PyRosetta 读入的中性 His 互变异构体；不做互变异构体穷举或 repack。

网络边是侧链 His 环接受来自 His、Arg 或 Lys 侧链的氢键。对每条有利边 e：

\[
q_e=\min[\operatorname{clip}(-E_{HB,e}/0.5,0,1),
         \operatorname{clip}(d_e/4,0,1)].
\]

d 是实际氢键两端重原子的平均 AtomicDepth。His 与 R/K 作为图节点，
共享 R/K 可把多个 His 连入同一网络。连通分量质量为其边质量最小值，
`core_neutral` 取质量最高的一个分量 N；不按 His 数或网络数量累计奖励。
强度 0.5 和深度 4 Å 借用已有网络设计的尺度，本处连续映射与分量规则是工程定义，
不是已验证的 TNFα cutoff。依据见 [现有文献说明](ph_sensitive_design.md#两个独立来源)。

对该分量内所有 His 组成的动态集合 C，在同一次 Potts 前向得到的上下文中计算：

\[
G=H(S_{C=HIS-P})-H(S_{C=HIS-S}),\qquad
R=N\tanh[(G/|C|-g_0)/s_g].
\]

两态同时切换 C 内全部 His，其他编码 token 完全相同。R 对应 `core_release`；
G 为正表示该固定结构上质子化态代价更高，符合低 pH destabilization 的设计方向。
默认 g0=0、sg=1 是工程缩放，不是 pKa 或结合自由能标定。
该读数未比较结合与游离态的完整自由能，不能直接解释为 pH 6.0/7.4 的 Kd 比。

`core_unpaired` 是深埋 His 环缺少氢键的数量：环平均深度至少 4 Å，且不存在
强于 -0.5 的环供体或受体氢键（包括与主链的氢键）。它不是全蛋白 buried-unsat 总数。
零 His 时三项均为 0；有孤立 His 时仍计该罚项。没有网络时 N、R 为 0，
原始 G 记 null/status=no_network，不请求模型，不把缺失能量伪造为零。
后端失败、编号/序列不一致或非有限能量直接报错。

示例每轮最大化：

\[
F=Q+0.2N+0.1R-0.1U,
\]

Q 沿用示例宿主的结构质量目标。新 loss 通过标准 loss registry 和
`weighted_score_expression` 进入每轮综合评分，沿用 Astrolabe 现有接受规则；
启用 Metropolis 时也使用同一个 F。没有初始 His/network gate。
上述权重尚需首轮校准；评分接通不等于已得到实验酸性释放 binder。

## 运行与审计

mutator 的 `runtime_key: auto_ph_core` 与释放 loss 的 `runner_key` 相同，
共享一个隔离 Proton daemon/模型；未显式设置 runtime_key 的旧 mutator 保持独立。
独立调用 loss 时须先注册共享 runner，或通过 `runner_config` 提供配置。
不安装新环境，不修改模型 checkpoint。

每次 loss 写出 `ph_core_network_audit` JSON 日志，并在 eval_result 上保留各 metric 记录，
包括源结构 hash、apo 路径与编号映射、所有网络边、选中 C、N/U、原始 G、两态能量
及 Proton raw 输出路径。标准 score 结果保留三个 alias 的标量。
缓存键包含结构内容与评分参数；不同候选不共享两态上下文或动态中心。

本地模型扩展补丁位于
`patches/vendor/proton_pottsmpnn/02_automatic_histidine_bias.patch`；
固定上游版本及补丁记录见 `vendor/proton_pottsmpnn/SOURCE_PROVENANCE.json`。
