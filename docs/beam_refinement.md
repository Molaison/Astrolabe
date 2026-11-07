# 通用 Beam refinement

通过 `beam_refinement` 显式选择 `branching/native_refinement`；`base.yaml`
保留日常轨迹入口。示例的宽度、预算和并行参数是演示值，不是最优值。

```bash
astrolabe run -t beam_refinement --dry-run env=env
astrolabe run -t beam_refinement env=env
```

示例使用 46 aa、seed 17、ESMFold 和 MCMC proposal。评估预算 3 包含 root，
因此最多评分 root 和两个子候选。`search.max_evaluations` 优先于 `steps`；
后者只在未给出显式评估预算时作为 root-inclusive 上限，不乘 Beam 宽度。

`search.execution.mode: isolated` 启用常驻预测与 CPU Score 进程；
`workers` 限制进程数，`in_flight` 限制包含 Score 的在途候选数。
`model_sharing: shared` 让 worker 附着到同一个兼容的模型 owner。
`execution.mode: resident_serial` 保持模型请求串行，同时允许准备、评分和
proposal 供给重叠。CPU worker 数不代表模型推理并发数。

父进程按稳定顺序生成 proposal 并提交结果，每代评分完成后再选择下一代父代。
候选不会因 worker 先完成而优先入选。`prediction_rng: fixed_request` 是并行
模式的显式科学合同；未配置执行模式时，旧串行 `continuation` 随机流仍保留。
两种随机流不能直接宣称逐位相同。

raw、Score 与成员分别提交。完整 raw 可以恢复 CPU Score；已发出但缺少可信
完成证据的预测保留为 unknown。请求、数据或 sidecar 不一致时拒绝恢复。
运行输出包括 `01_evaluations.jsonl`、`03_final_beam.json`、`.framework`
和 `.prediction_workers/receipts`。

恢复时保留原任务、seed、评分和搜索配置，设置 `search.resume=true`，并将
`out_dir` 指向上述文件所在的实际 run 目录，不是首次启动时的输出父目录。
入口要求恰好一个 repeat；完整 Score 回执直接复用，不再次推理或评分。

当前入口限定单个静态 stage；Metropolis 接受、stage transaction、结构初始化
及模型初始化器的搜索恢复不在此入口的已支持范围。真实后端的数值、恢复和吞吐
资格分别验收。合同和对应检查见
[Evaluator 与 Score 执行合同](development/08_evaluator_score_execution.md)。
