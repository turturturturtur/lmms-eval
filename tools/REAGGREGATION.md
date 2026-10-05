# 保存样本的离线重汇总

这两个工具不重新推理、不调用 judge API，也不覆盖已有结果。使用评测环境的 Python，从仓库根目录运行。

## SmolInstruct

```bash
python -m tools.reaggregate_smol_samples --output corrected.json /absolute/path/samples*.jsonl
```

支持 `forward_synthesis`、`retrosynthesis`、`molecule_generation`、`name_conversion-i2s` 的保存样本，每个输入文件只能包含一个任务。需要 `tools/requirements-smol-reaggregation.txt` 的补充依赖和项目基础依赖。

输出保留当前 benchmark 的有效预测条件下 Morgan 相似度口径，并同时报告空答案、非法分子和准确匹配数。所有预测均非法时相似度为 0；程序错误/worker 失败直接报错，不发布伪造的零分。反应物中的分号在预测和标签两侧都规范为点号。输出记录源码、输入文件、样本身份 SHA256 和 RDKit 版本。

## MMBench

```bash
python -m tools.reaggregate_mmbench_samples --output comparable.json /absolute/run1/samples_mmbench.jsonl /absolute/run2/samples_mmbench.jsonl
```

输入须保留 `gpt_eval_score` 结构。每题所有 cyclic 变体均答对才计正确，使用本仓库的静态答案提取器；这不是新运行的 LLM judge 分数。先验证样本 ID、问题、选项、答案集合一致，再输出各模型的可比分数和源文件/代码 SHA256。不检查图像字节身份，不用于比较不同数据版本。已有输出、缺少基准变体或样本集合不同都会报错。

2026-10-05 的审计从历史预测重新汇总了 13,704 条 Smol 样本，以及 7 组各 4,329 条 MMBench 变体（1,164 个问题）；证据在相邻训练仓库 `develop/weekly_claims_repair_20261005/evidence/`。
