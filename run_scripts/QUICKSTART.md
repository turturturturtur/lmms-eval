# lmms-eval 运行脚本快速开始

## 你需要填什么

准备三个配置文件（都在 `run_scripts/` 目录下）：

### 1. `config_eval.json` — 评测参数（本地/集群通用）

重点改这几项：

```json
{
  "env": {
    "hf_home": "/path/to/huggingface/cache",
    "venv_path": "/mnt/cpfsB/<USER>/lmms-eval/.venv",
    "api_type": "openai",
    "openai_api_key": "sk-...",
    "openai_api_url": "https://yunwu.ai/v1/chat/completions"
  },
  "model": {
    "path": "/path/to/Qwen3-VL-8B-Instruct",
    "tp": 1,
    "max_model_len": 65536,
    "gpu_memory_utilization": 0.9,
    "base_port": 8001
  },
  "eval": {
    "tasks": "mme,mmmu_val",
    "output_path": "/mnt/cpfsB/<USER>/lmms-eval/eval_result",
    "concurrency": 128,
    "gen_kwargs": "max_new_tokens=32768,temperature=1.0",
    "limit": -1,
    "debug": false
  }
}
```

- **`model.path`**：模型权重路径或 HuggingFace model_id。
- **`model.tp`**：Tensor Parallelism。单机 8 卡想启动 4 个 backend，就填 `2`（8/2=4）。
- **`eval.tasks`**：逗号分隔的任务名。不确定有哪些任务？先跑 `python -m lmms_eval --tasks list` 查看。
- **`eval.limit`**：本地调试时可改成 `8` 或 `16`；**正式评测保持 `-1`**。
- **`eval.debug`**：`true` 表示退出时不杀 vLLM 进程，方便查问题；**集群提交会自动强制为 `false`**。

### 2. `config_dlc.json` — 仅提交集群时需要

重点改这几项：

```json
{
  "dlc": {
    "binary": "/mnt/cpfsB/<USER>/dlc",
    "job_name": "eval_qwen3vl_tp1",
    "workers": 2,
    "worker_gpu": 8,
    "worker_cpu": 110,
    "worker_memory": "1500Gi",
    "worker_shared_memory": "1500Gi",
    "worker_image": "your-registry/python:3.13.9-gpu-...",
    "data_source_uris": "cpfs://...::/mnt/cpfsB,nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB,oss://...::/mnt/oss",
    "resource_id": "quotaev2tl4w6aw0",
    "workspace_id": "your-workspace-id",
    "vpc_id": "vpc-...",
    "switch_id": "vsw-...",
    "security_group_id": "sg-...",
    "extended_cidrs": "10.1.255.0/29"
  }
}
```

- **`dlc.binary`**：DLC CLI 可执行文件路径。
- **`dlc.workers`** / **`worker_gpu`**：集群节点数和每节点 GPU 数。
- **`dlc.data_source_uris`**：必须包含 `nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB`，否则 submitter 会拒绝提交。
- **`resource_id`**：固定使用 `quotaev2tl4w6aw0`；**`workspace_id`** / **`vpc_id`** 等网络字段找你所在平台的管理员要。

### 3. `config_judge.json` — Judge 配置（对答案进行打分/评判）

重点改这几项：

```json
{
  "judge": {
    "backend": "api",
    "parallel": 128,
    "model": "gpt-4o-mini",
    "api": {
      "key": "",
      "base_url": ""
    }
  },
  "eval": {
    "input_result_path": "/mnt/cpfsB/<USER>/lmms-eval/eval_result/<timestamp>/Qwen3-VL-8B-Instruct",
    "tasks": "mmbench_en_dev,mmmu_val",
    "output_path": "/mnt/cpfsB/<USER>/judge_results"
  }
}
```

- **`judge.backend`**：`api`（调用远程 OpenAI 兼容 API）或 `vllm`（本地启动 vLLM 作为 judge 后端）。
- **`judge.model`**：API 后端使用的 judge 模型名称。
- **`judge.api.key` / `base_url`**：API 密钥和地址，建议通过环境变量注入而不是写死在配置里。
- **`eval.input_result_path`**：待 judge 的 eval 结果目录（包含 `*samples_*.jsonl` 文件）。
- **`eval.tasks`**：需要 judge 的任务列表，逗号分隔。

---

## 本地测试

### 运行评测

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json
```

**只想跑 8 条数据快速验证：**
先把 `config_eval.json` 里的 `eval.limit` 改成 `8`，`eval.debug` 改成 `true`，再执行上面的命令。

**想换另一个模型权重：**
```bash
bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json /path/to/another/model
```

**本地日志在哪：**
默认在 `/mnt/cpfsB/<USER>/vllm_logs/<timestamp>/`，包含 `vllm_*.log` 和 `lmms_eval_*.log`。

### 运行 Judge

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/run_judge.sh run_scripts/config_judge.json
```

**Judge 日志在哪：**
默认在 `/mnt/cpfsB/<USER>/judge_logs/judge_<timestamp>/`，包含 `judge.log` 和（若使用 vLLM 后端）`vllm_judge_backend.log`。

---

## 提交到 DLC 集群

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/qwen3_vl_submit.sh run_scripts/config_dlc.json run_scripts/config_eval.json
```

执行后你会看到：
1. 生成 `runtime_config.json`（注入统一时间戳，强制 `debug=false`）
2. 调用 `dlc submit pytorchjob`
3. 输出预期的日志目录路径

然后就可以去 DLC 控制台或命令行看任务状态了。

---

## 一条命令总结

| 场景 | 命令 |
|------|------|
| 本地调试 | `bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json` |
| 本地换模型 | `bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json <model_path>` |
| 提交集群 | `bash run_scripts/qwen3_vl_submit.sh run_scripts/config_dlc.json run_scripts/config_eval.json` |
| 本地 Judge | `bash run_scripts/run_judge.sh run_scripts/config_judge.json` |

---

## 常见卡点和检查项

- **端口冲突？** 改 `config_eval.json` 里的 `model.base_port`，或改 `config_judge.json` 里的 `judge.vllm.port`。
- **显存不够？** 改小 `model.max_model_len` 或 `model.gpu_memory_utilization`。
- **找不到任务名？** `python -m lmms_eval --tasks list | grep 关键字`。
- **数据集想离线跑？** 把 `env.hf_datasets_offline` 和 `env.transformers_offline` 都设为 `true`。
- **Judge API 不通？** 检查 `OPENAI_API_KEY` / `OPENAI_API_BASE` 环境变量是否正确设置。

更详细的说明请参考同目录下的 [README.md](README.md)。
