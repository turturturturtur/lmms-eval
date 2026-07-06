# lmms-eval 运行脚本使用文档

本目录（`run_scripts/`）提供了一套用于 **本地单机调试** 和 **DLC 集群大规模提交** 的运行脚本，支持 `lmms-eval` 与 `vLLM` / `SGLang` 等推理后端的无缝集成。

> **lmms-eval** 是一个统一的多模态大模型评测框架，支持 100+ 评测任务、30+ 模型后端，覆盖图像、视频、音频等多模态场景。本目录的脚本主要面向生产环境的大规模分布式评测需求。

---

## 目录

1. [文件结构](#文件结构)
2. [核心设计思想](#核心设计思想)
3. [配置文件详解](#配置文件详解)
4. [本地单机运行](#本地单机运行)
5. [DLC 集群提交](#dlc-集群提交)
6. [Judge 评测](#judge-评测)
7. [适配其他模型](#适配其他模型)
8. [执行流程与数据流](#执行流程与数据流)
9. [日志与输出](#日志与输出)
10. [常见问题与排查](#常见问题与排查)
11. [与其他入口的关系](#与其他入口的关系)

---

## 文件结构

```
run_scripts/
├── README.md               # 本文档
├── eval_common.sh          # 共享 Bash 函数库（配置解析、启动 vLLM、执行 eval、清理资源）
├── qwen3_vl_worker.sh      # Worker 入口：启动 vLLM 后端并执行 lmms-eval
├── qwen3_vl_submit.sh      # Submitter 入口：读取 DLC 配置 + eval 配置，提交 DLC PyTorchJob
├── run_judge.sh            # Judge 入口：对 eval 结果进行打分/评判
├── config_eval.json        # 评测配置示例（环境 / 日志 / 分布式 / 模型 / 评测参数）
├── config_dlc.json         # DLC 集群调度配置示例
└── config_judge.json       # Judge 配置示例（API / vLLM 后端、任务列表、输入输出路径）
```

> **注意**：`qwen3_vl_*.sh` 当前以 Qwen3-VL 命名，但其底层逻辑（`eval_common.sh`）是通用的。你可以通过复制和微调快速适配 `vllm`、`sglang` 或其他模型。

---

## 核心设计思想

### 1. 职责分离（Submitter vs Worker）

旧版脚本将 **"DLC 提交"** 和 **"Worker 执行"** 混在一起，通过 `dlc.submit=true/false` 在同一份脚本里做模式切换，导致：
- 控制流晦涩难懂（递归自调用）
- DLC `--command` 是一团嵌套转义的长字符串，极易写错引号
- 本地调试和集群运行行为不一致

现在的设计把两者彻底拆成两个独立脚本，并且把配置也拆成两份：

| 脚本 | 运行位置 | 职责 |
|------|---------|------|
| `qwen3_vl_submit.sh` | 本地机器 | 读取配置，生成 runtime config，调用 `dlc submit` |
| `qwen3_vl_worker.sh` | 本地 / DLC Worker 容器 | 启动 vLLM 后端，等待就绪，运行 lmms-eval，退出清理 |
| `eval_common.sh` | 被 source | 提供可复用的 Bash 函数库 |

### 2. 配置拆分（`config_dlc.json` + `config_eval.json`）

- **`config_eval.json`**：包含模型、任务、环境、分布式参数。本地调试时只需要这一份配置。
- **`config_dlc.json`**：仅包含集群调度参数（worker 数量、镜像、资源 ID 等）。

**好处**：
- 本地调试时完全不需要关心 DLC 参数；
- 切换集群资源时只改 `config_dlc.json`，不动 `config_eval.json`；
- 同一份 `config_eval.json` 可搭配不同的 `config_dlc.json`（开发 / 生产 / 多 region）。

---

## 配置文件详解

### `config_eval.json` — 评测运行配置

```json
{
  "env": {
    "hf_home": "/mnt/cpfsB/public_data/public_dataset/.cache/huggingface",
    "hf_token": "",
    "venv_path": "/mnt/cpfsB/<USER>/lmms-eval/.venv",
    "lmms_eval_datasets_cache": "/mnt/cpfsB/public_data/public_dataset/.cache/huggingface/datasets",
    "hf_datasets_offline": true,
    "transformers_offline": true,
    "api_type": "openai",
    "openai_api_key": "sk-...",
    "openai_api_url": "https://yunwu.ai/v1/chat/completions"
  },
  "log": {
    "dir": "/mnt/cpfsB/<USER>/vllm_logs"
  },
  "distributed": {
    "master_addr": "127.0.0.1",
    "master_port": 23456,
    "world_size": 1,
    "rank": 0
  },
  "model": {
    "path": "/mnt/cpfsB/<USER>/data/model/Qwen3-VL-8B-Instruct",
    "tp": 1,
    "max_model_len": 65536,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 1024,
    "base_port": 8001
  },
  "eval": {
    "tasks": "ai2d,mmmu_val,mmbench_en_dev",
    "output_path": "/mnt/cpfsB/<USER>/lmms-eval/eval_result",
    "concurrency": 128,
    "gen_kwargs": "max_new_tokens=32768,max_pixels=4014080,temperature=1.0",
    "limit": -1,
    "task_timeout_seconds": 43200,
    "task_timeout_kill_after_seconds": 300,
    "debug": false
  }
}
```

#### 字段说明

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `env.hf_home` | string | HuggingFace 缓存根目录 |
| `env.hf_token` | string | HuggingFace Token（可选） |
| `env.venv_path` | string | Python 虚拟环境路径 |
| `env.lmms_eval_datasets_cache` | string | 显式 datasets cache；生产配置指向 CPFSB public 持久缓存 |
| `env.hf_datasets_offline` | bool | 是否离线加载 datasets |
| `env.transformers_offline` | bool | 是否离线加载 transformers |
| `env.api_type` | string | Judge API 类型（如 openai） |
| `env.openai_api_key` | string | Judge API Key（可选） |
| `env.openai_api_url` | string | Judge API 端点地址（可选） |
| `log.dir` | string | 日志根目录 |
| `distributed.master_addr` | string | 分布式主节点地址 |
| `distributed.master_port` | int | 分布式主节点端口 |
| `distributed.world_size` | int | 总节点数（DLC 语义下为机器数） |
| `distributed.rank` | int | 当前节点 rank |
| `model.path` | string | 模型路径或 HuggingFace model_id |
| `model.tp` | int | Tensor Parallelism 大小（每张卡上的 GPU 数 / TP = backend 数量） |
| `model.max_model_len` | int | vLLM `max-model-len` |
| `model.gpu_memory_utilization` | float | vLLM GPU 内存利用率 |
| `model.max_num_seqs` | int | vLLM 最大并发序列数 |
| `model.base_port` | int | vLLM backend 起始端口 |
| `eval.tasks` | string | 逗号分隔的评测任务列表 |
| `eval.output_path` | string | 评测结果输出根目录 |
| `eval.concurrency` | int | lmms-eval 对 vLLM backend 的并发请求数 |
| `eval.gen_kwargs` | string | 生成参数，逗号分隔键值对 |
| `eval.limit` | int | 每个任务最大样本数（`-1` 表示无限制，仅调试使用） |
| `eval.task_timeout_seconds` | int | 单个 lmms-eval task 的硬超时时间，必填正整数 |
| `eval.task_timeout_kill_after_seconds` | int | task 超时后 SIGTERM 到 SIGKILL 的等待时间，必填正整数 |
| `eval.debug` | bool | 调试模式（`true` 时退出不杀死 vLLM 进程） |

### `config_dlc.json` — DLC 集群调度配置

```json
{
  "dlc": {
    "submit": true,
    "job_name": "test_eval_jobs",
    "binary": "/mnt/cpfsB/<USER>/dlc",
    "run_script": "/mnt/cpfsB/<USER>/lmms-eval/scripts/qwen3_vl_worker.sh",
    "workers": 2,
    "worker_gpu": 8,
    "worker_cpu": 110,
    "worker_memory": "1500Gi",
    "worker_shared_memory": "1500Gi",
    "priority": 9,
    "running_timeout": 86400,
    "worker_image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/...",
    "data_source_uris": "cpfs://...::/mnt/cpfsB,nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB,oss://...::/mnt/oss",
    "resource_id": "quotaev2tl4w6aw0",
    "workspace_id": "245264",
    "vpc_id": "vpc-...",
    "switch_id": "vsw-...",
    "security_group_id": "sg-...",
    "extended_cidrs": "10.1.255.0/29,10.1.255.8/29"
  }
}
```

#### 字段说明

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `dlc.binary` | string | DLC CLI 可执行文件路径 |
| `dlc.job_name` | string | 作业名称（可选，默认自动生成） |
| `dlc.workers` | int | Worker 节点数量 |
| `dlc.worker_gpu` | int | 每个 Worker 的 GPU 数量 |
| `dlc.worker_cpu` | int | 每个 Worker 的 CPU 核数 |
| `dlc.worker_memory` | string | 每个 Worker 的内存配额 |
| `dlc.worker_shared_memory` | string | 每个 Worker 的共享内存配额 |
| `dlc.priority` | int | 作业优先级 |
| `dlc.running_timeout` | int | 运行超时时间（秒） |
| `dlc.worker_image` | string | Worker 容器镜像 |
| `dlc.data_source_uris` | string | 数据挂载 URI（CPFS / NAS / OSS 等），必须包含 `nas://292a8d49e93-kgi71.cn-wulanchabu.nas.aliyuncs.com/::/mnt/nasB` |
| `dlc.resource_id` | string | 固定资源配额 ID，必须为 `quotaev2tl4w6aw0` |
| `dlc.workspace_id` | string | 工作空间 ID |
| `dlc.vpc_id` / `switch_id` / `security_group_id` | string | 网络与安全组配置 |
| `dlc.extended_cidrs` | string | 扩展 CIDR 列表 |

### `config_judge.json` — Judge 评判配置

```json
{
  "env": {
    "hf_home": "/mnt/cpfsB/public_data/public_dataset/.cache/huggingface",
    "hf_token": "",
    "venv_path": "/mnt/cpfsB/<USER>/lmms-eval/.venv"
  },
  "log": {
    "dir": "/mnt/cpfsB/<USER>/judge_logs"
  },
  "judge": {
    "backend": "api",
    "parallel": 128,
    "model": "gpt-4o-mini",
    "api": {
      "key": "",
      "base_url": ""
    },
    "vllm": {
      "model_path": "/mnt/cpfsB/public_data/public_model/Qwen3.5/Qwen3.5-27B",
      "tp": 4,
      "max_model_len": 32768,
      "gpu_memory_utilization": "0.9",
      "max_num_seqs": 512,
      "port": 8002
    }
  },
  "eval": {
    "input_result_path": "/mnt/cpfsB/<USER>/lmms-eval/eval_result/2026-04-12_18-59-25/Qwen3-VL-8B-Instruct",
    "tasks": "mmbench_en_dev,MolParse,sfe-en,mmmu_val_qwen3_official",
    "output_path": "/mnt/cpfsB/<USER>/judge_results",
    "verbosity": "INFO",
    "debug": true
  }
}
```

#### 字段说明

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `judge.backend` | string | Judge 后端类型：`api`（调用远程 API）或 `vllm`（本地启动 vLLM） |
| `judge.parallel` | int | 并行 judge 的 worker 数量 |
| `judge.model` | string | Judge 模型名称（API 后端时使用） |
| `judge.api.key` | string | API Key（可选，优先从环境变量读取） |
| `judge.api.base_url` | string | API Base URL（可选，优先从环境变量读取） |
| `judge.vllm.model_path` | string | vLLM 后端模型路径（`backend=vllm` 时生效） |
| `judge.vllm.tp` | int | vLLM Tensor Parallelism 大小 |
| `judge.vllm.max_model_len` | int | vLLM `max-model-len` |
| `judge.vllm.gpu_memory_utilization` | string | vLLM GPU 内存利用率 |
| `judge.vllm.max_num_seqs` | int | vLLM 最大并发序列数 |
| `judge.vllm.port` | int | vLLM 服务端口 |
| `eval.input_result_path` | string | 待 judge 的 eval 结果目录或单个 `*samples_*.jsonl` 文件 |
| `eval.tasks` | string | 需要 judge 的任务列表，逗号分隔 |
| `eval.output_path` | string | Judge 结果输出目录 |
| `eval.verbosity` | string | 日志级别（`INFO` / `DEBUG` 等） |
| `eval.debug` | bool | Debug 模式（`true` 时不自动清理 vLLM 进程） |

---

## 本地单机运行

### 基础用法

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json
```

### 覆盖模型路径

```bash
bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json /path/to/another/model
```

### 强制开启数据集缓存 staging（默认本地不开启）

本地运行时**不会**自动 staging 数据集缓存（避免不必要的 100G+ 数据拷贝）。如果你确实需要：

```bash
LMMS_EVAL_STAGE_DATASETS=1 bash run_scripts/qwen3_vl_worker.sh run_scripts/config_eval.json
```

### `eval_common.sh` 执行流程（本地 / Worker 通用）

```
source eval_common.sh
      │
      ▼
load_config()        ← 读取 eval config，导出 HF_HOME / HF_TOKEN / offline flags
      │
      ▼
compute_resources()  ← 探测本地 GPU 数，计算 MACHINE_RANK、NUM_BACKENDS
      │
      ▼
setup_logging()      ← 确定 LOG_DIR（若 LMMS_EVAL_LOG_DIR 已设置则优先使用）
      │
      ▼
ensure_venv()        ← 激活 .venv
      │
      ▼
setup_cleanup_trap() ← 注册 EXIT/INT/TERM 时的 vLLM 清理函数
      │
      ▼
launch_vllm_backends() → 启动 NUM_BACKENDS 个 vLLM 进程
      │
      ├─── wait_for_backends() ───┐
      │      轮询 health check     │
      │                            │
      ├─── stage_datasets() ───────┤（仅在 LMMS_EVAL_STAGE_DATASETS=1 时后台执行）
      │      并行拷贝数据集缓存      │
      │                            │
      └──── wait 数据集拷贝完成 ◄──┘
             │
             ▼
      run_lmms_eval()  ← torchrun -m lmms_eval ...
             │
             ▼
      cleanup_vllm()   ← 杀死所有 backend 进程（debug=true 时跳过）
```

---

## DLC 集群提交

### 提交命令

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/qwen3_vl_submit.sh run_scripts/config_dlc.json run_scripts/config_eval.json
```

### Submitter 执行流程

```
读取 config_dlc.json + config_eval.json
      │
      ▼
生成 runtime_config.json
   （基于 eval_config，强制 dlc.submit=false, eval.debug=false，并注入统一时间戳）
      │
      ▼
构建 DLC command:
   export LMMS_EVAL_LOG_DIR=<FIXED_LOG_DIR>;
   export LMMS_EVAL_STAGE_DATASETS=1;
   bash qwen3_vl_worker.sh <runtime_config.json>
      │
      ▼
dlc submit pytorchjob --command="..."
```

### 关键行为

- **统一时间戳**：由 submitter 生成并写入 `runtime_config.json`，确保所有 worker 的输出目录一致。
- **强制非调试**：集群运行自动设置 `debug=false`，退出时会清理 vLLM 进程。
- **自动 staging**：DLC 提交自动设置 `LMMS_EVAL_STAGE_DATASETS=1`，worker 启动时会并行拷贝数据集缓存。

---

## Judge 评测

部分任务（如 `mmbench_en_dev`、`mmmu_val` 等）在模型生成答案后，还需要一个 **Judge 模型** 对答案进行打分或评判。`run_judge.sh` + `config_judge.json` 提供了独立的 judge 流程。

### 运行 Judge

```bash
cd /mnt/cpfsB/<USER>/lmms-eval
bash run_scripts/run_judge.sh run_scripts/config_judge.json
```

若不传参数，默认使用 `run_scripts/config_judge.json`。

### Judge 后端选择

#### 1. API 后端（推荐，无需本地 GPU）

在 `config_judge.json` 中设置：

```json
{
  "judge": {
    "backend": "api",
    "model": "gpt-4o-mini",
    "api": {
      "key": "sk-...",
      "base_url": "https://yunwu.ai/v1"
    }
  }
}
```

> **安全建议**：`key` 和 `base_url` 建议通过环境变量注入（`OPENAI_API_KEY`、`OPENAI_API_BASE` 或 `JUDGE_API_KEY`、`JUDGE_BASE_URL`），而不是写死在配置文件中。

#### 2. vLLM 后端（本地部署 Judge 模型）

```json
{
  "judge": {
    "backend": "vllm",
    "vllm": {
      "model_path": "/mnt/cpfsB/public_data/public_model/Qwen3.5/Qwen3.5-27B",
      "tp": 4,
      "port": 8002
    }
  }
}
```

`run_judge.sh` 会自动：
1. 启动本地 vLLM 服务作为 judge 后端
2. 等待服务就绪
3. 调用 `python -m lmms_eval judge` 执行评判
4. 退出时自动停止 vLLM 进程（`debug=false` 时）

### Judge 输入输出

- **输入**：`eval.input_result_path` 指向某个 eval 输出目录，脚本会自动查找该目录下与 `eval.tasks` 对应的 `*samples_<task>.jsonl` 文件。
- **输出**：Judge 结果会写入 `eval.output_path` 目录。
- **日志**：`${log.dir}/judge_<timestamp>/judge.log` 和 `vllm_judge_backend.log`（若使用 vLLM 后端）。

---

## 适配其他模型

当前脚本以 Qwen3-VL 为例，但底层逻辑通用。若要为其他模型（如 Qwen2.5-VL、InternVL、LLaVA 等）创建新的运行脚本，推荐做法：

### 1. 复制并修改 Worker / Submitter 脚本

```bash
cp run_scripts/qwen3_vl_worker.sh run_scripts/my_model_worker.sh
cp run_scripts/qwen3_vl_submit.sh run_scripts/my_model_submit.sh
```

修改脚本中的引用路径即可，核心逻辑（`eval_common.sh`）无需改动。

### 2. 创建对应的 `config_eval.json`

主要调整 `model` 和 `eval` 字段：

```json
{
  "model": {
    "path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "tp": 2,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 1024,
    "base_port": 8001
  },
  "eval": {
    "tasks": "mme,mmmu_val",
    "concurrency": 128,
    "gen_kwargs": "max_new_tokens=4096,temperature=0.0",
    "task_timeout_seconds": 43200,
    "task_timeout_kill_after_seconds": 300
  }
}
```

### 3. 调整 `run_lmms_eval()` 中的模型参数（如需要）

如果模型需要特殊的 `model_args`（例如 `is_qwen3_vl=True` 这类硬编码参数），可以在 `eval_common.sh` 的 `run_lmms_eval()` 函数中修改 `--model_args` 一行。建议为新模型单独 fork 一个 `eval_common_my_model.sh` 以避免互相影响。

### 4. 使用 sglang 或其他后端

`eval_common.sh` 当前启动的是 `vllm.entrypoints.openai.api_server`。若要用 `sglang`：

- 修改 `launch_vllm_backends()` 函数，将启动命令替换为 `python -m sglang.launch_server ...`
- 修改 `run_lmms_eval()` 中 `--model` 参数为 `sglang`
- 调整 health check URL 和 cleanup 目标进程名

---

## 执行流程与数据流

### 多机分布式

`eval_common.sh` 使用 `torchrun` 而非 `accelerate launch` 来启动 lmms-eval：

```bash
torchrun \
    --nnodes="${NUM_MACHINES}" \
    --node_rank="${MACHINE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m lmms_eval ...
```

原因：DLC PyTorchJob 设置 `WORLD_SIZE` / `RANK` 为节点级信息，但 `accelerate launch` 会被这些环境变量干扰，只能启动单进程。`torchrun` 能正确处理多节点多进程，且 lmms-eval 会自动检测到 `torch.distributed.is_initialized()` 并将 `distributed_executor_backend` 设为 `torchrun`。

### vLLM Backend 启动策略

`eval_common.sh` 根据本地 GPU 数量和 `model.tp` 计算需要启动的 backend 数量：

```
NUM_BACKENDS = LOCAL_GPU_NUM / MODEL_TP
```

例如：单机 8 卡，`tp=2`，则启动 4 个 vLLM backend，分别占用 GPU `[0,1]`、`[2,3]`、`[4,5]`、`[6,7]`，端口为 `base_port` ~ `base_port+3`。

lmms-eval 的 `vllm_backend` 模型会通过 `base_url` 中的分号分隔 URL 列表实现负载均衡。

---

## 日志与输出

### 本地运行

- **vLLM 日志**：`${LOG_BASE}/<timestamp>/vllm_model_rank${RANK}_port${PORT}.log`
- **评测日志**：`${LOG_BASE}/<timestamp>/lmms_eval_rank${RANK}.log`
- **评测结果**：`${eval.output_path}/<timestamp>/`

### DLC 集群运行

- **统一日志目录**：`${log.dir}/${JOB_NAME}/${TIMESTAMP}/`
- **vLLM 日志**：`.../vllm_model_rank${RANK}_port${PORT}.log`
- **评测日志**：`.../lmms_eval_rank${RANK}.log`
- **runtime config**：`.../runtime_config.json`
- **评测结果**：`${eval.output_path}/${TIMESTAMP}/`

### 环境变量覆盖

| 环境变量 | 作用 |
|---------|------|
| `LMMS_EVAL_LOG_DIR` | 强制指定日志输出目录 |
| `LMMS_EVAL_STAGE_DATASETS` | 设置为 `1` 时启用数据集缓存 staging |
| `MASTER_ADDR` / `MASTER_PORT` | 覆盖分布式主节点地址和端口 |
| `WORLD_SIZE` / `RANK` | 覆盖分布式规模（DLC 自动注入） |

---

## 常见问题与排查

### 1. 不要直接执行 `eval_common.sh`

它是被 `source` 的函数库，直接运行会报错：

```
[ERROR] eval_common.sh should be sourced, not executed directly.
```

### 2. `jq not found`

脚本会自动尝试 `apt-get install jq`，若容器无 apt 权限则需提前安装：

```bash
conda install -c conda-forge jq
# 或
pip install jq
```

### 3. vLLM backend 启动超时

`wait_for_backends()` 最长等待 30 分钟（360 次 × 5 秒）。超时通常因为：
- 模型文件路径错误或权限不足
- GPU 显存不足（尝试减小 `max_model_len` 或 `gpu_memory_utilization`）
- 端口冲突（修改 `base_port`）

### 4. 数据集 staging 导致启动慢

首次 DLC 运行时，`stage_datasets()` 会并行拷贝大量数据。可检查 `vllm_logs` 中是否有拷贝进度。后续运行若缓存已存在则会快很多。

### 5. 本地调试和 DLC 结果不一致

请确保两者使用相同的：
- `config_eval.json`（除 `debug` 外）
- 模型权重版本
- 虚拟环境（`.venv`）
- `gen_kwargs` 生成参数

### 6. 如何只跑部分任务调试？

修改 `config_eval.json` 中 `eval.tasks` 为单个或少量任务，并设置 `eval.limit`（如 8、16）：

```json
{
  "eval": {
    "tasks": "mme",
    "limit": 8,
    "debug": true
  }
}
```

> **注意**：`limit` 仅用于调试，正式评测请设为 `-1`。

---

## 与其他入口的关系

本目录的脚本是对 lmms-eval 上层能力的封装。项目还提供多种使用方式：

| 入口 | 适用场景 | 示例 |
|------|---------|------|
| `python -m lmms_eval` | 快速命令行评测 | `python -m lmms_eval --model qwen2_5_vl --tasks mme` |
| `lmms-eval eval` | 结构化子命令 | `lmms-eval eval --config configs/example_local.yaml` |
| `lmms-eval ui` | Web UI 交互式配置 | `uv run lmms-eval-ui` |
| `lmms-eval serve` | HTTP 远程评测服务 | `lmms-eval serve --host 0.0.0.0 --port 8000` |
| `lmms-eval tui` | 终端交互式 UI | `lmms-eval tui` |
| `run_scripts/` | 大规模分布式 / 生产环境 | 本文档 |

### 相关文档

- [项目 README](../README.md)
- [快速开始](../docs/getting-started/quickstart.md)
- [命令行参数](../docs/getting-started/commands.md)
- [模型接入指南](../docs/guides/model_guide.md)
- [任务配置指南](../docs/guides/task_guide.md)
- [现有任务列表](../docs/advanced/current_tasks.md)

---

## 附录：向后兼容说明

如果你之前使用过旧版单配置文件（同时包含 `dlc` 和 `eval` 字段，通过 `dlc.submit=true/false` 切换），可以继续通过旧脚本运行，但建议尽快迁移到新的双配置模式：

- **旧版单配置**：脚本内部含 `dlc.submit=true`，运行时会打印 deprecation 警告并降级为单配置提交模式。
- **新版推荐**：`config_dlc.json` + `config_eval.json`，职责清晰，本地与集群行为一致。

---

如有更多问题，请参考 [lmms-eval GitHub Issues](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues) 或联系维护者。
