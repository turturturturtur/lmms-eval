# LMMS-Eval Web UI

Web-based User Interface for LMMS-Eval built with React + Vite + Tailwind CSS and FastAPI.

## Architecture

- **Backend**: FastAPI (Python) - handles model/task discovery, evaluation runs
- **Frontend**: React + Vite + Tailwind CSS - modern web UI

## Requirements

- Python 3.10+
- Node.js 18+ (for building the frontend)

## Installation

The web UI will be automatically built on first run. Or build manually:

```bash
cd lmms_eval/tui/web
npm install
npm run build
```

## Usage

### Quick Start

```bash
uv run lmms-eval-ui
```

This starts the server on http://localhost:8000 and opens your browser.

### Manual Startup

```bash
# Start server only
uv run uvicorn lmms_eval.tui.server:app --host 0.0.0.0 --port 8000

# Then open http://localhost:8000 in your browser
```

### Custom Port

```bash
LMMS_SERVER_PORT=3000 uv run lmms-eval-ui
```

## Local Authentication

The Web UI requires AK/ASK login before any evaluation, DLC, task, or log API can be used. AK/ASK values are validated in real time by calling a read-only DLC API with the submitted credentials. The local plaintext file is only an admin allowlist:

```text
lmms-eval/local/webui_users.json
```

Create it from the example file:

```bash
cp local/webui_users.json.example local/webui_users.json
```

Admin file format:

```json
{
  "admins": [
    {
      "username": "admin",
      "display_name": "WebUI Admin",
      "access_key_id": "admin-access-key-id"
    }
  ]
}
```

Notes:

- The default session lifetime is 15 days. Override with `LMMS_EVAL_WEBUI_SESSION_TTL_SECONDS`.
- Override the user file path with `LMMS_EVAL_WEBUI_AUTH_FILE`.
- Login validation uses `dlc get job --workspace_id 247143 --page_size 1 --page_num 1` with `--access_id`, `--access_key`, `--ignore_local_config`, `--region=cn-wulanchabu`, and `--endpoint=pai-dlc.cn-wulanchabu.aliyuncs.com`.
- The browser receives only an HttpOnly session cookie. DLC submissions, queries, quota requests, and cancellation use the logged-in user's AK/ASK. Query caches are scoped to that identity.
- `role=admin` is returned by `/auth/me` only when the validated AccessKey ID appears in `admins`. Admin-only operations are reserved for future expansion.
- Vite proxies the WebUI API paths to port 8000 during development. If an external frontend calls the backend directly across origins, configure `LMMS_EVAL_WEBUI_ALLOWED_ORIGINS`.
- `local/webui_users.json` is ignored by git. It must not contain `secret_access_key`.

## DLC runtime and workflow

The WebUI supports DLC checkpoint evaluation through Qwen3.5/vLLM and remote
OpenAI-compatible API evaluation. Its default inference interpreter is the
repository `.venv`, independently of the Python running the WebUI. DLC requires
Python 3.11 or newer. Workers use `PYTHONSAFEPATH` so `python -m`
imports the saved source snapshot even when its working directory is the live
repository. Relative input-data paths still resolve from the original directory.
Prepare this runtime
with the workspace's `scripts/install_lmmeval_runtime.sh`. The submitter checks
the actual interpreter, installed vLLM dependency requirements, and imports before
creating a cloud job; workers repeat the check on the GPU machine. The inference
environment must not inherit incompatible CUDA extensions from another Conda
environment. The installer disables system site packages and installs its own
evaluation dependencies.

Start creates a background workflow immediately. Logs can be disconnected and
replayed without resubmission, and refreshing the same browser tab restores the
current log. Completed means the remote evaluation and any configured judge have
succeeded. Stop requests cancellation of the cloud jobs and reports stopping
until the remote state is confirmed. The View Logs page also supports cloud job
cancellation and refreshes the selected job's metrics when Sync DLC is clicked.
Generation results that require post-evaluation judging show `pending_judge`
until scored results arrive. Judge accuracy takes precedence over request-success
counters, and sample links remain attached to their source results as new files
arrive.

Judge job names depend on how they were submitted:

- API judging: `judge_<eval job name>_<YYYY-MM-DD_HH-MM-SS>` (sanitized and
  truncated to 120 characters; `dlc.judge.job_name` can override this).
- WebUI retry: `eval_judge_retry_<source DLC job ID>_<8 hex characters>`.
  This is a judge stage even though the submission prefix remains `eval_`.
- Inline local vLLM judging retains its parent `eval_*` job name.

Result discovery follows the exact `UserCommand` runtime config or explicit
`--output_path` / `--output-dir` / `--output_dir`. Eval output is
`eval.output_path/eval.timestamp`; a judge config writes directly into
`eval.output_path` without appending a timestamp. The standard judge directory
is `<eval result directory>/judge/`.

Each judge `*_results.json` contains `results[task]`, `n-samples[task]`, and
`judge` metadata (`model`, `mode`, `input_file`, `samples_file`, `failed_samples`).
The WebUI prefers native `overall_accuracy`, `accuracy`, or `exact_match` over
historical generic `total_acc` and operational counters. It preserves the
metric's original units. Native failure sentinels such as `accuracy=-1`,
non-finite scores, and failed judge requests display `judge_failed` without a
score. The Source column distinguishes Eval, Judge (with model), and Judge
samples only.

Sample links first use `judge.samples_file` (absolute or relative to the aggregate,
within its directory), then fall back to the matching timestamp/task filename.
Samples preserve the inference fields and add `metrics` and `judge_mode`;
correctness filtering uses these judge metrics before provisional eval scores.
When only judged JSONL exists, the sample link remains available and the value
shows **Awaiting aggregate**. The server does not present a per-row mean as a
completed native score: samples are also saved before failed aggregation, and
MMBench circular accuracy cannot be reconstructed by averaging individual hits.

Judge aggregation reads the final per-sample `metrics` rather than stale
generation fields. Native task scores take precedence over generic sample
means (including MMBench circular accuracy), and final scores are counted once.
An aggregation exception or invalid native score fails the job and retains the
judged JSONL for diagnosis without publishing a successful aggregate. Native
metric units are preserved: for example SimpleVQA/MMMU accuracy is a percentage,
while SFE `exact_match` is a fraction.

Raw model outputs are always retained in the sample JSONL `resps` field, even
when they equal `filtered_resps` or the answer matches successfully. After model
generation returns, the evaluator also writes `*_raw_<task>_rank<N>.jsonl` before
filtering or scoring. Failed/timed-out tasks retain raw files and sample files;
their aggregated score files are removed so they cannot look successful.
Outputs from requests that have not returned from the model are not covered by
this checkpoint.

The DLC worker runs benchmarks sequentially with `--log_samples`, saving each
under `tasks/<benchmark>/` before starting the next. A later failure does not
delete earlier results. View Logs discovers new result files even when a
summary file already exists.

For tasks requiring post-evaluation judging, Judge starts automatically after
all Eval tasks succeed: local vLLM runs in the same DLC worker after releasing
the evaluation GPUs, while API judging uses a subsequent CPU-only DLC job.
Until judge results exist, View Logs shows `pending_judge` without a score,
including for historical eval files without an execution-stage marker. Failed
judging also hides the score. Tasks using only rule-based scoring can display
their final score immediately after Eval.

If inference completed but scoring failed or was stopped, select the terminal
eval job in View Logs and click **Retry judge (8 GPUs)**. The source owner or an
admin can retry. The server validates every task's successful manifest and
complete sample IDs before submitting a new local Qwen3.5 judge run. It copies
inference artifacts into a separate result directory, records their source in
`inference_source.json`, and scores all required judge tasks without starting
inference backends. Original and partial judge results remain available on the
source job. The page shows submission errors and the new DLC job ID; refreshing
the page during submission reconnects to the existing workflow.

`run_scripts/config_judge.json` sets `judge.datapoint_timeout_seconds` (300),
`judge.request_timeout_seconds` (120), and `judge.datapoint_max_attempts` (3).
The request timeout must be shorter than the sample timeout. Valid MMMU `Z`
judgments are accepted as no matching option; exhausted extraction requests
and hard timeouts count as judge failures, make the workflow fail, and suppress
the incomplete score in View Logs.
Local vLLM also enforces the native MMMU/MMBench extractors' allowed output letters
(including their respective `Z`/`E` no-match options) through structured decoding. This prevents explanatory judge
responses from repeatedly failing the single-letter parser. Remote API payloads
retain their existing format, and malformed responses include a bounded excerpt
in the judge log for diagnosis.
MMBench native extraction uses the configured request timeout and at most three
requests; exhausted attempts fail aggregation instead of guessing an option.

Sample filtering has three slider positions: **All**, **Wrong only**, and
**Unknown**. Unknown includes unparseable choice answers, samples without a
decidable correctness value, judge errors, and malformed JSONL entries (with
the original line retained for inspection). Filtering happens before pagination.
For local Qwen3.5 inference, vLLM backends use all allocated GPUs and one HTTP
client process per worker dispatches evaluation requests. `eval.concurrency`
limits requests per worker; it is not multiplied by the GPU count. The default
per-task timeout is six hours to accommodate the full 23,609-sample MME-RealWorld
benchmark at native image resolution; successful tasks exit immediately.

Each submitted job stores a source snapshot and SHA-256 manifest under its log
directory, so editing the workspace does not change queued or running evaluation
code. The main worker output is persisted as `worker.log`.

Workers record container memory/OOM counters, process RSS, and GPU usage in
`worker_resources.jsonl` beside the task logs. A local backend exiting aborts the
active evaluation and remaining tasks; an unsuccessful inference stage does not
start the inline judge. Request preprocessing errors fail the task by default,
so broken inputs cannot silently become scored model answers.

Submission sessions are held in server memory: after a server restart, use View
Logs to inspect or cancel existing DLC jobs.

## Features

- Model selection from all available models
- Task selection with search/filter
- Real-time command preview
- Live evaluation output streaming
- Start/Stop evaluation controls
- Configuration: batch size, limit, device, verbosity
- Log Viewer: browse saved runs, metrics, and samples from `./logs/`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/auth/login` | POST | Login with AK/ASK from local user file |
| `/auth/me` | GET | Return the current logged-in user |
| `/auth/logout` | POST | Clear the current WebUI session |
| `/models` | GET | List available models |
| `/tasks` | GET | List available tasks |
| `/eval/preview` | POST | Generate command preview |
| `/eval/start` | POST | Start evaluation |
| `/eval/{job_id}` | GET | Read the current user's submission workflow status |
| `/eval/{job_id}/stream` | GET | Stream evaluation output (SSE) |
| `/eval/{job_id}/stop` | POST | Stop evaluation |
| `/logs/runs` | GET | List available evaluation runs under logs path |
| `/logs/runs/{run_id}/results` | GET | Load full results JSON for one run |
| `/logs/runs/{run_id}/samples/{task_name}` | GET | Paginated sample records for one task |

## File Structure

```
lmms_eval/tui/
├── __init__.py        # Python exports
├── cli.py             # lmms-eval-ui entry point
├── server.py          # FastAPI server
├── discovery.py       # Model/task discovery
├── README.md          # This file
└── web/               # React frontend
    ├── src/
    │   ├── App.tsx    # Main React component
    │   ├── main.tsx   # Entry point
    │   └── index.css  # Tailwind CSS
    ├── package.json
    ├── vite.config.ts
    └── dist/          # Built static files
```

## Development

For frontend development with hot reload:

```bash
# Terminal 1: Start backend server
uv run uvicorn lmms_eval.tui.server:app --port 8000

# Terminal 2: Start Vite dev server
cd lmms_eval/tui/web
npm run dev
```

Then open http://localhost:5173 (Vite proxies API requests to :8000)
