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
- Login validation uses `dlc get job --workspace_id 240810 --page_size 1 --page_num 1` with `--access_id`, `--access_key`, `--ignore_local_config`, `--region=cn-wulanchabu`, and `--endpoint=pai-dlc.cn-wulanchabu.aliyuncs.com`.
- The browser receives only an HttpOnly session cookie. DLC submissions use the logged-in user's AK/ASK from the server session.
- `role=admin` is returned by `/auth/me` only when the validated AccessKey ID appears in `admins`. Admin-only operations are reserved for future expansion.
- Cross-origin frontend development is disabled by default. Set `LMMS_EVAL_WEBUI_ALLOWED_ORIGINS=http://localhost:5173` if you run Vite separately.
- `local/webui_users.json` is ignored by git. It must not contain `secret_access_key`.

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
