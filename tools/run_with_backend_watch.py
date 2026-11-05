"""Stop an owned evaluation command if a required local backend exits."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BACKEND_EXIT = 75


def alive(pid):
    try:
        # kill(0) also reports zombies as alive.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # Also reap descendants if the launcher exits before its children.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-pids", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pids = [int(pid) for pid in args.backend_pids.split(",")]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda sig, frame: sys.exit(128 + sig))
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            exited = [pid for pid in pids if not alive(pid)]
            if exited:
                print(f"[ERROR] Required vLLM backend exited: {exited}; stopping evaluation. See worker_resources.jsonl and vLLM logs.", file=sys.stderr, flush=True)
                return BACKEND_EXIT
            time.sleep(1)
        return process.returncode if process.returncode >= 0 else 128 - process.returncode
    finally:
        stop(process)


if __name__ == "__main__":
    sys.exit(main())
