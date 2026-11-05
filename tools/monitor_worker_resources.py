"""Persist worker resource evidence without importing the model runtime."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def read_memory(directory):
    result = {}
    for name in ("memory.current", "memory.max", "memory.peak", "memory.events",
                 "memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes",
                 "memory.failcnt", "memory.oom_control", "memory.stat"):
        try:
            result[name] = (directory / name).read_text().strip()
        except OSError:
            pass
    return result


def memory_directories():
    # Container cgroup namespaces may expose the current group at the mount root.
    roots = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if not controllers or "memory" in controllers.split(","):
                roots.extend(root / path.lstrip("/") for root in list(roots[:2]))
    except OSError:
        pass
    return list(dict.fromkeys(roots))


def snapshot():
    result = {"time": datetime.now(timezone.utc).isoformat()}
    result["cgroup"] = {str(root): values for root in memory_directories()
                        if (values := read_memory(root))}
    processes = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            status = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
            processes.append({"pid": int(path.parent.name), "name": status["Name"].strip(),
                              "rss_kib": int(status.get("VmRSS", "0 kB").split()[0])})
        except (OSError, KeyError, ValueError):
            pass
    result["processes"] = sorted(processes, key=lambda p: p["rss_kib"], reverse=True)[:24]
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
        result["gpu"] = {"returncode": gpu.returncode, "output": gpu.stdout.strip(), "error": gpu.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["gpu"] = {"error": str(exc)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    with args.output.open("a", buffering=1) as output:
        while True:
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                return
            output.write(json.dumps(snapshot()) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
