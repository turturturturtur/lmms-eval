#!/usr/bin/env python3
"""Submit the worker CLI arguments through CreateJob, retaining CPFS metadata.

The legacy DLC CLI drops AccessPointId. Keep it for job queries, but use the
native API for creation so the verified CPFS A access point and RoleArn survive.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmms_eval.storage_policy import QUOTA_ID, WORKSPACE_ID, data_sources, validate_paths


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["submit"])
    p.add_argument("kind", choices=["pytorchjob"])
    for name in ("name", "command", "worker_image", "worker_cpu", "worker_gpu", "worker_memory",
                 "worker_shared_memory", "data_source_uris", "resource_id", "workspace_id",
                 "vpc_id", "switch_id", "security_group_id", "extended_cidrs"):
        p.add_argument(f"--{name}", required=True)
    for name, default in (("workers", 1), ("priority", 6), ("job_max_running_time_minutes", 0),
                          ("job_reserved_minutes", 0)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--shell", default="bash")
    p.add_argument("--region", default="cn-wulanchabu")
    p.add_argument("--endpoint", default="pai-dlc.cn-wulanchabu.aliyuncs.com")
    p.add_argument("--protocol", default="https")
    p.add_argument("--config", "-c", default=str(Path.home() / ".dlc/config"))
    p.add_argument("--access_id", "-a")
    p.add_argument("--access_key", "-k")
    p.add_argument("--security_token", "-S")
    p.add_argument("--ignore_local_config", "-I", action="store_true")
    p.add_argument("--preview", action="store_true")
    return p


def build_request(args):
    if args.resource_id != QUOTA_ID or args.workspace_id != WORKSPACE_ID:
        raise ValueError(f"Submissions require workspace {WORKSPACE_ID} and quota {QUOTA_ID}")
    validate_paths(args.command, "command")
    return {
        "DisplayName": args.name, "WorkspaceId": WORKSPACE_ID, "ResourceId": QUOTA_ID,
        "JobType": "PyTorchJob", "Priority": args.priority,
        "JobMaxRunningTimeMinutes": args.job_max_running_time_minutes,
        "JobSpecs": [{"Type": "Worker", "PodCount": args.workers, "Image": args.worker_image,
                      "ResourceConfig": {"CPU": args.worker_cpu, "GPU": args.worker_gpu,
                                         "Memory": args.worker_memory, "SharedMemory": args.worker_shared_memory}}],
        "DataSources": data_sources(args.data_source_uris),
        "UserVpc": {"VpcId": args.vpc_id, "SwitchId": args.switch_id,
                    "SecurityGroupId": args.security_group_id,
                    "ExtendedCIDRs": args.extended_cidrs.split(",")},
        "Settings": {"Shell": "/bin/" + args.shell.removeprefix("/bin/"),
                     "JobReservedMinutes": args.job_reserved_minutes},
        "UserCommand": args.command,
    }


def main():
    args = parser().parse_args()
    request = build_request(args)
    if args.preview:
        print(json.dumps(request, indent=2))
        return
    credentials = {}
    if not args.ignore_local_config and Path(args.config).is_file():
        credentials = tomllib.loads(Path(args.config).read_text()).get("user", {})
    access_id = args.access_id or credentials.get("access_id")
    access_key = args.access_key or credentials.get("access_key")
    token = args.security_token or credentials.get("security_token")
    if not access_id or not access_key:
        raise ValueError("DLC credentials are required; provide credentials or a DLC config file")
    command = ["aliyun", "pai-dlc", "CreateJob", "--region", args.region,
               "--endpoint", args.endpoint, "--mode", "StsToken" if token else "AK",
               "--access-key-id", access_id, "--access-key-secret", access_key,
               "--body", json.dumps(request)]
    if token:
        command += ["--sts-token", token]
    cancel_file = os.environ.get("LMMS_EVAL_WEBUI_CANCEL_FILE")
    if cancel_file and Path(cancel_file).exists():
        sys.exit("[ERROR] Workflow cancelled before DLC creation")
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        message = proc.stderr or proc.stdout
        for secret in (access_id, access_key, token):
            if secret:
                message = message.replace(secret, "********")
        raise RuntimeError(message)
    response = json.loads(proc.stdout)
    if not response.get("JobId"):
        raise RuntimeError("CreateJob returned no JobId")
    # Compatible with the existing submitters' table parser.
    print(f"| {response['JobId']} |")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        sys.exit(f"[ERROR] {error}")
