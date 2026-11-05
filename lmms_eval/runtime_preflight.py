"""Check the actual evaluation interpreter before allocating a DLC worker."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import shutil
import sys

from packaging.requirements import Requirement


def check_runtime(*, vllm: bool = False, imports: bool = False) -> dict[str, str]:
    """Validate required distributions and vLLM's installed dependency contract."""
    errors: list[str] = []
    if sys.version_info < (3, 11):
        errors.append("DLC runtime requires Python >=3.11 to isolate source snapshots with PYTHONSAFEPATH")
    versions: dict[str, str] = {}
    required = ["torch", "transformers", "datasets", "pytablewriter", "sacrebleu",
                "evaluate", "rouge-score", "rouge", "jiwer", "tenacity", "python-dotenv", "openpyxl"]
    if vllm:
        required += ["vllm", "nvidia-ml-py", "flashinfer-python"]
    for name in required:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"missing package: {name}")
    if vllm and "vllm" in versions:
        for raw in metadata.requires("vllm") or []:
            dependency = Requirement(raw)
            if dependency.marker and not dependency.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(dependency.name)
                if dependency.specifier and not dependency.specifier.contains(version, prereleases=True):
                    errors.append(f"vllm requires {dependency}, installed {version}")
            except metadata.PackageNotFoundError:
                errors.append(f"vllm requires missing package: {dependency.name}")
    if vllm and shutil.which("ninja") is None:
        errors.append("missing executable: ninja")
    if imports and not errors:
        modules = ["torch", "transformers", "datasets"]
        if vllm:
            modules += ["vllm", "vllm.transformers_utils", "flashinfer", "pynvml"]
            # vLLM loads this optional module on GPU even with FLASHINFER.
            # Stale inherited binaries raise ABI errors during model loading.
            if importlib.util.find_spec("flash_attn") is not None:
                modules.append("flash_attn.ops.triton.rotary")
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as exc:
                errors.append(f"{name} import failed: {exc}")
    if errors:
        raise RuntimeError(f"Evaluation runtime is not ready: {sys.executable}\n" + "\n".join(f"  - {error}" for error in errors))
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--imports", action="store_true")
    args = parser.parse_args()
    try:
        versions = check_runtime(vllm=args.vllm, imports=args.imports)
    except RuntimeError as exc:
        parser.exit(2, f"[ERROR] {exc}\n")
    print("[INFO] Evaluation runtime preflight passed: " + json.dumps(versions, sort_keys=True))


if __name__ == "__main__":
    main()
