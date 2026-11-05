import pytest

from lmms_eval import runtime_preflight


def test_python_without_safe_import_paths_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime_preflight.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda _: "1.0")
    with pytest.raises(RuntimeError, match="Python >=3.11"):
        runtime_preflight.check_runtime()


def test_missing_vllm_is_rejected_before_worker_allocation(monkeypatch):
    def version(name):
        if name == "vllm":
            raise runtime_preflight.metadata.PackageNotFoundError(name)
        return "1.0"
    monkeypatch.setattr(runtime_preflight.metadata, "version", version)
    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda _: "/bin/ninja")
    with pytest.raises(RuntimeError, match="missing package: vllm"):
        runtime_preflight.check_runtime(vllm=True)


def test_incompatible_torch_is_rejected_using_vllm_metadata(monkeypatch):
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda _: "2.8.0")
    monkeypatch.setattr(runtime_preflight.metadata, "requires", lambda _: ["torch==2.11.0", 'unused; extra == "audio"'])
    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda _: "/bin/ninja")
    with pytest.raises(RuntimeError, match="vllm requires torch==2.11.0, installed 2.8.0"):
        runtime_preflight.check_runtime(vllm=True)


def test_optional_flash_attention_binary_incompatibility_is_rejected(monkeypatch):
    monkeypatch.setattr(runtime_preflight.metadata, "version", lambda _: "1.0")
    monkeypatch.setattr(runtime_preflight.metadata, "requires", lambda _: [])
    monkeypatch.setattr(runtime_preflight.shutil, "which", lambda _: "/bin/ninja")
    monkeypatch.setattr(runtime_preflight.importlib.util, "find_spec", lambda _: object())

    def import_module(name):
        if name == "flash_attn.ops.triton.rotary":
            raise ImportError("flash_attn_2_cuda.so: undefined symbol: c10_cuda_check")

    monkeypatch.setattr(runtime_preflight.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="flash_attn.*undefined symbol"):
        runtime_preflight.check_runtime(vllm=True, imports=True)
