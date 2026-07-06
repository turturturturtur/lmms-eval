from subprocess import CompletedProcess

from lmms_eval.tui import server


def test_validate_auth_credentials_uses_resolved_dlc_binary(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(server, "_resolve_dlc_binary", lambda: "/tmp/resolved-dlc")
    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server._validate_auth_credentials("ak", "sk") is True

    assert calls
    command = calls[0]
    assert command[0] == "/tmp/resolved-dlc"
    assert "/mnt/cpfsB/<USER>/dlc" not in command
    assert "--access_id" in command
    assert "--access_key" in command
    assert "--ignore_local_config" in command
