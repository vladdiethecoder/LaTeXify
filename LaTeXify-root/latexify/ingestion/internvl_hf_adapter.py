from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from latexify.utils.logging import log_debug, log_info, log_warning

RUNNER_SCRIPT_DEFAULT = Path(__file__).resolve().parents[2] / "scripts" / "run_internvl_hf.py"


class InternVLHFAdapter:
    """Bridges the ingestion council with the HF runner subprocess."""

    def __init__(
        self,
        *,
        runner_script: str | os.PathLike[str] | None = None,
        model_dir: str | os.PathLike[str] | None = None,
        python_executable: str | None = None,
        max_new_tokens: int = 700,
        extra_args: Optional[list[str]] = None,
        device_map: str | None = None,
        max_memory: Mapping[str, str] | None = None,
        cuda_visible_devices: str | None = None,
        offload_folder: str | os.PathLike[str] | None = None,
        timeout_seconds: int | None = None,
        env_overrides: Mapping[str, Any] | None = None,
        retries: int = 1,
    ) -> None:
        self.runner_script = Path(runner_script or RUNNER_SCRIPT_DEFAULT)
        self.model_dir = str(model_dir or RUNNER_SCRIPT_DEFAULT.parents[1] / "models" / "ocr" / "internvl-3.5-14b")
        self.python_executable = python_executable or sys.executable
        self.max_new_tokens = max_new_tokens
        self.extra_args = list(extra_args or [])
        self.device_map = device_map
        self.max_memory = {str(k): str(v) for k, v in (max_memory or {}).items()}
        self.cuda_visible_devices = cuda_visible_devices
        self.offload_folder = str(offload_folder) if offload_folder else None
        self.timeout_seconds = timeout_seconds
        self.env_overrides = {str(k): "" if v is None else str(v) for k, v in (env_overrides or {}).items()}
        self.retries = max(1, int(retries or 1))

    def generate(self, prompt: str, image_path: Path | None = None) -> tuple[str, Dict[str, Any]]:
        """Invoke the HF runner and return text + metadata."""

        if not self.runner_script.exists():
            raise FileNotFoundError(
                f"InternVL HF runner missing at {self.runner_script}. "
                "Ensure scripts/run_internvl_hf.py is available."
            )

        cmd = [
            self.python_executable,
            str(self.runner_script),
            "--model-dir",
            str(self.model_dir),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--prompt",
            prompt,
        ]
        if image_path and image_path.exists():
            cmd.extend(["--image", str(image_path)])
        if self.device_map:
            cmd.extend(["--device-map", self.device_map])
        for device, value in self.max_memory.items():
            cmd.extend(["--max-memory", f"{device}={value}"])
        if self.offload_folder:
            cmd.extend(["--offload-folder", self.offload_folder])
        cmd.extend(self.extra_args)

        env = self._build_env()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return self._invoke_subprocess(cmd, env, attempt)
            except Exception as exc:  # noqa: BLE001 - need to capture multiple error types
                last_error = exc
                log_warning(
                    "InternVL HF runner attempt failed",
                    attempt=attempt,
                    retries=self.retries,
                    error=str(exc),
                )
                if attempt == self.retries:
                    raise
        assert last_error is not None  # pragma: no cover - defensive
        raise last_error

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        for key, value in self.env_overrides.items():
            if value == "":
                env.pop(key, None)
            else:
                env[key] = value
        return env

    def _invoke_subprocess(
        self,
        cmd: list[str],
        env: Dict[str, str],
        attempt: int,
    ) -> tuple[str, Dict[str, Any]]:
        readable_cmd = " ".join(shlex.quote(part) for part in cmd)
        log_info(
            "Invoking InternVL HF runner",
            command=readable_cmd,
            attempt=attempt,
            retries=self.retries,
            timeout_seconds=self.timeout_seconds,
            device_map=self.device_map,
            max_memory=self.max_memory,
            cuda_visible_devices=self.cuda_visible_devices,
            offload_folder=self.offload_folder,
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Hugging Face runner timeout after {self.timeout_seconds}s: {readable_cmd}"
            ) from exc
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            log_warning(
                "InternVL HF runner exited non-zero",
                returncode=proc.returncode,
                command=readable_cmd,
                stderr_tail=stderr[-512:] if stderr else "",
                stdout_tail=stdout[-512:] if stdout else "",
            )
            raise RuntimeError(
                f"Hugging Face runner failed (exit {proc.returncode}). "
                f"command={readable_cmd} stdout={stdout[-256:]} stderr={stderr[-256:]}"
            )
        text, parsed_meta = self._parse_stdout(stdout)
        metadata: Dict[str, Any] = {
            "runner": "hf",
            "raw_stdout_tail": stdout[-512:],
            "stderr_tail": stderr[-512:] if stderr else "",
            "timeout_seconds": self.timeout_seconds,
            "device_map": self.device_map,
            "max_memory": self.max_memory,
            "cuda_visible_devices": self.cuda_visible_devices,
            "offload_folder": self.offload_folder,
            "command": readable_cmd,
            "attempt": attempt,
        }
        metadata.update(parsed_meta)
        log_debug(
            "InternVL HF runner succeeded",
            attempt=attempt,
            command=readable_cmd,
        )
        return text, metadata

    @staticmethod
    def _parse_stdout(payload: str) -> tuple[str, Dict[str, Any]]:
        """Best-effort parsing of runner stdout. Supports JSON payloads or text."""

        payload = payload.strip()
        if not payload:
            return "", {}
        lines = payload.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = parsed.get("text") or parsed.get("response") or ""
                return text.strip(), {"format": "json", "raw_json": parsed}
        marker = "Assistant:"
        if marker in payload:
            _, tail = payload.split(marker, 1)
            tail = tail.strip()
            if tail.startswith("----------"):
                tail = tail.split("\n", 1)[1] if "\n" in tail else ""
            return tail.strip(), {"format": "text"}
        return payload, {"format": "text"}


__all__ = ["InternVLHFAdapter"]
