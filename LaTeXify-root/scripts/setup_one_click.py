#!/usr/bin/env python3
"""One-click environment bootstrapper for LaTeXify.

This script installs Python dependencies, downloads local model weights,
launches the required vLLM + HF runner processes (optional), and executes a
smoke test that exercises the main pipeline plus a subset of pytest suites.

Example:
    python scripts/setup_one_click.py --pdf "Basic Skills Review Unit Assessment.pdf"
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env_bootstrap import ensure_supported_python, ensure_virtualenv
DEFAULT_PDF = "Basic Skills Review Unit Assessment.pdf"
SETUP_LOG_DIR = REPO_ROOT / "build" / "setup_logs"


def run_cmd(cmd: Sequence[str], *, env: Dict[str, str] | None = None, cwd: Path | None = None) -> None:
    location = str(cwd or REPO_ROOT)
    print(f"[setup] $ {' '.join(cmd)}  (cwd={location})")
    subprocess.run(cmd, check=True, cwd=location, env=env)


def pip_install(packages: Sequence[Sequence[str]]) -> None:
    for pkg_args in packages:
        run_cmd([sys.executable, "-m", "pip", "install", *pkg_args])


def ensure_setup_logs() -> Path:
    SETUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return SETUP_LOG_DIR


def wait_for_http(name: str, url: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status < 500:
                    print(f"[setup] {name} ready at {url}")
                    return
        except urllib.error.URLError:
            time.sleep(2)
    raise RuntimeError(f"{name} did not become ready at {url} within {timeout}s")


@dataclass
class ServiceHandle:
    name: str
    process: subprocess.Popen[str]
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        print(f"[setup] stopping {self.name} (pid={self.process.pid})")
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"[setup] force-killing {self.name}")
            self.process.kill()


class OneClickSetup:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.services: List[ServiceHandle] = []
        self.env = os.environ.copy()
        ensure_setup_logs()

    # ------------------------------------------------------------------ #
    # High-level flow
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            if not self.args.skip_deps:
                self.install_dependencies()
            if not self.args.skip_models:
                self.install_models()
            hf_meta = self.maybe_run_hf_smoke()
            endpoints = self.maybe_launch_vllm()
            if not self.args.skip_smoke:
                self.run_pipeline_smoke(endpoints, hf_meta)
            if not self.args.skip_pytest:
                self.run_pytests()
            print("[setup] ✅ Environment ready.")
        finally:
            self.teardown()

    # ------------------------------------------------------------------ #
    def install_dependencies(self) -> None:
        print("[setup] Installing pip dependencies...")
        pip_install([["-U", "pip"]])
        extras = ".[dev,ocr,llama,hf-runner]"
        pip_install([["-e", extras]])
        if not self.args.skip_vllm and not self.args.skip_vllm_install:
            if sys.version_info >= (3, 14) and not self.args.force_vllm:
                print(
                    "[setup] Skipping vLLM install because Python "
                    f"{sys.version.split()[0]} is not supported by vLLM/numba. "
                    "Use --force-vllm with Python <= 3.13 if you need it."
                )
                self.args.skip_vllm = True
            else:
                pip_install([["vllm>=0.4.2"]])

    def install_models(self) -> None:
        print("[setup] Downloading model weights...")
        models = ["ocr/internvl-3.5-14b", "ocr/florence-2-large"]
        if self.args.install_all_models:
            models = ["all"]
        run_cmd([sys.executable, "scripts/install_models.py", "--models", *models])

    def maybe_run_hf_smoke(self) -> Dict[str, str]:
        if self.args.skip_hf_smoke:
            return {}
        model_dir = REPO_ROOT / "models" / "ocr" / "internvl-3.5-14b"
        if not model_dir.exists():
            print("[setup] Skipping HF smoke (model missing). Run install_models first.")
            return {}
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_internvl_hf.py"),
            "--model-dir",
            str(model_dir),
            "--dry-run",
        ]
        run_cmd(cmd)
        return {"HF": "dry-run completed"}

    def maybe_launch_vllm(self) -> Dict[str, str]:
        if self.args.skip_vllm:
            return {}
        try:
            import torch  # type: ignore  # noqa: F401
        except Exception:
            if not self.args.force_vllm:
                print("[setup] torch not available; skipping vLLM launch.")
                return {}
            raise

        endpoints: Dict[str, str] = {}
        endpoints["internvl"] = self.launch_vllm_service(
            name="internvl-vllm",
            model_path=REPO_ROOT / "models" / "ocr" / "internvl-3.5-14b",
            port=self.args.internvl_port,
        )
        endpoints["florence"] = self.launch_vllm_service(
            name="florence-vllm",
            model_path=REPO_ROOT / "models" / "ocr" / "florence-2-large",
            port=self.args.florence_port,
        )
        self.env["LATEXIFY_INTERNVL_ENDPOINT"] = f"http://{self.args.host}:{self.args.internvl_port}/v1"
        self.env["LATEXIFY_FLORENCE_ENDPOINT"] = f"http://{self.args.host}:{self.args.florence_port}/v1"
        return endpoints

    def launch_vllm_service(self, *, name: str, model_path: Path, port: int) -> str:
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing. Run scripts/install_models.py first.")
        log_path = SETUP_LOG_DIR / f"{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--port",
            str(port),
            "--host",
            self.args.host,
            "--trust-remote-code",
        ]
        print(f"[setup] launching {name} -> {log_path}")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=self.env.copy(), cwd=REPO_ROOT)  # noqa: S603
        self.services.append(ServiceHandle(name=name, process=proc, log_path=log_path))
        wait_for_http(name, f"http://{self.args.host}:{port}/v1/models", timeout=self.args.service_timeout)
        return f"http://{self.args.host}:{port}/v1"

    def run_pipeline_smoke(self, endpoints: Dict[str, str], hf_meta: Dict[str, str]) -> None:
        print("[setup] Running pipeline smoke test via run_local.py ...")
        sample_pdf = self.resolve_sample_pdf()
        cmd = [
            sys.executable,
            "run_local.py",
            "--pdf",
            str(sample_pdf),
            "--title",
            "Setup Smoke Test",
            "--allow-fallback",
            "--skip-qa",
        ]
        env = self.env.copy()
        if self.args.pipeline_internvl_mode == "hf":
            env["INTERNVL_MODE"] = "hf"
        run_cmd(cmd, env=env)
        print("[setup] Smoke test complete (endpoints:", endpoints, "hf:", hf_meta, ")")

    def run_pytests(self) -> None:
        print("[setup] Running focused pytest suites...")
        targets = [
            "tests/test_ingestion_council.py",
            "tests/test_vision_backend.py",
            "tests/test_e2e_smoke.py",
        ]
        run_cmd([sys.executable, "-m", "pytest", "-q", *targets])

    def teardown(self) -> None:
        for service in reversed(self.services):
            service.stop()

    def resolve_sample_pdf(self) -> Path:
        candidate = Path(self.args.pdf)
        if candidate.exists():
            return candidate
        default_dir = REPO_ROOT / "dev" / "inputs"
        fallback = default_dir / self.args.pdf
        if fallback.exists():
            return fallback
        default = default_dir / DEFAULT_PDF
        if default.exists():
            return default
        raise FileNotFoundError(
            f"Sample PDF not found. Checked {candidate}, {fallback}, and {default}. "
            "Place a PDF under dev/inputs or provide --pdf /path/to.pdf"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=str, default=None, help="Preferred Python interpreter (3.10-3.13) for the venv.")
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF, help="Sample PDF name or absolute path for smoke test.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for local vLLM servers.")
    parser.add_argument("--internvl-port", type=int, default=8090, help="Port for InternVL vLLM server.")
    parser.add_argument("--florence-port", type=int, default=8091, help="Port for Florence vLLM server.")
    parser.add_argument("--service-timeout", type=int, default=180, help="Seconds to wait for services to become ready.")
    parser.add_argument("--pipeline-internvl-mode", choices=["vllm", "hf"], default="vllm", help="Mode used during the pipeline smoke test.")

    parser.add_argument("--skip-deps", action="store_true", help="Skip pip install step.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument("--skip-vllm", action="store_true", help="Skip launching vLLM servers.")
    parser.add_argument("--skip-vllm-install", action="store_true", help="Do not pip-install vllm (assume pre-installed).")
    parser.add_argument("--skip-hf-smoke", action="store_true", help="Skip the HF runner dry-run check.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip run_local smoke test.")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest suite.")
    parser.add_argument("--install-all-models", action="store_true", help="Download every model defined in scripts/install_models.py.")
    parser.add_argument("--force-vllm", action="store_true", help="Launch vLLM even if CUDA/torch checks fail.")
    parser.add_argument("--venv-path", type=Path, default=Path(".venv"), help="Virtualenv directory (auto-created).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    script_path = Path(__file__).resolve()
    ensure_supported_python(script_path, sys.argv, args.python_bin)
    ensure_virtualenv(args.venv_path, script_path, sys.argv)
    setup = OneClickSetup(args)
    try:
        setup.run()
    except subprocess.CalledProcessError as exc:
        print(f"[setup] Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
