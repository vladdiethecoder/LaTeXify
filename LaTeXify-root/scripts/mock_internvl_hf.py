#!/usr/bin/env python3
"""Mock InternVL HF runner used for unit tests.

Accepts a subset of the real runner arguments and emits deterministic text or
JSON responses so tests can exercise the ingestion adapter without loading
actual models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock InternVL runner.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt payload.")
    parser.add_argument("--mode", choices=["text", "json"], default="text", help="Output format.")
    parser.add_argument("--fail-code", type=int, default=0, help="Non-zero exit to simulate failures.")
    parser.add_argument("--fail-once", action="store_true", help="Fail only on the first invocation.")
    parser.add_argument("--state-file", type=Path, default=None, help="Path used to store fail-once state.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay before responding.")
    parser.add_argument("--stderr-message", type=str, default=None, help="Message to emit on stderr.")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument("--extra", action="append", default=[])
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    if args.sleep:
        time.sleep(args.sleep)
    if args.stderr_message:
        print(args.stderr_message, file=sys.stderr)

    should_fail = args.fail_code != 0
    if args.fail_once:
        marker = args.state_file
        if marker is None:
            marker = Path(".mock_internvl_fail_once")
        if not marker.exists():
            marker.write_text("failed", encoding="utf-8")
        else:
            should_fail = False
            try:
                marker.unlink()
            except OSError:
                pass
    if should_fail:
        return int(args.fail_code)

    if args.mode == "json":
        payload = {
            "text": f"JSON:{args.prompt}",
            "prompt": args.prompt,
            "mode": "json",
            "images": args.image or [],
        }
        print(json.dumps(payload))
    else:
        print("Assistant:\n----------")
        print(f"TEXT:{args.prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
