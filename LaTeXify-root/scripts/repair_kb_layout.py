# scripts/repair_kb_layout.py
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime

def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "kb/latex").resolve()
    src_root = run_dir / "chunks.jsonl"
    dst_dir = run_dir / "chunks"
    dst_file = dst_dir / "chunks.jsonl"

    events = run_dir / "build_latex_kb.log.jsonl"
    def log(event, **details):
        rec = {"time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **details}
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not src_root.exists():
        print(f"[repair] Nothing to do; {src_root} not found.", file=sys.stderr)
        log("repair_noop", reason="missing_root_chunks", path=str(src_root))
        sys.exit(0)

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root, dst_file)
    log("repair_copied", src=str(src_root), dst=str(dst_file))
    print(json.dumps({"ok": True, "copied": str(dst_file)}, indent=2))

if __name__ == "__main__":
    main()
