# scripts/build_chunks.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from dev.chunking.page_aware_chunker import build_chunks_for_run

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="dev/runs/<STAMP>")
    p.add_argument("--pdf", required=True, help="Path to the source PDF (for bookkeeping)")
    p.add_argument("--max_chars", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--min_par_len", type=int, default=60)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    pdf_path = Path(args.pdf)

    chunks = build_chunks_for_run(
        run_dir=run_dir,
        pdf_path=pdf_path,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_par_len=args.min_par_len,
    )

    out_path = run_dir / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # Also write a tiny manifest for convenience
    manifest = {
        "pdf": str(pdf_path),
        "pages": len(list((run_dir / "pages").glob("*.png"))),
        "chunks": len(chunks),
        "params": {
            "max_chars": args.max_chars,
            "overlap": args.overlap,
            "min_par_len": args.min_par_len,
        },
    }
    (run_dir / "chunks.meta.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(chunks)} chunks")

if __name__ == "__main__":
    main()
