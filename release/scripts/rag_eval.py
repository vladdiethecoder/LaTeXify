#!/usr/bin/env python3
"""Summarize coverage of the RAG exemplar index."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_index = REPO_ROOT / "cache" / "rag_index.json"
    parser = argparse.ArgumentParser(description="Report coverage statistics for rag_index.json")
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index,
        help="Path to rag_index.json (default: release/cache/rag_index.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="How many sample doc_ids to show per snippet type",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index
    if not index_path.is_absolute():
        index_path = (REPO_ROOT / index_path).resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"rag index not found: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not data:
        print(f"{index_path} is empty; add reference_tex/ corpora to populate exemplars.")
        return
    type_counts = Counter(entry.get("type", "unknown") for entry in data)
    domain_counts = Counter(entry.get("domain") or "unlabeled" for entry in data)
    total = len(data)
    print(f"Loaded {total} exemplars from {index_path}")
    print("\nCounts by snippet type:")
    for snippet_type, count in sorted(type_counts.items()):
        pct = (count / total) * 100
        print(f"  - {snippet_type}: {count} ({pct:.1f}% of corpus)")
    print("\nCounts by domain:")
    for domain, count in sorted(domain_counts.items()):
        pct = (count / total) * 100
        print(f"  - {domain}: {count} ({pct:.1f}% of corpus)")
    if args.sample > 0:
        print("\nSample doc_ids per snippet type:")
        for snippet_type in sorted(type_counts.keys()):
            doc_ids = [entry.get("doc_id", "?") for entry in data if entry.get("type") == snippet_type]
            sample = doc_ids[: args.sample]
            joined = ", ".join(sample) if sample else "<none>"
            print(f"  - {snippet_type}: {joined}")


if __name__ == "__main__":
    main()
