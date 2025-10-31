#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pathlib, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    found = 0
    for cls in src.glob("*.cls"):
        shutil.copy2(cls, dst / cls.name)
        found += 1

    if found == 0:
        print("[sync-classes][WARN] No .cls files in", src)
    else:
        print(f"[sync-classes] Copied {found} class file(s) → {dst}")

if __name__ == "__main__":
    main()
