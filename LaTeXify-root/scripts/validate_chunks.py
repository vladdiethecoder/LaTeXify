#!/usr/bin/env python
import json, collections, sys, pathlib
p = pathlib.Path(sys.argv[1])
need = {
  "doc_id":str, "chunk_id":str, "page_start":int, "page_end":int,
  "block_ids":list, "text":str, "math_latex":list, "flags":list, "meta":dict
}
ok=True; n=0; roles=collections.Counter()
for n,line in enumerate(p.open('r',encoding='utf-8'), 1):
    try: obj=json.loads(line)
    except Exception as e: print(f"LINE {n} JSON error: {e}"); ok=False; continue
    for k,t in need.items():
        if k not in obj or not isinstance(obj[k], t):
            print(f"LINE {n} missing/typed {k}"); ok=False
    if obj.get("page_start",0) > obj.get("page_end",0):
        print(f"LINE {n} bad page range"); ok=False
    if not obj.get("block_ids"): print(f"LINE {n} empty block_ids"); ok=False
    if not isinstance(obj.get("meta",{}).get("role",""), str):
        print(f"LINE {n} meta.role type"); ok=False
    roles[obj.get("meta",{}).get("role","")]+=1
print("SCHEMA_OK:", ok, "LINES:", n, "ROLES:", dict(roles))
