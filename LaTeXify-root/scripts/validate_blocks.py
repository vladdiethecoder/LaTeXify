#!/usr/bin/env python
import sys, json, collections, pathlib
p = pathlib.Path(sys.argv[1])
need = {"doc_id":str,"page":int,"block_id":str,"type":str,"text":str,"flags":list}
valid = {"text","formula","figure","table","header","footer","title","section"}
ok=True; n=0; types=collections.Counter(); flags=collections.Counter()
for n,line in enumerate(p.open('r',encoding='utf-8'),1):
    try: obj=json.loads(line)
    except Exception as e: print(f"LINE {n} JSON error: {e}"); ok=False; continue
    for k,t in need.items():
        if k not in obj or not isinstance(obj[k], t): print(f"LINE {n} missing/typed {k}"); ok=False
    if obj.get("type") not in valid: print(f"LINE {n} bad type {obj.get('type')}"); ok=False
    if obj.get("type")=="formula" and "latex_consensus" in obj and not isinstance(obj["latex_consensus"], str):
        print(f"LINE {n} latex_consensus wrong type"); ok=False
    types[obj.get("type","?")]+=1
    for fl in obj.get("flags",[]):
        if isinstance(fl, dict) and "tag" in fl: flags[fl["tag"]]+=1
print("SCHEMA_OK:", ok, "LINES:", n)
print("TYPE_COUNTS:", dict(types))
print("FLAG_COUNTS:", dict(flags))
