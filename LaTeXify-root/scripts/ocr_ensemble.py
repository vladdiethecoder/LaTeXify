import sys, json, subprocess, os, shlex

PAGE = sys.argv[1]

def run(cmd, env):
    p = subprocess.run(shlex.split(cmd), env=env, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return {"error": p.stderr.strip(), "cmd": cmd}
    try:
        return json.loads(p.stdout.splitlines()[-1])
    except Exception:
        return {"raw": p.stdout[-4000:], "cmd": cmd}

root = os.path.dirname(os.path.abspath(__file__)) + "/.."
env_nn2   = os.environ.copy(); env_nn2["PATH"] = f"{root}/.venv-nn2/bin:" + env_nn2["PATH"]
env_dots  = os.environ.copy(); env_dots["PATH"] = f"{root}/.venv-dots/bin:" + env_dots["PATH"]

results = []
results.append(run(f"python {root}/scripts/bin/nn2_ocr.py {PAGE}",  env_nn2))
results.append(run(f"python {root}/scripts/bin/dots_ocr.py {PAGE}", env_dots))

print(json.dumps({"page": PAGE, "results": results}, indent=2))
