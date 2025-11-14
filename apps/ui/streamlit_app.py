import json
from collections import defaultdict
from typing import Dict, Tuple

import requests
import streamlit as st

BACKEND = st.secrets.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Holographic Builder", layout="wide")
st.title("Holographic Document Builder")

status_box = st.sidebar.empty()
status_log = []
placeholders: Dict[str, Tuple[st.delta_generator.DeltaGenerator, str]] = {}
buffers = defaultdict(str)
columns = []


def sse_events(url: str):
    with requests.get(url, stream=True, timeout=60) as response:
        for raw in response.iter_lines():
            if not raw:
                continue
            if raw.startswith(b"data: "):
                payload = json.loads(raw[6:].decode("utf-8"))
                yield payload


def render_plan(plan: Dict[str, any]):
    global columns
    placeholders.clear()
    buffers.clear()
    st.subheader("Layout Wireframe")
    columns = st.columns(plan["columns"])
    for block in sorted(plan["blocks"], key=lambda b: (b["column"], b["order"])):
        col = columns[block["column"] - 1]
        with col:
            st.caption(block["meta"].get("title", block["id"]))
            slot = st.empty()
            placeholders[block["id"]] = (slot, block["type"])


def update_block(block_id: str, token: str):
    buffers[block_id] += token
    placeholder, block_type = placeholders[block_id]
    content = buffers[block_id]
    if block_type == "math":
        placeholder.latex(content)
    elif block_type in ("text", "code"):
        placeholder.code(content, language="latex")
    else:
        placeholder.write(content)


def log_status(message: str):
    status_log.append(message)
    status_box.write("\n".join(status_log[-10:]))


def main():
    for event in sse_events(f"{BACKEND}/events"):
        etype = event.get("type")
        if etype == "plan":
            render_plan(event["plan"])
        elif etype == "status":
            status = event["status"]
            log_status(f"[{status.get('state')}] {status.get('agent')}: {status.get('task', '')}")
        elif etype == "token":
            block_id = event.get("id")
            if block_id in placeholders:
                update_block(block_id, event.get("token", ""))
        elif etype == "block_done":
            log_status(f"[DONE] {event.get('id')}")
        elif etype == "error":
            st.error(event.get("message", "Error"))
            break
        elif etype == "done":
            st.success("Build complete")
            break


if __name__ == "__main__":
    main()
