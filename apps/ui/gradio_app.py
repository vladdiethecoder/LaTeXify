"""Minimal Gradio viewer for the streaming planner (placeholder)."""
import asyncio
import json
from collections import defaultdict
from typing import Dict

import gradio as gr
import requests

BACKEND = "http://localhost:8000"


def fetch_events():
    with requests.get(f"{BACKEND}/events", stream=True, timeout=60) as response:
        for raw in response.iter_lines():
            if raw and raw.startswith(b"data: "):
                yield json.loads(raw[6:].decode("utf-8"))


def run_client():
    buffers: Dict[str, str] = defaultdict(str)
    for event in fetch_events():
        if event.get("type") == "token":
            buffers[event["id"]] += event.get("token", "")
        if event.get("type") == "done":
            break
    return "\n\n".join(f"{bid}:\n{content}" for bid, content in buffers.items())


with gr.Blocks() as demo:
    gr.Markdown("# Holographic Builder (Gradio Preview)")
    output = gr.Textbox(label="Streamed Blocks", lines=10)
    demo.load(run_client, inputs=None, outputs=output)

if __name__ == "__main__":
    demo.queue().launch()
