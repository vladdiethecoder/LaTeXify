#!/usr/bin/env python3
"""Lightweight PPO loop that optimizes syntax reward using the release pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from release.scripts.pdf_render import available_templates, render_prompt_to_pdf


@dataclass
class TrainingState:
    global_step: int = 0
    ema_reward: float | None = None


@dataclass
class RewardTracker:
    beta: float = 0.1
    value: float | None = None

    def update(self, reward: float) -> float:
        if self.value is None:
            self.value = reward
        else:
            self.value = self.beta * reward + (1 - self.beta) * self.value
        return self.value


def run_pipeline(pdf_path: Path, run_dir: Path, script_path: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(script_path),
        "--pdf",
        str(pdf_path),
        "--skip-compile",
        "--run-dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)
    rewards_path = run_dir / "rewards.json"
    if rewards_path.exists():
        return json.loads(rewards_path.read_text(encoding="utf-8"))
    return {"components": {"syntax": -1.0}}


def _load_prompt_records(path: Path) -> List[Dict[str, str]]:
    dataset = load_dataset("json", data_files=str(path))["train"]
    columns = list(dataset.column_names)
    records: List[Dict[str, str]] = []
    for item in dataset:
        record = {}
        for key in columns:
            value = item.get(key)
            if value is None:
                continue
            record[key] = value
        records.append(record)
    return records


def _append_reward_trace(trace_path: Path, payload: Dict[str, object]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_training_state(path: Path, state: TrainingState) -> None:
    payload = {"global_step": state.global_step, "ema_reward": state.ema_reward}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_training_state(path: Path) -> TrainingState:
    if not path.exists():
        return TrainingState()
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainingState(
        global_step=int(data.get("global_step", 0)),
        ema_reward=data.get("ema_reward"),
    )


def _load_optimizer_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _save_optimizer_state(path: Path, trainer: PPOTrainer, state: TrainingState) -> None:
    payload = {
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": getattr(trainer, "lr_scheduler", None).state_dict()
        if getattr(trainer, "lr_scheduler", None)
        else None,
        "global_step": state.global_step,
        "ema_reward": state.ema_reward,
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO loop that optimizes syntax reward.")
    parser.add_argument("--model", required=True, help="Path to DPO-tuned model")
    parser.add_argument("--prompts", type=Path, required=True, help="JSONL file with prompt texts")
    parser.add_argument("--script", type=Path, default=REPO_ROOT / "run_release.py")
    parser.add_argument("--output", type=Path, default=Path("models/ppo_syntax"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None, help="Resume PPO state from this checkpoint directory")
    parser.add_argument("--logdir", type=Path, default=Path("runs/ppo_syntax"))
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoints every N steps")
    parser.add_argument(
        "--default-template",
        choices=available_templates(),
        default="article",
        help="Fallback LaTeX template for prompt rendering",
    )
    parser.add_argument(
        "--reward-trace",
        type=Path,
        default=None,
        help="Optional path for JSONL reward trace (defaults to <output>/reward_trace.jsonl)",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.1,
        help="Smoothing factor for exponential moving average of rewards",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_trace_path = args.reward_trace or args.output / "reward_trace.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "training_state.json"
    checkpoint_state = TrainingState()
    optimizer_state = None
    if args.resume:
        checkpoint_state = _load_training_state(Path(args.resume) / "training_state.json")
        optimizer_state = _load_optimizer_state(Path(args.resume) / "trainer_state.pt")
    tracker = RewardTracker(beta=args.ema_beta)
    if checkpoint_state.ema_reward is not None:
        tracker.value = checkpoint_state.ema_reward

    policy_init_path = args.resume or args.model
    tokenizer = AutoTokenizer.from_pretrained(policy_init_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(policy_init_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model)
    prompt_records = _load_prompt_records(args.prompts)
    if not prompt_records:
        raise ValueError("Prompt dataset is empty")
    ppo_config = PPOConfig(
        model_name=args.model,
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        target_kl=0.05,
        ppo_epochs=1,
    )
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    if optimizer_state:
        trainer.optimizer.load_state_dict(optimizer_state["optimizer"])
        scheduler_state = optimizer_state.get("scheduler")
        if scheduler_state and getattr(trainer, "lr_scheduler", None):
            trainer.lr_scheduler.load_state_dict(scheduler_state)
    writer = SummaryWriter(log_dir=str(args.logdir))
    _save_training_state(state_path, checkpoint_state)
    for local_step in range(args.steps):
        record_idx = (checkpoint_state.global_step + local_step) % len(prompt_records)
        record = prompt_records[record_idx]
        prompt = record.get("prompt", "")
        template_name = record.get("template") or record.get("layout") or args.default_template
        metadata = {
            "title": record.get("title") or "Prompt Context",
            "subtitle": record.get("subtitle") or "",
        }
        inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)
        response_ids = trainer.generate(inputs["input_ids"], max_new_tokens=512)
        response_text = tokenizer.batch_decode(response_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            pdf_path = render_prompt_to_pdf(
                response_text,
                tmp_dir / "pdf",
                template=template_name,
                metadata=metadata,
            )
            rewards = run_pipeline(pdf_path, tmp_dir / "run", args.script)
        syntax_reward = rewards.get("components", {}).get("syntax", -1.0)
        trainer.step([prompt], [response_text], [syntax_reward])
        global_step = checkpoint_state.global_step + local_step + 1
        ema_reward = tracker.update(syntax_reward)
        writer.add_scalar("reward/syntax", syntax_reward, global_step)
        writer.add_scalar("reward/syntax_ema", ema_reward, global_step)
        _append_reward_trace(
            reward_trace_path,
            {
                "step": global_step,
                "reward_syntax": syntax_reward,
                "reward_syntax_ema": ema_reward,
                "prompt_index": record_idx,
                "template": template_name,
            },
        )
        checkpoint_state.global_step = global_step
        checkpoint_state.ema_reward = ema_reward
        _save_training_state(state_path, checkpoint_state)
        if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
            checkpoint_dir = args.output / f"checkpoint_{global_step:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            _save_training_state(checkpoint_dir / "training_state.json", checkpoint_state)
            _save_optimizer_state(checkpoint_dir / "trainer_state.pt", trainer, checkpoint_state)
            print(f"Saved checkpoint to {checkpoint_dir}")
        print(f"Step {global_step} - syntax reward: {syntax_reward:.3f}")
    final_dir = args.output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    _save_training_state(final_dir / "training_state.json", checkpoint_state)
    _save_optimizer_state(final_dir / "trainer_state.pt", trainer, checkpoint_state)
    writer.close()
    print(f"PPO-trained model saved to {final_dir}")


if __name__ == "__main__":
    main()
