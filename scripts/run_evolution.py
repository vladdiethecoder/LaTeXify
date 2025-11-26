import argparse
import logging
from pathlib import Path
import sys

# Ensure src is on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from latexify.self_improvement import (
    Archive,
    AgentVersion,
    EvolutionConfig,
    LLMPatchGenerator,
    GeneratorConfig,
    EvaluatorRunner,
    EvaluationConfig,
    EvolutionRunner,
    KnowledgeGraph,
    VectorMemory,
    LocalTextGenerator,
    LocalLLMConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement loop (DGM+ITRS).")
    parser.add_argument("--tests", nargs="+", default=["src/latexify/tests/test_smoke_release.py::test_smoke_pipeline_produces_rewards"], help="Pytest targets for evaluation.")
    parser.add_argument("--max-generations", type=int, default=2, help="Generations to run.")
    parser.add_argument("--dry-run", action="store_true", help="Use no-op patch generator.")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Enable LLM patch generator.")
    parser.add_argument("--model-path", type=str, default="/run/media/vdubrov/Non-SSD Storage/models/hf/latexify_llm/Qwen__Qwen2.5-1.5B-Instruct", help="Local HF model path for LLM patching.")
    parser.add_argument("--fallback-model-path", type=str, default="/run/media/vdubrov/Non-SSD Storage/models/hf/latexify_llm/deepseek-ai__DeepSeek-Coder-V2-Lite-Instruct", help="Fallback model if primary fails to load.")
    parser.add_argument("--device", type=str, default="cpu", help="Device map for model (auto|cuda|cpu).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repo_root = Path(__file__).resolve().parents[1]

    archive = Archive()
    base_agent = AgentVersion(version_id="v0", parent_id=None, strategy="VALIDATION", summary="baseline")

    evaluator = EvaluatorRunner(repo_root, EvaluationConfig(tests=args.tests))
    base_agent = evaluator.evaluate(base_agent)
    archive.add(base_agent)

    text_gen = None
    if not args.dry_run:
        llm_cfg = LocalLLMConfig(
            model_path=args.model_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_4bit=False,
            fallback_model_path=args.fallback_model_path,
        )
        text_gen = LocalTextGenerator(llm_cfg)

    patch_gen = LLMPatchGenerator(GeneratorConfig(dry_run=args.dry_run), text_generator=text_gen)
    runner = EvolutionRunner(
        repo_root=repo_root,
        archive=archive,
        config=EvolutionConfig(max_generations=args.max_generations),
        patch_generator=lambda parent, graph, mem: patch_gen.generate(parent, graph, mem),
        evaluator=lambda agent: evaluator.evaluate(agent),
        graph=KnowledgeGraph(),
        vector_memory=VectorMemory(),
    )

    runner.run()


if __name__ == "__main__":
    main()
