"""End-to-end 5-stage synthetic data pipeline runner over unzipped harvey_lab_corpus/ folder (or .zip)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_client import ChatClient, load_env_file
from stage1_generate_configs import build_domain_configs
from stage2_synthesize import run_stage2_synthesis
from stage3_clean_and_filter import run_stage3_filter_and_split
from stage4_eval_and_feedback import run_stage4_eval_and_dpo

CLOUDTOP_ARTIFACT_ZIP = Path(
    "/usr/local/google/home/malireddi/.gemini/jetski/brain/95036e22-0ed9-4fec-90bc-f25602fd0066/harvey_lab_corpus.zip"
)
HERE = Path(__file__).resolve().parent
DEFAULT_SKILL_FILE = HERE / "skills" / "sft_rl_synthesis" / "SKILL.md"


def resolve_corpus_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path
    env_vars = load_env_file()
    for key in ("CORPUS_PATH", "CORPUS_ZIP_PATH"):
        if env_vars.get(key):
            return Path(env_vars[key]).expanduser()
    # Priority 1: Unzipped folder right inside harvey_synth_pipeline/harvey_lab_corpus/
    for candidate in (
        HERE / "harvey_lab_corpus",
        HERE.parent / "harvey_lab_corpus",
        HERE / "harvey_lab_corpus.zip",
        HERE.parent / "harvey_lab_corpus.zip",
        CLOUDTOP_ARTIFACT_ZIP,
    ):
        if candidate.exists():
            return candidate
    return HERE / "harvey_lab_corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-stage synthetic SFT/RL/DPO pipeline over harvey_lab_corpus")
    parser.add_argument(
        "--corpus-path",
        "--corpus-zip",
        dest="corpus_path",
        type=Path,
        default=None,
        help="Path to unzipped harvey_lab_corpus/ folder (default: ./harvey_lab_corpus) or harvey_lab_corpus.zip",
    )
    parser.add_argument(
        "--skill-file",
        type=Path,
        default=DEFAULT_SKILL_FILE,
        help="Path to SKILL.md injected into Teacher and Judge system prompts",
    )
    parser.add_argument("--work-dir", type=Path, default=HERE / "output")
    parser.add_argument("--teacher-api-key", type=str, default=None, help="Teacher API key (or edit api_keys.env)")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model name (default: gemini-argon-4)")
    parser.add_argument("--teacher-base-url", type=str, default=None)
    parser.add_argument("--api-format", choices=["auto", "gemini", "openai"], default="auto")
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--limit-domains", type=int, default=None, help="Limit number of domains for quick test runs")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--test-api", action="store_true", help="Send 1 test call to verify API key + model, then exit")
    parser.add_argument("--skip-student-eval", action="store_true", help="Run Stages 1-3 only (skip Student eval/DPO)")
    parser.add_argument("--mock", action="store_true", help="Offline smoke test using mock Teacher/Judge/Student responses")
    args = parser.parse_args()

    teacher = ChatClient.from_env(
        "TEACHER",
        cli_base_url=args.teacher_base_url,
        cli_api_key=args.teacher_api_key,
        cli_model=args.teacher_model,
        api_format=args.api_format,
        mock=args.mock,
    )
    judge = ChatClient.from_env(
        "JUDGE",
        fallback_prefix="TEACHER",
        cli_base_url=args.teacher_base_url if not args.mock else None,
        cli_api_key=args.teacher_api_key if not args.mock else None,
        cli_model=args.teacher_model if not args.mock else None,
        api_format=args.api_format,
        mock=args.mock,
    )

    if args.test_api:
        print(f"Testing connection -> format={teacher._resolved_format()} url={teacher.base_url} model={teacher.model}")
        thought, reply = teacher.complete(
            [{"role": "user", "content": 'Reply with ONLY valid JSON: {"status": "ok"}'}],
            temperature=0.0,
            max_tokens=128,
        )
        print("Success! Thought:", thought[:120], "| Reply:", reply)
        return

    skill_text = ""
    if args.skill_file and args.skill_file.exists():
        skill_text = args.skill_file.read_text(encoding="utf-8")
        print(f"[Skill] Loaded {args.skill_file.name} ({len(skill_text)} chars) into Teacher & Judge prompts")

    corpus_path = resolve_corpus_path(args.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Unzip harvey_lab_corpus.zip into ./harvey_lab_corpus/ "
            f"(inside harvey_synth_pipeline/), set CORPUS_PATH in api_keys.env, or pass --corpus-path."
        )

    work_dir: Path = args.work_dir
    cfg_path = work_dir / "domain_configs.json"
    raw_jsonl = work_dir / "raw_candidates.jsonl"
    feedback_path = work_dir / "eval_feedback.json"

    mode_str = "unzipped folder" if corpus_path.is_dir() else "zip archive"
    print(f"[Stage 1] Using corpus ({mode_str}): {corpus_path}")
    cfg = build_domain_configs(
        corpus_path=corpus_path,
        out_path=cfg_path,
        base_samples_per_domain=args.samples_per_domain,
        feedback_path=feedback_path if feedback_path.exists() else None,
    )
    if args.limit_domains:
        active = cfg["domains"][: args.limit_domains]
        active_samples = sum(d["target_samples"] for d in active)
        print(
            f"  Active domains: {len(active)} / {cfg['total_domains']} | "
            f"Active target samples: {active_samples} (full 49-domain config: {cfg['total_target_samples']})"
        )
    else:
        print(f"  Domains: {cfg['total_domains']} | Target samples: {cfg['total_target_samples']}")

    print(f"[Stage 2] Synthesizing grounded legal tasks from {corpus_path.name} (teacher={teacher.model})...")
    new_rows = run_stage2_synthesis(
        config_path=cfg_path,
        out_jsonl=raw_jsonl,
        teacher=teacher,
        workers=args.workers,
        limit_domains=args.limit_domains,
        skill_text=skill_text,
    )
    print(f"  Wrote {new_rows} new raw candidates -> {raw_jsonl}")

    print(f"[Stage 3] Deduplicating, checking grounding, and grading with Judge ({judge.model})...")
    stats = run_stage3_filter_and_split(
        raw_jsonl=raw_jsonl,
        out_dir=work_dir,
        judge=judge,
        workers=args.workers,
        skill_text=skill_text,
    )
    print("  Filter summary:", json.dumps(stats))

    if not args.skip_student_eval:
        student = ChatClient.from_env("STUDENT", mock=args.mock)
        if args.mock:
            orig_complete = judge.complete

            def mock_judge_complete(messages, temperature=0.0, max_tokens=1024):
                if "Short summary: the document sets out general legal rules" in messages[-1]["content"]:
                    return "", json.dumps({"criterion_scores": [0.3, 0.0, 0.3], "rubric_score": 0.2, "notes": "Incomplete"})
                return orig_complete(messages, temperature=temperature, max_tokens=max_tokens)

            judge.complete = mock_judge_complete  # type: ignore[method-assign]

        print(f"[Stage 4 & 5] Benchmarking Student ({student.model}) on holdout + building DPO pairs...")
        fb = run_stage4_eval_and_dpo(out_dir=work_dir, student=student, judge=judge, workers=args.workers)
        print(
            f"  Holdout mean: {fb['overall_holdout_mean']} | "
            f"DPO pairs: {fb['dpo_pairs_created']} | "
            f"Weak domains (<0.70): {len(fb['weak_domains_under_0_70'])}"
        )

    print(f"\nDone! Outputs written under: {work_dir}")


if __name__ == "__main__":
    main()
