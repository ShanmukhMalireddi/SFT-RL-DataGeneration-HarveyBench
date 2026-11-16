"""Stage 4 & 5: Evaluate Student (Qwen/Qwen3.5-9B), build DPO pairs, and emit domain failure feedback."""

from __future__ import annotations

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_client import ChatClient
from stage3_clean_and_filter import judge_row


def run_stage4_eval_and_dpo(
    out_dir: Path,
    student: ChatClient,
    judge: ChatClient,
    min_dpo_margin: float = 0.25,
    max_dpo_candidates: int = 200,
    workers: int = 16,
) -> dict[str, Any]:
    holdout_path = out_dir / "holdout_eval.jsonl"
    sft_path = out_dir / "sft_train.jsonl"
    dpo_path = out_dir / "dpo_preferences.jsonl"
    feedback_path = out_dir / "eval_feedback.json"

    holdout_rows = [json.loads(ln) for ln in holdout_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sft_rows = [json.loads(ln) for ln in sft_path.read_text(encoding="utf-8").splitlines() if ln.strip()][:max_dpo_candidates]

    # 1. Benchmark student on immutable holdout set -> domain_scores for next round's Stage 1
    domain_Buckets: dict[str, list[float]] = defaultdict(list)
    task_Buckets: dict[str, list[float]] = defaultdict(list)

    def eval_holdout_item(r: dict[str, Any]) -> tuple[str, str, float]:
        user_msg = r["messages"][0]
        _, student_ans = student.complete([user_msg], temperature=0.2, max_tokens=4096)
        sc = judge_row(judge, user_msg["content"], student_ans, r["rubric"])
        return r["domain"], r["task_type"], sc

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for dom, ttype, sc in ex.map(eval_holdout_item, holdout_rows):
            domain_Buckets[dom].append(sc)
            task_Buckets[ttype].append(sc)

    domain_scores = {d: round(sum(v) / len(v), 4) for d, v in sorted(domain_Buckets.items())}
    task_scores = {t: round(sum(v) / len(v), 4) for t, v in sorted(task_Buckets.items())}
    all_scores = [s for v in domain_Buckets.values() for s in v]
    overall_mean = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # 2. Roll out student on SFT train prompts to build (chosen, rejected) DPO pairs
    def build_pair(r: dict[str, Any]) -> dict[str, Any] | None:
        user_msg = r["messages"][0]
        _, student_ans = student.complete([user_msg], temperature=0.7, max_tokens=4096)
        student_score = judge_row(judge, user_msg["content"], student_ans, r["rubric"])
        teacher_score = float(r.get("rubric_score", 1.0))
        if teacher_score - student_score >= min_dpo_margin:
            return {
                "id": f"dpo__{r['id']}",
                "domain": r["domain"],
                "task_type": r["task_type"],
                "prompt": [user_msg],
                "chosen": [{"role": "assistant", "content": r["messages"][1]["content"]}],
                "rejected": [{"role": "assistant", "content": student_ans}],
                "chosen_score": teacher_score,
                "rejected_score": round(student_score, 4),
            }
        return None

    dpo_written = 0
    with dpo_path.open("w", encoding="utf-8") as fdpo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for pair in ex.map(build_pair, sft_rows):
                if pair is not None:
                    fdpo.write(json.dumps(pair, ensure_ascii=True) + "\n")
                    dpo_written += 1

    feedback = {
        "holdout_cases_evaluated": len(all_scores),
        "overall_holdout_mean": overall_mean,
        "dpo_pairs_created": dpo_written,
        "weak_domains_under_0_70": [d for d, sc in domain_scores.items() if sc < 0.70],
        "domain_scores": domain_scores,
        "task_type_scores": task_scores,
    }
    feedback_path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
    return feedback
