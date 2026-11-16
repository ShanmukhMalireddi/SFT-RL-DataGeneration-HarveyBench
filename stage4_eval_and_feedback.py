"""Stage 4: Roll out Student on training prompts to build DPO pairs and compute domain failure feedback."""

from __future__ import annotations

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_client import ChatClient
from stage3_clean_and_filter import judge_row


def run_stage4_dpo_and_feedback(
    out_dir: Path,
    student: ChatClient,
    judge: ChatClient,
    min_dpo_margin: float = 0.25,
    max_dpo_candidates: int = 200,
    workers: int = 16,
    skill_text: str = "",
) -> dict[str, Any]:
    sft_path = out_dir / "sft_train.jsonl"
    dpo_path = out_dir / "dpo_preferences.jsonl"
    feedback_path = out_dir / "eval_feedback.json"

    sft_rows = [json.loads(ln) for ln in sft_path.read_text(encoding="utf-8").splitlines() if ln.strip()][:max_dpo_candidates]

    domain_buckets: dict[str, list[float]] = defaultdict(list)
    task_buckets: dict[str, list[float]] = defaultdict(list)

    def process_candidate(r: dict[str, Any]) -> tuple[str, str, float, dict[str, Any] | None]:
        user_msg = r["messages"][0]
        _, student_ans = student.complete([user_msg], temperature=0.7, max_tokens=4096)
        student_score = judge_row(judge, user_msg["content"], student_ans, r["rubric"], skill_text=skill_text)
        teacher_score = float(r.get("rubric_score", 1.0))

        pair = None
        if teacher_score - student_score >= min_dpo_margin:
            pair = {
                "id": f"dpo__{r['id']}",
                "domain": r["domain"],
                "task_type": r["task_type"],
                "prompt": [user_msg],
                "chosen": [{"role": "assistant", "content": r["messages"][1]["content"]}],
                "rejected": [{"role": "assistant", "content": student_ans}],
                "chosen_score": teacher_score,
                "rejected_score": round(student_score, 4),
            }
        return r["domain"], r["task_type"], student_score, pair

    dpo_written = 0
    with dpo_path.open("w", encoding="utf-8") as fdpo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for dom, ttype, sc, pair in ex.map(process_candidate, sft_rows):
                domain_buckets[dom].append(sc)
                task_buckets[ttype].append(sc)
                if pair is not None:
                    fdpo.write(json.dumps(pair, ensure_ascii=True) + "\n")
                    dpo_written += 1

    domain_scores = {d: round(sum(v) / len(v), 4) for d, v in sorted(domain_buckets.items())}
    task_scores = {t: round(sum(v) / len(v), 4) for t, v in sorted(task_buckets.items())}
    all_scores = [s for v in domain_buckets.values() for s in v]
    overall_mean = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    feedback = {
        "evaluated_prompts": len(all_scores),
        "student_train_rollout_mean": overall_mean,
        "dpo_pairs_created": dpo_written,
        "weak_domains_under_0_70": [d for d, sc in domain_scores.items() if sc < 0.70],
        "domain_scores": domain_scores,
        "task_type_scores": task_scores,
    }
    feedback_path.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
    return feedback
