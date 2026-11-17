"""Stage 3: Deduplication, structural checks, separate Judge LLM grading, and locked holdout split."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from llm_client import ChatClient, extract_json_object


def word_shingles(text: str, k: int = 5) -> set[int]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if len(tokens) < k:
        return {hash(" ".join(tokens))}
    return {hash(" ".join(tokens[i : i + k])) for i in range(0, len(tokens) - k + 1, 2)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


def deterministic_quality_ok(row: dict[str, Any]) -> tuple[bool, str]:
    msgs = row.get("messages", [])
    if len(msgs) != 2:
        return False, "bad_message_count"
    user_c = str(msgs[0].get("content", ""))
    asst_c = str(msgs[1].get("content", ""))
    rubric = row.get("rubric", [])
    if len(asst_c) < 200:
        return False, "answer_too_short"
    if len(rubric) < 2:
        return False, "too_few_rubric_items"
    if "SYNTHESIZE_TASK_JSON" in asst_c or "rubric_score" in asst_c:
        return False, "prompt_leakage"
    # Grounding check: at least one source_doc name or substantive packet term appears in response
    doc_names = row.get("source_docs", [])
    if doc_names and not any(dn.split(".")[0][:10].lower() in (user_c + asst_c).lower() for dn in doc_names):
        return False, "missing_source_grounding"
    return True, "ok"


def judge_row(
    judge: ChatClient,
    user_content: str,
    answer_content: str,
    rubric: list[dict[str, Any]],
    skill_text: str = "",
) -> float:
    rubric_str = json.dumps(rubric, ensure_ascii=True)
    sys_msg = "You are an impartial legal rubric verifier. Score strictly against the provided source packet and rubric."
    if skill_text:
        sys_msg += f"\n\n=== SKILL INSTRUCTIONS ===\n{skill_text.strip()}"
    msgs = [
        {
            "role": "system",
            "content": sys_msg,
        },
        {
            "role": "user",
            "content": (
                f"GRADE_RUBRIC_JSON\n"
                f"Task & Source Packet:\n{user_content[:9000]}\n\n"
                f"Rubric:\n{rubric_str}\n\n"
                f"Candidate Response:\n{answer_content[:6000]}\n\n"
                'Return ONLY JSON: {"criterion_scores": [0.0_to_1.0, ...], "rubric_score": 0.0_to_1.0, "notes": "..."}'
            ),
        },
    ]
    _, out = judge.complete(msgs, temperature=0.0, max_tokens=1024)
    parsed = extract_json_object(out)
    return float(parsed.get("rubric_score", 0.0))


def run_stage3_filter_and_split(
    raw_jsonl: Path,
    out_dir: Path,
    judge: ChatClient,
    min_rubric_score: float = 0.90,
    max_jaccard: float = 0.75,
    holdout_ratio: float = 0.10,
    workers: int = 16,
    skill_text: str = "",
) -> dict[str, Any]:
    rows = [json.loads(ln) for ln in raw_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]

    stats: dict[str, int] = {
        "raw_total": len(rows),
        "dropped_structural": 0,
        "dropped_duplicate": 0,
        "dropped_low_judge": 0,
        "kept_total": 0,
        "sft_train": 0,
        "holdout_eval": 0,
    }

    # 1. Structural & exact/near-duplicate filtering
    seen_hashes: set[str] = set()
    kept_shingles: list[set[int]] = []
    deduped: list[dict[str, Any]] = []

    for r in rows:
        ok, _ = deterministic_quality_ok(r)
        if not ok:
            stats["dropped_structural"] += 1
            continue
        ans = r["messages"][1]["content"]
        h = r.get("sha16") or hashlib.sha256(ans.encode("utf-8")).hexdigest()[:16]
        if h in seen_hashes:
            stats["dropped_duplicate"] += 1
            continue
        sh = word_shingles(ans)
        if any(jaccard(sh, prev) > max_jaccard for prev in kept_shingles[-250:]):
            stats["dropped_duplicate"] += 1
            continue
        seen_hashes.add(h)
        kept_shingles.append(sh)
        deduped.append(r)

    # 2. Separate Judge LLM verification
    def score_one(r: dict[str, Any]) -> dict[str, Any] | None:
        try:
            sc = judge_row(judge, r["messages"][0]["content"], r["messages"][1]["content"], r["rubric"], skill_text=skill_text)
        except Exception:
            return None
        r["rubric_score"] = round(sc, 4)
        return r

    verified: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for scored in ex.map(score_one, deduped):
            if scored is None or scored["rubric_score"] < min_rubric_score:
                stats["dropped_low_judge"] += 1
            else:
                verified.append(scored)

    stats["kept_total"] = len(verified)

    # 3. Deterministic hash split into SFT Train vs. Immutable Holdout + RL Prompts
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft_train.jsonl"
    rl_path = out_dir / "rl_prompts.jsonl"
    holdout_path = out_dir / "holdout_eval.jsonl"

    # Split first, then write. If the holdout would be empty, MOVE (not copy) one row
    # from train to holdout so no row is ever in both sets.
    train_rows, hold_rows = [], []
    for r in verified:
        bucket = int(hashlib.md5(r["id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (hold_rows if bucket < holdout_ratio else train_rows).append(r)
    if not hold_rows and len(train_rows) > 1:
        hold_rows.append(train_rows.pop())
    with (
        sft_path.open("w", encoding="utf-8") as fsft,
        rl_path.open("w", encoding="utf-8") as frl,
        holdout_path.open("w", encoding="utf-8") as fhold,
    ):
        for r in hold_rows:
            fhold.write(json.dumps(r, ensure_ascii=True) + "\n")
        for r in train_rows:
            fsft.write(json.dumps(r, ensure_ascii=True) + "\n")
            frl.write(json.dumps({"id": r["id"], "domain": r["domain"], "task_type": r["task_type"],
                                  "prompt": [r["messages"][0]], "rubric": r["rubric"]}, ensure_ascii=True) + "\n")
    stats["sft_train"] = len(train_rows)
    stats["holdout_eval"] = len(hold_rows)

    (out_dir / "filter_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
