"""Stage 5: Emit SFT + On-Policy GRPO + DPO GPU bundle, and benchmark Candidate vs. Base on immutable holdout."""

from __future__ import annotations

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_client import ChatClient
from stage3_clean_and_filter import judge_row

REWARD_GRPO_TEMPLATE = '''"""On-policy GRPO reward function for output/rl_prompts.jsonl (works 100% offline in Python or via Judge API).

Compatible with:
  1. HuggingFace TRL GRPOTrainer: trl_reward_func(completions=..., rubric=..., task_type=..., **kwargs)
  2. ARB Miles GRPO (--custom-rm-path reward_grpo.reward_func): async reward_func(args, samples)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

STOPWORDS = {
    "identifies", "analyzes", "explains", "discusses", "cites", "includes", "provides", "states",
    "accurately", "whether", "under", "from", "with", "that", "this", "these", "those", "their",
    "packet", "document", "documents", "source", "section", "clause", "provisions", "provision",
    "legal", "memorandum", "response", "answer", "using", "without", "external", "facts", "outside",
}


def _completion_to_text(comp: Any) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        last = comp[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(comp, dict):
        return str(comp.get("content", ""))
    return str(comp)


def _extract_anchors(criterion: str) -> list[str]:
    # High-precision anchors: Section/Rule refs, dollar figures, percentages, quoted phrases
    exact = re.findall(
        r'"([^"]{3,40})"|\\$[\\d,]+(?:\\.\\d+)?|\\b\\d+(?:\\.\\d+)?%|\\b(?:Section|Rule|Article|Title)\\s+\\d+[A-Za-z0-9().-]*',
        criterion,
    )
    anchors = [a.lower() for a in exact if a]
    if not anchors:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9_.-]{5,}", criterion)]
        anchors = [w for w in words if w not in STOPWORDS][:6]
    return anchors


def score_offline_deterministic(text: str, rubric: list[dict[str, Any]], task_type: str = "") -> float:
    clean = text.strip()
    if len(clean) < 180:
        return 0.0

    low = clean.lower()
    # Anti-placeholder & meta-leakage penalty
    if any(bad in low for bad in ("[tbd]", "[insert", "synthesize_task_json", "grade_rubric_json")):
        return 0.05

    # Anti-repetition loop check (unique 6-word shingles / total 6-word windows)
    words = low.split()
    if len(words) >= 40:
        windows = [" ".join(words[i : i + 6]) for i in range(len(words) - 5)]
        uniq_ratio = len(set(windows)) / float(len(windows))
        if uniq_ratio < 0.55:
            return 0.05

    # 1. Rubric criterion anchor coverage (75% of score)
    total_w = 0.0
    earned_w = 0.0
    for item in rubric:
        crit = str(item.get("criterion", ""))
        w = float(item.get("weight", 1.0))
        total_w += w
        if "does not" in crit.lower() and "outside" in crit.lower():
            # Negative hallucination check passes if answer cites packet sections/docs without "[citation needed]"
            earned_w += w if ("section" in low or ".txt" in low or "article" in low) else (0.5 * w)
            continue
        anchors = _extract_anchors(crit)
        if not anchors:
            earned_w += 0.5 * w
            continue
        hits = sum(1 for a in anchors if a in low)
        earned_w += w * min(1.0, hits / max(1.0, len(anchors) * 0.6))

    rubric_frac = (earned_w / total_w) if total_w > 0 else 0.5

    # 2. Task-format structural bonus (25% of score)
    fmt_score = 0.7
    if task_type == "legal_memo":
        headings = sum(1 for h in ("issue", "analysis", "conclusion", "memorandum") if h in low)
        fmt_score = min(1.0, headings / 3.0)
    elif task_type == "compliance_checklist_table":
        pipe_lines = sum(1 for ln in clean.splitlines() if ln.count("|") >= 3)
        fmt_score = 1.0 if pipe_lines >= 3 else 0.2
    elif task_type == "issue_spotting_and_risk_redline":
        red_hits = sum(1 for k in ("risk", "redline", "replace", "proposed", "current") if k in low)
        fmt_score = min(1.0, red_hits / 3.0)

    return round(0.75 * rubric_frac + 0.25 * fmt_score, 4)


def score_via_judge_api(prompt_text: str, answer_text: str, rubric: list[dict[str, Any]]) -> float:
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("TEACHER_API_KEY", "")
    model = os.environ.get("JUDGE_MODEL") or os.environ.get("TEACHER_MODEL", "gemini-argon-4")
    base_url = (os.environ.get("JUDGE_BASE_URL") or os.environ.get("TEACHER_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if not api_key:
        return score_offline_deterministic(answer_text, rubric)

    user_msg = (
        f"Task & Packet:\\n{prompt_text[:7000]}\\n\\n"
        f"Rubric:\\n{json.dumps(rubric)}\\n\\n"
        f"Candidate Answer:\\n{answer_text[:5000]}\\n\\n"
        \'Return ONLY JSON: {"rubric_score": 0.0_to_1.0}\'
    )
    url = f"{base_url}/models/{model}:generateContent?key={api_key}"
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": user_msg}]}], "generationConfig": {"temperature": 0.0}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        txt = "".join(p.get("text", "") for p in data["candidates"][0]["content"].get("parts", []) if not p.get("thought"))
        s, e = txt.find("{"), txt.rfind("}")
        return float(json.loads(txt[s : e + 1]).get("rubric_score", 0.0))
    except Exception:
        return score_offline_deterministic(answer_text, rubric)


def trl_reward_func(
    prompts: list[Any],
    completions: list[Any],
    rubric: list[list[dict[str, Any]]],
    task_type: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    mode = os.environ.get("GRPO_REWARD_MODE", "offline").lower()
    task_types = task_type or [""] * len(completions)
    scores: list[float] = []
    for p, c, rub, ttype in zip(prompts, completions, rubric, task_types):
        ans_txt = _completion_to_text(c)
        if mode == "judge":
            p_txt = _completion_to_text(p)
            scores.append(score_via_judge_api(p_txt, ans_txt, rub))
        else:
            scores.append(score_offline_deterministic(ans_txt, rub, ttype))
    return scores


async def reward_func(args: Any, samples: list[Any]) -> list[float]:
    """Async entrypoint for ARB Miles GRPO (--custom-rm-path reward_grpo.reward_func)."""
    scores: list[float] = []
    for s in samples:
        meta = getattr(s, "metadata", None) or (s.get("metadata", {}) if isinstance(s, dict) else {})
        rub = meta.get("rubric") or (s.get("rubric", []) if isinstance(s, dict) else [])
        ttype = meta.get("task_type") or (s.get("task_type", "") if isinstance(s, dict) else "")
        resp = getattr(s, "response", None) or (s.get("response", "") if isinstance(s, dict) else "")
        scores.append(score_offline_deterministic(str(resp), rub, str(ttype)))
    return scores
'''

TRAIN_GRPO_TEMPLATE = '''"""Stage 5 On-Policy GRPO Trainer (LoRA or Full-Weight) over output/rl_prompts.jsonl.

Runs 100% offline on air-gapped GPUs by default (GRPO_REWARD_MODE=offline),
or uses Judge API if GRPO_REWARD_MODE=judge is exported.
Pass --full-weights to update all 9B parameters directly instead of LoRA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from reward_grpo import trl_reward_func


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative on-policy GRPO on Qwen/Qwen3.5-9B")
    parser.add_argument("--model", type=str, default="./checkpoints/qwen35-9b-sft-merged")
    parser.add_argument("--prompts-jsonl", type=str, default="output/rl_prompts.jsonl")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/qwen35-9b-grpo")
    parser.add_argument("--full-weights", action="store_true", help="Update all 9B weights directly (4x-8x 80GB GPUs)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (default: 2e-7 LoRA, 1e-7 full)")
    parser.add_argument("--num-generations", type=int, default=4, help="Group size G per prompt")
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[GRPO] Merged SFT folder {args.model} not found; falling back to base Qwen/Qwen3.5-9B")
        args.model = "Qwen/Qwen3.5-9B"

    lr = args.lr if args.lr is not None else (1e-7 if args.full_weights else 2e-7)
    dataset = load_dataset("json", data_files=args.prompts_jsonl, split="train")

    peft_cfg = None
    if not args.full_weights:
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=lr,
        beta=0.04,  # KL penalty toward SFT reference
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        max_prompt_length=8192,
        max_completion_length=4096,
        temperature=0.9,
        bf16=True,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=40,
    )

    mode_str = "FULL-WEIGHT" if args.full_weights else "LoRA"
    print(f"[GRPO] Starting {mode_str} on-policy GRPO (lr={lr}) from {args.model}...")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[trl_reward_func],
        args=grpo_cfg,
        train_dataset=dataset,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[GRPO] Saved {mode_str} GRPO checkpoint -> {args.output_dir}")


if __name__ == "__main__":
    main()
'''


def write_training_bundle(out_dir: Path) -> Path:
    sft_path = (out_dir / "sft_train.jsonl").resolve()
    dpo_path = (out_dir / "dpo_preferences.jsonl").resolve()
    rl_path = (out_dir / "rl_prompts.jsonl").resolve()

    sft_yaml = f"""# Stage 5 Generated Axolotl LoRA SFT Config (1x-2x GPUs)
base_model: Qwen/Qwen3.5-9B
chat_template: tokenizer_default
datasets:
  - path: {sft_path}
    ds_type: json
    type: chat_template
    field_messages: messages
    roles_to_train: [assistant]
output_dir: ./checkpoints/qwen35-9b-sft-lora
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
sequence_len: 16384
sample_packing: true
bf16: auto
flash_attention: true
gradient_checkpointing: true
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 2
optimizer: adamw_torch_fused
learning_rate: 0.0001
lr_scheduler: cosine
warmup_ratio: 0.05
"""

    full_sft_yaml = f"""# Stage 5 Generated Axolotl Full-Weight 9B SFT Config (FSDP2 on 4x-8x 80GB+ GPUs)
base_model: Qwen/Qwen3.5-9B
chat_template: tokenizer_default
datasets:
  - path: {sft_path}
    ds_type: json
    type: chat_template
    field_messages: messages
    roles_to_train: [assistant]
output_dir: ./checkpoints/qwen35-9b-sft-merged
sequence_len: 16384
sample_packing: true
bf16: auto
flash_attention: true
gradient_checkpointing: true
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 2
optimizer: adamw_torch_fused
learning_rate: 0.00001
lr_scheduler: cosine
warmup_ratio: 0.05
weight_decay: 0.01
fsdp_version: 2
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: FULL_STATE_DICT
"""

    dpo_yaml = f"""# Stage 5 Generated Axolotl LoRA DPO Config (Run on SFT checkpoint)
base_model: ./checkpoints/qwen35-9b-sft-merged
chat_template: tokenizer_default
rl: dpo
rl_beta: 0.1
datasets:
  - path: {dpo_path}
    ds_type: json
    type: chat_template.default
    field_messages: prompt
    field_chosen: chosen
    field_rejected: rejected
output_dir: ./checkpoints/qwen35-9b-dpo-lora
adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
sequence_len: 16384
bf16: auto
flash_attention: true
gradient_checkpointing: true
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 1
learning_rate: 0.000005
"""

    launch_sh = f"""#!/usr/bin/env bash
set -euo pipefail
# Run on a GPU machine:
#   STAGE5_WEIGHT_MODE: lora (default, 1x-2x GPUs) | full (all 9B params via FSDP2, 4x-8x 80GB GPUs)
#   STAGE5_RL_MODE:     grpo (default, on-policy rl_prompts.jsonl) | dpo (offline dpo_preferences.jsonl) | none
STAGE5_WEIGHT_MODE="${{STAGE5_WEIGHT_MODE:-lora}}"
STAGE5_RL_MODE="${{STAGE5_RL_MODE:-grpo}}"

if [ "$STAGE5_WEIGHT_MODE" = "full" ]; then
  echo "[1/2] Running Stage-5 FULL-WEIGHT 9B SFT (FSDP2)..."
  accelerate launch -m axolotl.cli.train output/axolotl_full_sft.yaml
else
  echo "[1/2] Running Stage-5 LoRA SFT..."
  accelerate launch -m axolotl.cli.train output/axolotl_sft.yaml
  echo "      Merging SFT LoRA into standalone 9B checkpoint..."
  python3 -m axolotl.cli.merge_lora output/axolotl_sft.yaml \\
    --lora-model-dir ./checkpoints/qwen35-9b-sft-lora \\
    --output-dir ./checkpoints/qwen35-9b-sft-merged
fi

if [ "$STAGE5_RL_MODE" = "grpo" ] && [ -s "{rl_path}" ]; then
  GRPO_EXTRA=""
  if [ "$STAGE5_WEIGHT_MODE" = "full" ]; then
    GRPO_EXTRA="--full-weights"
  fi
  echo "[2/2] Running Stage-5 On-Policy GRPO ($STAGE5_WEIGHT_MODE mode)..."
  PYTHONPATH=output accelerate launch output/train_grpo.py \\
    --model ./checkpoints/qwen35-9b-sft-merged \\
    --prompts-jsonl "{rl_path}" \\
    --output-dir "./checkpoints/qwen35-9b-grpo-$STAGE5_WEIGHT_MODE" \\
    $GRPO_EXTRA
elif [ "$STAGE5_RL_MODE" = "dpo" ] && [ -s "{dpo_path}" ]; then
  echo "[2/2] Running Stage-5 Offline DPO refinement..."
  accelerate launch -m axolotl.cli.train output/axolotl_dpo.yaml
else
  echo "[2/2] Skipping RL/DPO refinement (STAGE5_RL_MODE=$STAGE5_RL_MODE)."
fi
"""

    (out_dir / "axolotl_sft.yaml").write_text(sft_yaml, encoding="utf-8")
    (out_dir / "axolotl_full_sft.yaml").write_text(full_sft_yaml, encoding="utf-8")
    (out_dir / "axolotl_dpo.yaml").write_text(dpo_yaml, encoding="utf-8")
    (out_dir / "reward_grpo.py").write_text(REWARD_GRPO_TEMPLATE, encoding="utf-8")
    (out_dir / "train_grpo.py").write_text(TRAIN_GRPO_TEMPLATE, encoding="utf-8")
    script_path = out_dir / "launch_gpu_training.sh"
    script_path.write_text(launch_sh, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def run_stage5_holdout_benchmark(
    out_dir: Path,
    judge: ChatClient,
    candidate_student: ChatClient | None = None,
    baseline_student: ChatClient | None = None,
    workers: int = 16,
    skill_text: str = "",
) -> dict[str, Any]:
    launch_script = write_training_bundle(out_dir)

    holdout_path = out_dir / "holdout_eval.jsonl"
    holdout_rows = [json.loads(ln) for ln in holdout_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    summary: dict[str, Any] = {
        "training_launch_script": str(launch_script),
        "grpo_trainer_script": str(out_dir / "train_grpo.py"),
        "grpo_reward_script": str(out_dir / "reward_grpo.py"),
        "holdout_cases": len(holdout_rows),
        "benchmarked": False,
    }

    if candidate_student is None or not holdout_rows:
        (out_dir / "stage5_benchmark_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def eval_model_on_holdout(model_client: ChatClient) -> tuple[float, dict[str, float], dict[str, float]]:
        dom_b: dict[str, list[float]] = defaultdict(list)
        task_b: dict[str, list[float]] = defaultdict(list)

        def score_item(r: dict[str, Any]) -> tuple[str, str, float]:
            user_msg = r["messages"][0]
            _, ans = model_client.complete([user_msg], temperature=0.2, max_tokens=4096)
            sc = judge_row(judge, user_msg["content"], ans, r["rubric"], skill_text=skill_text)
            return r["domain"], r["task_type"], sc

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for dom, ttype, sc in ex.map(score_item, holdout_rows):
                dom_b[dom].append(sc)
                task_b[ttype].append(sc)

        all_sc = [s for v in dom_b.values() for s in v]
        mean_sc = round(sum(all_sc) / len(all_sc), 4) if all_sc else 0.0
        dom_sc = {d: round(sum(v) / len(v), 4) for d, v in sorted(dom_b.items())}
        tsk_sc = {t: round(sum(v) / len(v), 4) for t, v in sorted(task_b.items())}
        return mean_sc, dom_sc, tsk_sc

    cand_mean, cand_dom, cand_task = eval_model_on_holdout(candidate_student)
    summary.update(
        {
            "benchmarked": True,
            "candidate_model": candidate_student.model,
            "candidate_holdout_mean": cand_mean,
            "candidate_by_domain": cand_dom,
            "candidate_by_task_type": cand_task,
        }
    )

    if baseline_student is not None:
        base_mean, _, _ = eval_model_on_holdout(baseline_student)
        delta = round(cand_mean - base_mean, 4)
        summary["baseline_model"] = baseline_student.model
        summary["baseline_holdout_mean"] = base_mean
        summary["delta_vs_baseline"] = delta
        summary["promote_candidate"] = delta > 0.0

    (out_dir / "stage5_benchmark_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Stage 5 Immutable Holdout Benchmark Report",
        "",
        f"- **Holdout Cases Evaluated:** `{len(holdout_rows)}`",
        f"- **Candidate Model (`{candidate_student.model}`):** `{cand_mean:.4f}`",
    ]
    if "baseline_holdout_mean" in summary:
        md_lines.append(f"- **Baseline Model (`{summary['baseline_model']}`):** `{summary['baseline_holdout_mean']:.4f}`")
        md_lines.append(f"- **Delta vs. Baseline:** `{summary['delta_vs_baseline']:+.4f}` (`promote={summary['promote_candidate']}`)")
    md_lines.extend(["", "## Per-Task-Type Scores"])
    for t, sc in cand_task.items():
        md_lines.append(f"- `{t}`: `{sc:.4f}`")
    (out_dir / "stage5_benchmark_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return summary
