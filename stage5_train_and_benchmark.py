"""Stage 5: Emit SFT/DPO GPU training configs & launch script, and benchmark Candidate vs. Base on immutable holdout."""

from __future__ import annotations

import concurrent.futures
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_client import ChatClient
from stage3_clean_and_filter import judge_row


def write_training_bundle(out_dir: Path) -> Path:
    sft_path = (out_dir / "sft_train.jsonl").resolve()
    dpo_path = (out_dir / "dpo_preferences.jsonl").resolve()

    sft_yaml = f"""# Stage 5 Generated Axolotl LoRA SFT Config (Qwen/Qwen3.5-9B)
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

    dpo_yaml = f"""# Stage 5 Generated Axolotl LoRA DPO Config (Run on SFT-merged checkpoint)
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

    launch_sh = """#!/usr/bin/env bash
set -euo pipefail
# Run on a GPU machine (e.g. 2x-8x A100/H100/H200) with axolotl + vllm/sglang installed:

echo "[1/3] Running Stage-5 LoRA SFT..."
accelerate launch -m axolotl.cli.train output/axolotl_sft.yaml

echo "[2/3] Merging SFT LoRA into base model..."
python3 -m axolotl.cli.merge_lora output/axolotl_sft.yaml --lora-model-dir ./checkpoints/qwen35-9b-sft-lora --output-dir ./checkpoints/qwen35-9b-sft-merged

if [ -s output/dpo_preferences.jsonl ]; then
  echo "[3/3] Optional Stage-5 DPO refinement..."
  accelerate launch -m axolotl.cli.train output/axolotl_dpo.yaml
fi
"""

    (out_dir / "axolotl_sft.yaml").write_text(sft_yaml, encoding="utf-8")
    (out_dir / "axolotl_dpo.yaml").write_text(dpo_yaml, encoding="utf-8")
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

    # Write JSON + clean Markdown benchmark summary
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
    md_lines.extend(["", "## Per-Task-Type Scores", "| Task Type | Score |", "| :--- | :---: |"])
    for t, sc in cand_task.items():
        md_lines.append(f"| `{t}` | `{sc:.4f}` |")
    (out_dir / "stage5_benchmark_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return summary
