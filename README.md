# Harvey Legal Synthetic Post-Training Pipeline (`Qwen/Qwen3.5-9B`)

Contamination-free 5-stage synthetic legal data generation, rubric verification, and post-training pipeline (SFT, GRPO RL, and DPO) built over 142,959 decontaminated legal documents across 49 practice domains. Runs on standard Python 3.12 with zero external pip dependencies required on Cloudtop.

---

## Stage-by-Stage Overview

### Stage 1: Domain Discovery & Task Config (`stage1_generate_configs.py`)
- **Input:** Unzipped `./harvey_lab_corpus/` folder (or `.zip`), plus optional `output/eval_feedback.json` from a previous round.
- **What it does:** Scans all 49 clean legal domain folders and defines the 4-task synthesis mix (`legal_memo`, `issue_spotting_and_risk_redline`, `compliance_checklist_table`, and `clause_extraction_qa`) along with per-domain sample quotas.
- **Output:** `output/domain_configs.json`.

### Stage 2: Boilerplate Cleaning & Teacher Synthesis (`stage2_synthesize.py`)
- **Input:** `output/domain_configs.json`, `skills/sft_rl_synthesis/SKILL.md`, and `TEACHER_API_KEY` (`gemini-argon-4`).
- **What it does:** Strips web-navigation boilerplate from raw legal files, samples 1-2 substantive documents per task, and runs parallel resumable Teacher calls (`gemini-argon-4` guided by `SKILL.md`) to synthesize closed-world `(task_prompt, rubric, reasoning, answer)` candidates.
- **Output:** `output/raw_candidates.jsonl` (resumable: re-running skips already-generated IDs).

### Stage 3: Deduplication, Judge Verification & Holdout Split (`stage3_clean_and_filter.py`)
- **Input:** `output/raw_candidates.jsonl`, `skills/sft_rl_synthesis/SKILL.md`, and Judge API (reuses `gemini-argon-4` at `temperature=0.0` unless separate `JUDGE_*` keys are set).
- **What it does:** Runs deterministic structural/packet-grounding checks, drops exact and near-duplicate answers (5-gram Jaccard similarity), grades every candidate against its atomic rubric, keeps only verified high-quality rows (`rubric_score >= 0.90`), and splits them deterministically (`md5(row["id"])`):
  - **90% Train:** Written to `output/sft_train.jsonl` (pointwise SFT) and `output/rl_prompts.jsonl` (GRPO / RFT prompts + rubrics).
  - **10% Locked Holdout:** Written **exclusively** to `output/holdout_eval.jsonl` (never included in SFT, RL, or DPO training).
- **Output:** `output/sft_train.jsonl`, `output/rl_prompts.jsonl`, `output/holdout_eval.jsonl`, and `output/filter_stats.json`.

### Stage 4: Student Rollouts, DPO Pairs & Failure Feedback (`stage4_eval_and_feedback.py`)
- **Input:** `output/sft_train.jsonl`, `STUDENT_*` endpoint (`Qwen/Qwen3.5-9B`), and Judge API. *(Skipped when `--skip-student-eval` is passed.)*
- **What it does:** Generates candidate answers from `Qwen/Qwen3.5-9B` on `sft_train.jsonl` prompts (never touching `holdout_eval.jsonl`), grades them with the Judge, creates `(chosen, rejected)` DPO pairs whenever `teacher_score - student_score >= 0.25`, and computes `Qwen`'s average score per legal domain.
- **Output:** `output/dpo_preferences.jsonl` and `output/eval_feedback.json`.

### Stage 5: GPU Training Bundle & Holdout Benchmarking (`stage5_train_and_benchmark.py`)
- **Input:** `output/sft_train.jsonl`, `output/dpo_preferences.jsonl`, and `output/holdout_eval.jsonl`.
- **What it does:** Writes ready-to-run GPU training configs (assistant-only loss masking, SGLang-safe LoRA modules) and a bash launch script. When a Student endpoint is active, it also benchmarks Candidate vs. Baseline models on the untouched `output/holdout_eval.jsonl` split.
- **Output:** `output/axolotl_sft.yaml`, `output/axolotl_dpo.yaml`, `output/launch_gpu_training.sh`, and `output/stage5_benchmark_report.md` (+ `.json`).

---

## How `eval_feedback.json` Works (Round 1 vs. Round 2 Loop)

`output/eval_feedback.json` is the automatic hard-domain scorecard connecting Stage 4 back to Stage 1:

1. **Round 1 (First Run -- `eval_feedback.json` does not exist yet):**
   Stage 1 gives every one of the 49 legal domains the exact same quota (for example, `--samples-per-domain 20` -> `20` tasks per domain). If you only want a single data-generation pass (`--skip-student-eval`), you never need `eval_feedback.json`.
2. **After Stage 4 Runs (`eval_feedback.json` now exists):**
   Stage 4 tests `Qwen/Qwen3.5-9B` and records per-domain scores (e.g., `01_delaware_corporate: 0.88`, `14_tax_controversy: 0.42`).
3. **Round 2 (Re-running `run_pipeline.py`):**
   Stage 1 automatically reads `output/eval_feedback.json` and boosts quotas **only** on weak domains while skipping already-generated IDs:
   - **Domain score `>= 0.70`:** Quota stays `20` (`0` new API calls spent on easy domains).
   - **Domain score `0.50 - 0.69`:** Quota boosted **`1.75x`** (`20 -> 35`, generating `15` extra tasks).
   - **Domain score `< 0.50`:** Quota boosted **`2.5x`** (`20 -> 50`, generating `30` extra tasks).

---

## Supporting Files

- `api_keys.env`: Single config file for `TEACHER_API_KEY`, `TEACHER_MODEL` (`gemini-argon-4`), optional `JUDGE_*`, and optional `STUDENT_*` endpoints (auto-loaded on startup).
- `skills/sft_rl_synthesis/SKILL.md`: Synthesis and grading rules injected into Teacher and Judge system prompts (closed-world packet grounding, multi-clause legal tasks, 4-7 atomic rubric checks for GRPO variance, clean `<think>` reasoning with zero meta-leakage).
- `llm_client.py`: Unified HTTP client supporting native Google Gemini `generateContent` and OpenAI-compatible `/chat/completions` endpoints, plus `--mock` offline testing.
- `run_pipeline.py`: Main CLI orchestrator running Stages 1 through 5.

---

## Running Instructions

### 1. Setup & Connectivity Test
Place the unzipped `harvey_lab_corpus/` folder inside `harvey_synth_pipeline/`, paste your key into `api_keys.env`, and test connectivity:

```bash
cd ~/harvey_synth_pipeline
python3 run_pipeline.py --test-api
```

### 2. Pilot Run (2 Domains x 4 Samples = 8 Tasks)
```bash
python3 run_pipeline.py \
  --limit-domains 2 \
  --samples-per-domain 4 \
  --workers 8 \
  --skip-student-eval
```

### 3. Full 49-Domain SFT + RL Dataset Generation
```bash
python3 run_pipeline.py \
  --samples-per-domain 20 \
  --workers 16 \
  --skip-student-eval
```

### 4. GPU Training + Stage 4 DPO & Stage 5 Holdout Benchmark
1. Run `bash output/launch_gpu_training.sh` on your GPU machine (`axolotl` LoRA SFT / DPO).
2. Set `STUDENT_BASE_URL` in `api_keys.env` to your served `Qwen/Qwen3.5-9B` endpoint and run without `--skip-student-eval`:
   ```bash
   python3 run_pipeline.py --samples-per-domain 20 --workers 16
   ```
