# Harvey Legal Synthetic Post-Training Pipeline (`Qwen/Qwen3.5-9B`)

Contamination-free 5-stage synthetic legal data generation, rubric verification, and post-training pipeline (SFT, GRPO RL, and DPO) built over 142,959 decontaminated legal documents across 49 practice domains. Runs on standard Python 3.12 with zero external pip dependencies required on Cloudtop.

---

## Stage-by-Stage Overview

### Stage 1: Domain Discovery & Task Quotas (`stage1_generate_configs.py`)
- **Input:** Unzipped `./harvey_lab_corpus/` folder (or `.zip`), plus optional `output/eval_feedback.json` from a previous round.
- **What it does:** Scans all 49 clean legal domain folders, registers the 4 core legal task formats (`legal_memo`, `issue_spotting_and_risk_redline`, `compliance_checklist_table`, and `clause_extraction_qa`), and computes how many samples (`target_samples` quota) to create inside each domain.
- **Output:** `output/domain_configs.json`.

### Stage 2: Content-Aware Task Matching & Teacher Synthesis (`stage2_synthesize.py`)
- **Input:** `output/domain_configs.json`, `skills/sft_rl_synthesis/SKILL.md`, and `TEACHER_API_KEY` (`gemini-argon-4`).
- **What it does:**
  1. **Strips boilerplate:** Removes web-scrape navigation headers from raw legal `.txt` files.
  2. **Classifies document kind before assigning a task type:** Inspects each sampled document first so task formats always match the document's nature (and balances counts across compatible formats within each domain):
     - *Contracts & Agreements:* Eligible for all 4 formats (`issue_spotting_and_risk_redline`, `legal_memo`, `compliance_checklist_table`, `clause_extraction_qa`).
     - *Judicial Opinions (`court_*`, case holdings):* Matched only to `legal_memo` or `clause_extraction_qa` (never forced into contract redlines).
     - *Statutes, Regulations & Tax Instructions:* Matched to `compliance_checklist_table`, `legal_memo`, or `clause_extraction_qa`.
     - *Teacher override (`selected_task_type`):* `gemini-argon-4` can also switch to another compatible task type if the specific excerpt lacks structure for the preferred type.
  3. **Parallel Teacher synthesis:** Calls `gemini-argon-4` (with `SKILL.md` injected) across multi-threaded workers to generate `(task_prompt, rubric, reasoning, answer)` strictly grounded in the 1-2 sampled documents.
- **Output:** `output/raw_candidates.jsonl` (resumable: re-running skips already-generated case slots).

### Stage 3: Deduplication, Judge Verification & Holdout Split (`stage3_clean_and_filter.py`)
- **Input:** `output/raw_candidates.jsonl`, `skills/sft_rl_synthesis/SKILL.md`, and Judge API (automatically reuses `gemini-argon-4` at `temperature=0.0` unless separate `JUDGE_*` keys are set).
- **What it does:** Enforces structural and packet-grounding checks, removes exact and near-duplicate samples (5-gram Jaccard similarity), grades every candidate against its atomic rubric, keeps only verified rows (`rubric_score >= 0.90`), and splits them deterministically (`md5(row["id"])`):
  - **90% Train:** Written to `output/sft_train.jsonl` (pointwise SFT) and `output/rl_prompts.jsonl` (GRPO / RFT prompts + rubrics).
  - **10% Locked Holdout:** Written **exclusively** to `output/holdout_eval.jsonl` (never included in SFT, RL, or DPO training).
- **Output:** `output/sft_train.jsonl`, `output/rl_prompts.jsonl`, `output/holdout_eval.jsonl`, and `output/filter_stats.json`.

### Stage 4: Student Rollouts, DPO Pairs & Failure Feedback (`stage4_eval_and_feedback.py`)
- **Input:** `output/sft_train.jsonl`, `STUDENT_*` endpoint (`Qwen/Qwen3.5-9B`), and Judge API. *(Skipped when `--skip-student-eval` is passed.)*
- **What it does:** Generates candidate answers from `Qwen/Qwen3.5-9B` on `sft_train.jsonl` prompts (never touching `holdout_eval.jsonl`), grades them with the Judge, creates `(chosen, rejected)` DPO pairs whenever `teacher_score - student_score >= 0.25`, and computes `Qwen`'s average score per legal domain.
- **Output:** `output/dpo_preferences.jsonl` and `output/eval_feedback.json`.

### Stage 5: GPU Training Bundle (LoRA or Full-Weight SFT + GRPO + DPO) & Holdout Benchmarking (`stage5_train_and_benchmark.py`)
- **Input:** `output/sft_train.jsonl`, `output/rl_prompts.jsonl`, `output/dpo_preferences.jsonl`, and `output/holdout_eval.jsonl`.
- **What it does:**
  1. **Writes both LoRA and Full-Weight GPU configs:** Generates `output/axolotl_sft.yaml` (LoRA SFT on `1x-2x` GPUs, `lr=1e-4`), `output/axolotl_full_sft.yaml` (full 9B-parameter FSDP2 SFT on `4x-8x` 80GB GPUs, `lr=1e-5`), `output/axolotl_dpo.yaml` (offline DPO), `output/reward_grpo.py` (100% offline deterministic or Judge-API GRPO reward compatible with `TRL` and `miles`), and `output/train_grpo.py` (supports both LoRA at `lr=2e-7` and `--full-weights` at `lr=1e-7`).
  2. **Benchmarks on locked holdout:** When a Student endpoint is active, evaluates Candidate vs. Baseline models on the untouched `output/holdout_eval.jsonl` split.
- **Output:** `output/axolotl_sft.yaml`, `output/axolotl_full_sft.yaml`, `output/reward_grpo.py`, `output/train_grpo.py`, `output/axolotl_dpo.yaml`, `output/launch_gpu_training.sh`, and `output/stage5_benchmark_report.md` (+ `.json`).

---

## How Domains, `--samples-per-domain`, and Quota Boosting Connect

### 1. Domains x Cases Per Domain
- `--limit-domains D` chooses how many domain folders to run (default = all `49`).
- `--samples-per-domain N` sets the base quota of `N` cases per domain (`Total Round-1 cases = D x N`).
- Inside each domain, Stage 2 balances those `N` cases evenly across whichever of the 4 task types are compatible with the sampled documents in that folder.

### 2. Quota Boosting via `eval_feedback.json` (Round 1 vs. Round 2)
`--samples-per-domain N` sets the **base** count (`base = N`), and Stage 1 multiplies it by a feedback multiplier (`quota = base * multiplier`):

- **Round 1 (no `eval_feedback.json` yet):** `multiplier = 1.0` everywhere. Passing `--samples-per-domain 20` creates `20` cases in every domain. (If you only run a single pass with `--skip-student-eval`, `eval_feedback.json` is never needed.)
- **Round 2 (after Stage 4 writes `output/eval_feedback.json`):** Even if you pass the exact same `--samples-per-domain 20` command again, Stage 1 automatically boosts weak domains while Stage 2 skips the `0..19` cases already generated:
  - **Strong domain (`score >= 0.70`, e.g. `0.90`):** `20 * 1.0 = 20` -> `0` new API calls wasted on an easy domain.
  - **Medium-weak domain (`0.50 - 0.69`):** `20 * 1.75 = 35` -> generates **`15` new** training cases (`case_0020..0034`).
  - **Very weak domain (`score < 0.50`, e.g. `0.40`):** `20 * 2.5 = 50` -> generates **`30` new** training cases (`case_0020..0049`).

---

## Supporting Files

- `api_keys.env`: Single config file for `TEACHER_API_KEY`, `TEACHER_MODEL` (`gemini-argon-4`), optional `JUDGE_*`, and optional `STUDENT_*` endpoints (auto-loaded on startup).
- `skills/sft_rl_synthesis/SKILL.md`: Synthesis and grading rules injected into Teacher and Judge system prompts (closed-world packet grounding, content-aware task matching, 4-7 atomic rubric checks for GRPO variance, clean `<think>` reasoning with zero meta-leakage).
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

### 4. GPU Training (`LoRA` vs. `Full-Weight`, `GRPO` vs. `DPO`) & Holdout Benchmark
1. On your GPU machine, `launch_gpu_training.sh` supports `STAGE5_WEIGHT_MODE=lora|full` and `STAGE5_RL_MODE=grpo|dpo|none`:
   ```bash
   # Default: LoRA SFT -> Merge -> LoRA On-Policy GRPO (1x-2x GPUs)
   bash output/launch_gpu_training.sh

   # Full 9B Weights: FSDP2 Full SFT -> Full-Weight On-Policy GRPO (4x-8x 80GB GPUs)
   STAGE5_WEIGHT_MODE=full bash output/launch_gpu_training.sh

   # Offline DPO instead of On-Policy GRPO:
   STAGE5_RL_MODE=dpo bash output/launch_gpu_training.sh
   ```
2. Set `STUDENT_BASE_URL` in `api_keys.env` to your served `Qwen/Qwen3.5-9B` endpoint and run without `--skip-student-eval`:
   ```bash
   python3 run_pipeline.py --samples-per-domain 20 --workers 16
   ```
