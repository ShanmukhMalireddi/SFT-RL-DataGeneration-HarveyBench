# Harvey Legal Synthetic Post-Training Pipeline (`Qwen/Qwen3.5-9B`)

Contamination-free 5-stage synthetic legal data generation, rubric verification, and post-training pipeline (SFT, GRPO RL, and DPO) built over 142,959 decontaminated legal documents across 49 practice domains. Zero external pip dependencies required on Cloudtop (standard Python 3.12).

---

## Architecture: What Each Stage Does

| Stage / File | Inputs | What It Does | Outputs |
| :--- | :--- | :--- | :--- |
| **Stage 1**<br>`stage1_generate_configs.py` | `./harvey_lab_corpus/` (or `.zip`), optional `eval_feedback.json` | Scans all 49 clean legal domains, sets task-type mixes (`legal_memo`, `issue_spotting_and_risk_redline`, `compliance_checklist_table`, `clause_extraction_qa`), and automatically boosts target quotas by `1.75x-2.5x` on weak domains (`score < 0.70`) from prior rounds. | `output/domain_configs.json` |
| **Stage 2**<br>`stage2_synthesize.py` | `domain_configs.json`, `SKILL.md`, Teacher API (`gemini-argon-4`) | Strips web-scrape boilerplate from raw legal files, samples 1-2 substantive documents per task, and runs multi-threaded resumable Teacher calls to synthesize closed-world `(task_prompt, rubric, reasoning, answer)` candidates. | `output/raw_candidates.jsonl` |
| **Stage 3**<br>`stage3_clean_and_filter.py` | `raw_candidates.jsonl`, `SKILL.md`, Judge API | Enforces structural and packet-grounding checks, removes exact and near-duplicate samples (5-gram Jaccard), grades every candidate against its atomic rubric (`temperature=0.0`), keeps only rows with `rubric_score >= 0.90`, and locks an untouched 10% holdout split. | `output/sft_train.jsonl`<br>`output/rl_prompts.jsonl`<br>`output/holdout_eval.jsonl`<br>`output/filter_stats.json` |
| **Stage 4**<br>`stage4_eval_and_feedback.py` | `sft_train.jsonl`, Student API (`Qwen/Qwen3.5-9B`), Judge API | Rolls out the Student model on training prompts, pairs Teacher vs. Student answers into `(chosen, rejected)` preference rows whenever `chosen_score - rejected_score >= 0.25`, and records per-domain weakness scores to drive Round-2 Stage-1 sampling. | `output/dpo_preferences.jsonl`<br>`output/eval_feedback.json` |
| **Stage 5**<br>`stage5_train_and_benchmark.py` | `holdout_eval.jsonl`, `sft_train.jsonl`, `dpo_preferences.jsonl` | Generates ready-to-run GPU training configs (assistant-only loss masking, SGLang-safe LoRA modules) plus a launch script, and benchmarks Candidate vs. Baseline models on the untouched `holdout_eval.jsonl` split. | `output/axolotl_sft.yaml`<br>`output/axolotl_dpo.yaml`<br>`output/launch_gpu_training.sh`<br>`output/stage5_benchmark_report.md` |

### Supporting Files
- `skills/sft_rl_synthesis/SKILL.md`: Prompting & grading rules injected into Teacher and Judge system prompts (enforces closed-world packet grounding, bans low-signal summary tasks, calibrates 4-7 atomic rubric items for GRPO variance, and prevents meta-leakage in `<think>` traces).
- `api_keys.env`: Single config file for `TEACHER_API_KEY`, `TEACHER_MODEL` (`gemini-argon-4`), optional `JUDGE_*`, and optional `STUDENT_*` endpoints (auto-loaded on startup).
- `llm_client.py`: Unified HTTP client supporting both native Google Gemini `generateContent` and OpenAI-compatible `/chat/completions` endpoints, plus `--mock` offline testing.
- `run_pipeline.py`: End-to-end CLI entrypoint orchestrating Stages 1 through 5.

### Train vs. Holdout Split (`output/holdout_eval.jsonl`)
- Created in **Stage 3** (`stage3_clean_and_filter.py`) immediately after deduplication and Judge verification (`rubric_score >= 0.90`).
- Uses a deterministic hash on each verified sample ID (`md5(row["id"])`):
  - **`10%` (`bucket < 0.10`)** are written **exclusively** to `output/holdout_eval.jsonl` (never included in `sft_train.jsonl` or `rl_prompts.jsonl`).
  - **`90%`** are written to `output/sft_train.jsonl` and `output/rl_prompts.jsonl`.
- **Stage 4** (`stage4_eval_and_feedback.py`) builds `output/dpo_preferences.jsonl` and domain weakness feedback strictly from `sft_train.jsonl`, leaving `holdout_eval.jsonl` completely untouched by training or Round-2 curriculum selection.
- **Stage 5** (`stage5_train_and_benchmark.py`) evaluates Candidate vs. Baseline models on `output/holdout_eval.jsonl` and writes `output/stage5_benchmark_report.md` + `.json`.

---

## How to Run

### 1. Configure `api_keys.env` & Test Connection
Place the unzipped `harvey_lab_corpus/` folder inside `harvey_synth_pipeline/` (auto-detected by default), paste your key into `api_keys.env`, and test connectivity:

```bash
cd ~/harvey_synth_pipeline
python3 run_pipeline.py --test-api
```

### 2. Quick Pilot Run (2 Domains x 4 Samples = 8 Tasks)
```bash
python3 run_pipeline.py \
  --limit-domains 2 \
  --samples-per-domain 4 \
  --workers 8 \
  --skip-student-eval
```

### 3. Full 49-Domain SFT + RL Dataset Generation (Stages 1-3 + Stage 5 Configs)
```bash
python3 run_pipeline.py \
  --samples-per-domain 20 \
  --workers 16 \
  --skip-student-eval
```
*(Stage 2 is resumable: if interrupted, re-running skips already-generated IDs in `output/raw_candidates.jsonl`.)*

### 4. GPU Training + Stage 4 DPO & Stage 5 Holdout Benchmark
1. Run `bash output/launch_gpu_training.sh` on your GPU machine (`axolotl` LoRA SFT / DPO).
2. Serve `Qwen/Qwen3.5-9B` (base or fine-tuned) on `STUDENT_BASE_URL` in `api_keys.env` and run without `--skip-student-eval`:
   ```bash
   python3 run_pipeline.py --samples-per-domain 20 --workers 16
   ```
   Re-running `run_pipeline.py` afterward automatically reads `output/eval_feedback.json` in Stage 1 to oversample any domain scoring `< 0.70`.
