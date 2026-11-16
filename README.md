# 5-Stage Synthetic Data Pipeline for `Qwen/Qwen3.5-9B` (Configured for `gemini-argon-4`)

Zero external pip dependencies required (runs on standard Python 3.12 on your Cloudtop across 24 CPU cores) and reads directly from `./harvey_lab_corpus/` (unzipped folder inside `harvey_synth_pipeline/`) or `harvey_lab_corpus.zip`.

Supports **both** native Google Gemini `generateContent` endpoints (`https://generativelanguage.googleapis.com/v1beta`) and OpenAI-compatible `/chat/completions` endpoints automatically, and **automatically reuses your `TEACHER_API_KEY` + `gemini-argon-4` as the Stage-3 Judge** at `temperature=0.0` if no separate `JUDGE_API_KEY` is set.

## 1. Set Your Key in `api_keys.env` & Test Connection
Open `api_keys.env` in VS Code and replace `PASTE_YOUR_TEACHER_API_KEY_HERE` with your key (`TEACHER_MODEL="gemini-argon-4"` is already set). `run_pipeline.py` loads `api_keys.env` automatically -- no terminal `export` needed:

```bash
cd ~/harvey_synth_pipeline
python3 run_pipeline.py --test-api
```

## 2. Pilot Run (2 Domains x 4 Samples = 8 Tasks First)
```bash
python3 run_pipeline.py \
  --limit-domains 2 \
  --samples-per-domain 4 \
  --workers 8 \
  --skip-student-eval
```

## 3. Full 49-Domain Run (`sft_train.jsonl` + `rl_prompts.jsonl` + `holdout_eval.jsonl`)
```bash
python3 run_pipeline.py \
  --samples-per-domain 20 \
  --workers 16 \
  --skip-student-eval
```
*(Note: Stage 2 is resumable -- if interrupted, re-running skips already-generated IDs in `output/raw_candidates.jsonl`.)*
