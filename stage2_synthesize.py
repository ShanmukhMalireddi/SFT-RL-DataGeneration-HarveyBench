"""Stage 2: Clean boilerplate from unzipped harvey_lab_corpus/ folder (or .zip) and run parallel Teacher synthesis."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import re
import threading
import zipfile
from pathlib import Path
from typing import Any

from llm_client import ChatClient, extract_json_object

BOILERPLATE_PATTERNS = re.compile(
    r"^(skip to main content|find a lawyer|legal forms|learn about the law|explore legal topics|"
    r"search laws|search case law|search u\.s\. codes|legal professionals|sign in|menu|cookie|privacy policy)$",
    re.IGNORECASE,
)


def clean_legal_document(raw_text: str, max_chars: int = 5500) -> str:
    lines = []
    for raw_line in raw_text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if len(line) < 40 and BOILERPLATE_PATTERNS.match(line):
            continue
        lines.append(line)

    start_idx = 0
    for idx, ln in enumerate(lines):
        if len(ln) >= 80:
            start_idx = idx
            break
    cleaned = "\n".join(lines[start_idx:])
    return cleaned[:max_chars].strip()


def build_teacher_prompt(
    domain: str,
    spec: dict[str, Any],
    docs: list[tuple[str, str]],
    skill_text: str = "",
) -> list[dict[str, str]]:
    doc_blocks = []
    for idx, (doc_name, text) in enumerate(docs, 1):
        doc_blocks.append(f"=== DOCUMENT {idx}: {doc_name} ===\n{text}")
    joined_docs = "\n\n".join(doc_blocks)

    system_msg = (
        "You are an expert legal post-training data architect. Synthesize realistic, rigorous legal "
        "tasks, grading rubrics, step-by-step legal reasoning, and complete target responses strictly "
        "grounded in the provided source documents. Do NOT invent outside facts or citations."
    )
    if skill_text:
        system_msg += f"\n\n=== SKILL INSTRUCTIONS ===\n{skill_text.strip()}"

    user_msg = (
        f"SYNTHESIZE_TASK_JSON\n"
        f"Legal Domain: {domain}\n"
        f"Task Type: {spec['task_type']}\n"
        f"Style Requirement: {spec['style_guide']}\n\n"
        f"Source Packet:\n{joined_docs}\n\n"
        "Return ONLY a valid JSON object with this exact schema:\n"
        "{\n"
        '  "task_prompt": "realistic legal user instruction referencing the packet documents (do not leak the answer)",\n'
        '  "rubric": [\n'
        '    {"criterion": "specific verifiable factual/legal check grounded in the packet", "weight": 1.0}\n'
        "  ],\n"
        '  "reasoning": "thorough step-by-step legal reasoning analyzing the packet",\n'
        '  "answer": "complete, publication-grade final response satisfying every rubric criterion"\n'
        "}"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def sample_jobs_from_dir(
    corpus_dir: Path,
    domains: list[dict[str, Any]],
    done_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    root = corpus_dir / "harvey_lab_corpus" if (corpus_dir / "harvey_lab_corpus").is_dir() else corpus_dir
    jobs: list[dict[str, Any]] = []

    for dom_entry in domains:
        dom = dom_entry["domain"]
        dom_dir = root / dom
        if not dom_dir.is_dir():
            continue
        pool = sorted(f for f in dom_dir.iterdir() if f.is_file())
        if not pool:
            continue
        rng.shuffle(pool)
        cursor = 0
        for sample_idx in range(dom_entry["target_samples"]):
            spec = dom_entry["task_mix"][sample_idx % len(dom_entry["task_mix"])]
            sample_id = f"{dom}__{spec['task_type']}__{sample_idx:04d}"
            if sample_id in done_ids:
                continue

            k = spec["docs_per_sample"]
            picked_docs: list[tuple[str, str]] = []
            tries = 0
            while len(picked_docs) < k and tries < len(pool) * 2:
                fpath = pool[cursor % len(pool)]
                cursor += 1
                tries += 1
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                cleaned = clean_legal_document(raw)
                if len(cleaned) >= 500:
                    picked_docs.append((fpath.name, cleaned))

            if picked_docs:
                jobs.append({"id": sample_id, "domain": dom, "spec": spec, "docs": picked_docs})
    return jobs


def sample_jobs_from_zip(
    corpus_zip: Path,
    domains: list[dict[str, Any]],
    done_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with zipfile.ZipFile(corpus_zip) as zf:
        by_domain: dict[str, list[str]] = {}
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) >= 3:
                by_domain.setdefault(parts[1], []).append(name)

        for dom_entry in domains:
            dom = dom_entry["domain"]
            pool = sorted(by_domain.get(dom, []))
            if not pool:
                continue
            rng.shuffle(pool)
            cursor = 0
            for sample_idx in range(dom_entry["target_samples"]):
                spec = dom_entry["task_mix"][sample_idx % len(dom_entry["task_mix"])]
                sample_id = f"{dom}__{spec['task_type']}__{sample_idx:04d}"
                if sample_id in done_ids:
                    continue

                k = spec["docs_per_sample"]
                picked_docs: list[tuple[str, str]] = []
                tries = 0
                while len(picked_docs) < k and tries < len(pool) * 2:
                    member = pool[cursor % len(pool)]
                    cursor += 1
                    tries += 1
                    raw = zf.read(member).decode("utf-8", errors="replace")
                    cleaned = clean_legal_document(raw)
                    if len(cleaned) >= 500:
                        picked_docs.append((Path(member).name, cleaned))

                if picked_docs:
                    jobs.append({"id": sample_id, "domain": dom, "spec": spec, "docs": picked_docs})
    return jobs


def run_stage2_synthesis(
    config_path: Path,
    out_jsonl: Path,
    teacher: ChatClient,
    workers: int = 16,
    seed: int = 42,
    limit_domains: int | None = None,
    skill_text: str = "",
) -> int:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = Path(cfg.get("corpus_path") or cfg["corpus_zip"])
    rng = random.Random(seed)

    done_ids: set[str] = set()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        for ln in out_jsonl.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                done_ids.add(json.loads(ln)["id"])

    domains = cfg["domains"][:limit_domains] if limit_domains else cfg["domains"]
    if corpus_path.is_dir():
        jobs = sample_jobs_from_dir(corpus_path, domains, done_ids, rng)
    else:
        jobs = sample_jobs_from_zip(corpus_path, domains, done_ids, rng)

    lock = threading.Lock()
    written = 0

    def process_job(job: dict[str, Any]) -> dict[str, Any] | None:
        msgs = build_teacher_prompt(job["domain"], job["spec"], job["docs"], skill_text=skill_text)
        try:
            _, raw_out = teacher.complete(msgs, temperature=job["spec"]["temperature"], max_tokens=4096)
            parsed = extract_json_object(raw_out)
            prompt_txt = str(parsed.get("task_prompt", "")).strip()
            answer_txt = str(parsed.get("answer", "")).strip()
            reasoning_txt = str(parsed.get("reasoning", "")).strip()
            rubric = parsed.get("rubric", [])
            if not prompt_txt or not answer_txt or not isinstance(rubric, list) or not rubric:
                return None

            packet_block = "\n\n".join(f"=== {name} ===\n{txt}" for name, txt in job["docs"])
            full_user = f"{prompt_txt}\n\n---\nSOURCE PACKET:\n{packet_block}"
            content_hash = hashlib.sha256((prompt_txt + "\n" + answer_txt).encode("utf-8")).hexdigest()[:16]

            return {
                "id": job["id"],
                "sha16": content_hash,
                "domain": job["domain"],
                "task_type": job["spec"]["task_type"],
                "source_docs": [name for name, _ in job["docs"]],
                "messages": [
                    {"role": "user", "content": full_user},
                    {
                        "role": "assistant",
                        "reasoning_content": reasoning_txt,
                        "content": answer_txt,
                    },
                ],
                "rubric": rubric,
            }
        except Exception:
            return None

    with out_jsonl.open("a", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(process_job, jobs):
                if res is not None:
                    with lock:
                        fout.write(json.dumps(res, ensure_ascii=True) + "\n")
                        fout.flush()
                        written += 1

    return written
