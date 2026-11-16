"""Stage 2: Clean boilerplate, classify document kind, match compatible task type, and run parallel Teacher synthesis."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import re
import threading
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from llm_client import ChatClient, extract_json_object

BOILERPLATE_PATTERNS = re.compile(
    r"^(skip to main content|find a lawyer|legal forms|learn about the law|explore legal topics|"
    r"search laws|search case law|search u\.s\. codes|legal professionals|sign in|menu|cookie|privacy policy)$",
    re.IGNORECASE,
)

OPINION_MARKERS = re.compile(
    r"\b(appellant|appellee|petitioner|respondent|plaintiff|defendant|court of chancery|"
    r"supreme court|court of appeals|memorandum opinion|affirmed|reversed|remanded)\b",
    re.IGNORECASE,
)
CONTRACT_MARKERS = re.compile(
    r"\b(whereas|witnesseth|hereinafter|indemnif\w+|representations and warranties|"
    r"governing law|counterparts|entire agreement|this agreement|credit agreement|merger agreement|lease)\b",
    re.IGNORECASE,
)
STATUTE_REG_MARKERS = re.compile(
    r"\b(u\.s\.c\.|c\.f\.r\.|subchapter|subsection|promulgated|rulemaking|instructions for|"
    r"filing fee|statutory|franchise tax|delaware code|internal revenue code)\b",
    re.IGNORECASE,
)

COMPATIBLE_TASKS_BY_DOC_KIND: dict[str, list[str]] = {
    "contract_or_agreement": [
        "issue_spotting_and_risk_redline",
        "legal_memo",
        "compliance_checklist_table",
        "clause_extraction_qa",
    ],
    "judicial_opinion": [
        "legal_memo",
        "clause_extraction_qa",
    ],
    "statute_or_regulation": [
        "compliance_checklist_table",
        "legal_memo",
        "clause_extraction_qa",
    ],
    "general_legal": [
        "legal_memo",
        "clause_extraction_qa",
        "compliance_checklist_table",
    ],
}


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


def classify_doc_kind(filename: str, cleaned_text: str) -> str:
    fname = filename.lower()
    if fname.startswith("court_") or "opinion" in fname:
        return "judicial_opinion"

    op_hits = len(OPINION_MARKERS.findall(cleaned_text))
    ct_hits = len(CONTRACT_MARKERS.findall(cleaned_text))
    st_hits = len(STATUTE_REG_MARKERS.findall(cleaned_text))

    if ct_hits >= 3 and ct_hits >= op_hits:
        return "contract_or_agreement"
    if op_hits >= 4 and op_hits > ct_hits:
        return "judicial_opinion"
    if st_hits >= 3:
        return "statute_or_regulation"
    return "general_legal"


def select_compatible_spec(
    doc_kind: str,
    task_mix: list[dict[str, Any]],
    type_counts: Counter[str],
) -> dict[str, Any]:
    allowed_names = set(COMPATIBLE_TASKS_BY_DOC_KIND.get(doc_kind, COMPATIBLE_TASKS_BY_DOC_KIND["general_legal"]))
    candidates = [sp for sp in task_mix if sp["task_type"] in allowed_names]
    if not candidates:
        candidates = task_mix
    # Pick whichever compatible task type has been used least so far in this domain
    candidates.sort(key=lambda sp: (type_counts[sp["task_type"]], sp["task_type"]))
    chosen = candidates[0]
    type_counts[chosen["task_type"]] += 1
    return chosen


def build_teacher_prompt(
    domain: str,
    doc_kind: str,
    spec: dict[str, Any],
    allowed_task_types: list[str],
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

    allowed_str = ", ".join(allowed_task_types)
    user_msg = (
        f"SYNTHESIZE_TASK_JSON\n"
        f"Legal Domain: {domain}\n"
        f"Detected Source Document Kind: {doc_kind}\n"
        f"Preferred Task Type: {spec['task_type']}\n"
        f"Compatible Task Types for This Document: [{allowed_str}]\n"
        f"Style Requirement: {spec['style_guide']}\n"
        "(Note: Use Preferred Task Type unless the document excerpt clearly lacks the structure for it, "
        "in which case pick the best-fitting type from Compatible Task Types and set selected_task_type accordingly.)\n\n"
        f"Source Packet:\n{joined_docs}\n\n"
        "Return ONLY a valid JSON object with this exact schema:\n"
        "{\n"
        f'  "selected_task_type": "{spec["task_type"]}",\n'
        '  "task_prompt": "realistic legal user instruction referencing the packet documents (do not leak the answer)",\n'
        '  "rubric": [\n'
        '    {"criterion": "specific verifiable factual/legal check grounded in the packet", "weight": 1.0}\n'
        "  ],\n"
        '  "reasoning": "thorough step-by-step legal reasoning analyzing the packet",\n'
        '  "answer": "complete, publication-grade final response satisfying every rubric criterion"\n'
        "}"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def _build_domain_jobs(
    dom: str,
    target_samples: int,
    task_mix: list[dict[str, Any]],
    doc_reader: list[tuple[str, Any]],
    done_prefixes: set[str],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    cursor = 0
    n_pool = len(doc_reader)
    if n_pool == 0:
        return jobs

    for sample_idx in range(target_samples):
        slot_prefix = f"{dom}__case_{sample_idx:04d}"
        if slot_prefix in done_prefixes:
            continue

        # Sample primary doc first so we classify its real legal document kind before picking task_type
        primary_doc: tuple[str, str] | None = None
        doc_kind = "general_legal"
        tries = 0
        while primary_doc is None and tries < n_pool * 2:
            doc_name, loader = doc_reader[cursor % n_pool]
            cursor += 1
            tries += 1
            cleaned = clean_legal_document(loader())
            if len(cleaned) >= 500:
                primary_doc = (doc_name, cleaned)
                doc_kind = classify_doc_kind(doc_name, cleaned)

        if primary_doc is None:
            continue

        spec = select_compatible_spec(doc_kind, task_mix, type_counts)
        picked_docs = [primary_doc]

        # Sample 2nd doc if spec requests 2 docs and pool has more files
        while len(picked_docs) < spec["docs_per_sample"] and tries < n_pool * 2:
            doc_name, loader = doc_reader[cursor % n_pool]
            cursor += 1
            tries += 1
            if doc_name == primary_doc[0]:
                continue
            cleaned = clean_legal_document(loader())
            if len(cleaned) >= 500:
                picked_docs.append((doc_name, cleaned))

        allowed_types = COMPATIBLE_TASKS_BY_DOC_KIND.get(doc_kind, COMPATIBLE_TASKS_BY_DOC_KIND["general_legal"])
        jobs.append(
            {
                "id": f"{slot_prefix}__{spec['task_type']}",
                "slot_prefix": slot_prefix,
                "domain": dom,
                "doc_kind": doc_kind,
                "spec": spec,
                "allowed_types": allowed_types,
                "docs": picked_docs,
            }
        )
    return jobs


def sample_jobs_from_dir(
    corpus_dir: Path,
    domains: list[dict[str, Any]],
    done_prefixes: set[str],
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
        rng.shuffle(pool)
        readers = [(f.name, lambda p=f: p.read_text(encoding="utf-8", errors="replace")) for f in pool]
        jobs.extend(_build_domain_jobs(dom, dom_entry["target_samples"], dom_entry["task_mix"], readers, done_prefixes))
    return jobs


def sample_jobs_from_zip(
    corpus_zip: Path,
    domains: list[dict[str, Any]],
    done_prefixes: set[str],
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
            rng.shuffle(pool)
            readers = [(Path(m).name, lambda mem=m: zf.read(mem).decode("utf-8", errors="replace")) for m in pool]
            jobs.extend(_build_domain_jobs(dom, dom_entry["target_samples"], dom_entry["task_mix"], readers, done_prefixes))
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

    done_prefixes: set[str] = set()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        for ln in out_jsonl.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                row = json.loads(ln)
                rid = str(row["id"])
                done_prefixes.add(rid)
                # Also record <domain>__case_NNNN prefix so resumed runs never duplicate a case slot
                parts = rid.split("__")
                if len(parts) >= 2 and parts[1].startswith("case_"):
                    done_prefixes.add(f"{parts[0]}__{parts[1]}")

    domains = cfg["domains"][:limit_domains] if limit_domains else cfg["domains"]
    if corpus_path.is_dir():
        jobs = sample_jobs_from_dir(corpus_path, domains, done_prefixes, rng)
    else:
        jobs = sample_jobs_from_zip(corpus_path, domains, done_prefixes, rng)

    lock = threading.Lock()
    written = 0

    def process_job(job: dict[str, Any]) -> dict[str, Any] | None:
        msgs = build_teacher_prompt(
            domain=job["domain"],
            doc_kind=job["doc_kind"],
            spec=job["spec"],
            allowed_task_types=job["allowed_types"],
            docs=job["docs"],
            skill_text=skill_text,
        )
        try:
            _, raw_out = teacher.complete(msgs, temperature=job["spec"]["temperature"], max_tokens=4096)
            parsed = extract_json_object(raw_out)
            prompt_txt = str(parsed.get("task_prompt", "")).strip()
            answer_txt = str(parsed.get("answer", "")).strip()
            reasoning_txt = str(parsed.get("reasoning", "")).strip()
            rubric = parsed.get("rubric", [])
            final_task_type = str(parsed.get("selected_task_type") or job["spec"]["task_type"]).strip()
            if final_task_type not in job["allowed_types"]:
                final_task_type = job["spec"]["task_type"]

            if not prompt_txt or not answer_txt or not isinstance(rubric, list) or not rubric:
                return None

            packet_block = "\n\n".join(f"=== {name} ===\n{txt}" for name, txt in job["docs"])
            full_user = f"{prompt_txt}\n\n---\nSOURCE PACKET:\n{packet_block}"
            content_hash = hashlib.sha256((prompt_txt + "\n" + answer_txt).encode("utf-8")).hexdigest()[:16]

            return {
                "id": f"{job['slot_prefix']}__{final_task_type}",
                "sha16": content_hash,
                "domain": job["domain"],
                "doc_kind": job["doc_kind"],
                "task_type": final_task_type,
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
