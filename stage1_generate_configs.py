"""Stage 1: Discover clean domains in unzipped harvey_lab_corpus/ folder (or .zip) and build task configs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

TASK_SPECS: list[dict[str, Any]] = [
    {
        "task_type": "legal_memo",
        "temperature": 0.4,
        "docs_per_sample": 2,
        "style_guide": "Draft a formal multi-section legal memorandum (Issue, Brief Answer, Controlling Provisions, Analysis, Conclusion) grounded strictly in the provided documents.",
    },
    {
        "task_type": "issue_spotting_and_risk_redline",
        "temperature": 0.4,
        "docs_per_sample": 2,
        "style_guide": "Identify material legal risks, ambiguities, or non-standard provisions in the packet and provide concrete redline replacement language for each issue.",
    },
    {
        "task_type": "compliance_checklist_table",
        "temperature": 0.25,
        "docs_per_sample": 2,
        "style_guide": "Produce a Markdown schedule/table mapping every statutory or contractual requirement, deadline, responsible party, and citation from the packet.",
    },
    {
        "task_type": "clause_extraction_qa",
        "temperature": 0.2,
        "docs_per_sample": 1,
        "style_guide": "Ask a precise legal question answerable from a specific clause/holding in the text, requiring step-by-step reasoning and an exact cited conclusion.",
    },
]


def count_domain_files(corpus_path: Path) -> dict[str, int]:
    domain_files: dict[str, int] = {}
    if corpus_path.is_dir():
        # Support either <corpus_dir>/<domain>/<file> or <corpus_dir>/harvey_lab_corpus/<domain>/<file>
        root = corpus_path / "harvey_lab_corpus" if (corpus_path / "harvey_lab_corpus").is_dir() else corpus_path
        for dom_dir in sorted(root.iterdir()):
            if dom_dir.is_dir() and not dom_dir.name.startswith("."):
                cnt = sum(1 for f in dom_dir.iterdir() if f.is_file())
                if cnt > 0:
                    domain_files[dom_dir.name] = cnt
    else:
        with zipfile.ZipFile(corpus_path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                parts = name.split("/")
                if len(parts) >= 3:
                    dom = parts[1]
                    domain_files[dom] = domain_files.get(dom, 0) + 1
    return domain_files


def build_domain_configs(
    corpus_path: Path,
    out_path: Path,
    base_samples_per_domain: int = 8,
    feedback_path: Path | None = None,
) -> dict[str, Any]:
    feedback_scores: dict[str, float] = {}
    if feedback_path and feedback_path.exists():
        fb = json.loads(feedback_path.read_text(encoding="utf-8"))
        feedback_scores = fb.get("domain_scores", {})

    domain_files = count_domain_files(corpus_path)

    domains_cfg = []
    for dom in sorted(domain_files):
        prev_score = feedback_scores.get(dom)
        multiplier = 1.0
        if prev_score is not None:
            if prev_score < 0.50:
                multiplier = 2.5
            elif prev_score < 0.70:
                multiplier = 1.75
        quota = max(1, round(base_samples_per_domain * multiplier))
        domains_cfg.append(
            {
                "domain": dom,
                "clean_doc_count": domain_files[dom],
                "prev_holdout_score": prev_score,
                "target_samples": quota,
                "task_mix": TASK_SPECS,
            }
        )

    config = {
        "pipeline": "harvey_lab_fireworks_5stage_v1",
        "corpus_path": str(corpus_path),
        "corpus_mode": "directory" if corpus_path.is_dir() else "zip",
        "total_domains": len(domains_cfg),
        "total_target_samples": sum(d["target_samples"] for d in domains_cfg),
        "domains": domains_cfg,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config
