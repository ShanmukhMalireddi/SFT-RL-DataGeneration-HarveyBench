
# Skill: High-Signal SFT & RL Legal Data Synthesis (`Qwen/Qwen3.5-9B`)

Follow every rule below when synthesizing `(task_prompt, rubric, reasoning, answer)` rows or grading candidate outputs.

## 1. Closed-World Source Grounding (Hard Constraint)
- **Packet-only facts:** Every party name, date, dollar amount, percentage, section number, statutory citation, and holding in `reasoning`, `answer`, and `rubric` MUST appear verbatim or by exact calculation from the provided `SOURCE PACKET`.
- **No fabricated placeholders:** Never output `[Company A]`, `[Insert Date]`, `[TBD]`, or `...` unless that literal string appears in the source document.
- **Reject thin excerpts:** If a document excerpt lacks concrete legal rules, clauses, or holdings, focus strictly on the operative provisions present rather than inventing background facts.

## 2. Content-Aware Task Matching & Prompt Design (`task_prompt`)
- **Match task type to document kind (`selected_task_type`):**
  - *Contracts / Agreements:* Eligible for `issue_spotting_and_risk_redline`, `legal_memo`, `compliance_checklist_table`, or `clause_extraction_qa`.
  - *Judicial Opinions / Case Law:* Use `legal_memo` or `clause_extraction_qa` ONLY (never force contract redlining onto a court opinion).
  - *Statutes / Regulations / Tax Instructions:* Use `compliance_checklist_table`, `legal_memo`, or `clause_extraction_qa` (never contract redlining).
  - If `Preferred Task Type` does not naturally fit the excerpt, select the best-fitting option from `Compatible Task Types` and return it in `selected_task_type`.
- **Ban generic summaries:** Never write prompts like `"Summarize this document"` or `"Explain the main points"`. Base `Qwen/Qwen3.5-9B` already passes generic summaries, yielding near-zero SFT/RL learning signal.
- **Require multi-clause / cross-document reasoning:** Design realistic practitioner assignments that force the student model to combine 3+ specific provisions:
  - *Legal Memo:* Reconcile a general rule with its carve-outs, conditions precedent, notice windows, and liability caps across multiple sections/documents.
  - *Issue Spotting & Redline:* Identify 3-5 concrete one-sided risks, ambiguities, or missing protections in the packet and draft exact replacement clause language for each.
  - *Compliance / Schedule Table:* Extract a Markdown table of every deadline, threshold, dollar figure, notice period, responsible party, and exact section citation, including any computed totals or deltas.
  - *Clause Extraction QA:* Ask a targeted edge-case scenario (`"If Party X terminates 45 days after..."`) whose answer hinges on an exception or modifier buried in the text.
- **Zero answer leakage:** State the scenario, role, and required deliverable format clearly in `task_prompt`, but NEVER hint at the legal conclusion or list the hidden exceptions in the prompt itself.

## 3. Atomic, Verifiable Rubrics (`rubric`) for RL (`GRPO`) & Judge Filtering
- **Atomic & binary-verifiable (4 to 7 items):** Each rubric criterion must test ONE checkable fact, exception, calculation, citation, or structural requirement so automated verifiers and LLM judges agree (`1.0` = present and accurate, `0.0` = missed or wrong).
- **Include exact packet anchors:** Write criteria with explicit anchors, e.g.:
  - *Good:* `"Identifies the $2,500,000 indemnification cap in Section 9.2(a) and notes that fraud and willful misconduct are carved out under Section 9.2(c)."`
  - *Bad:* `"Discusses indemnification thoroughly and accurately."`
- **Target the GRPO learning zone:** Calibrate task difficulty so an un-tuned 9B student model gets partial credit (`0.25` to `0.75`) rather than `0.0` (impossible) or `1.0` (trivial). Including both straightforward extraction criteria and subtle exception/carve-out criteria guarantees non-zero within-group reward variance (`std(r) > 0`) during GRPO.
- **Include 1 negative / anti-hallucination criterion:** Add one rubric check verifying that the response does not invent external statutes/facts outside the packet or omit explicit caveats.

## 4. Student-Ready Chain-of-Thought (`reasoning`)
- **Clean first-person analytical trace:** Write `reasoning` exactly as the target student (`Qwen/Qwen3.5-9B` `<think>` block) should think at inference time:
  1. Deconstruct the user's question and required deliverable sections/columns.
  2. Scan each packet document, quoting exact section numbers, defined terms, dates, and figures.
  3. Check cross-references, modifiers (`"except as provided in..."`, `"notwithstanding..."`, `"provided, however..."`), and arithmetic step by step.
  4. Verify completeness before drafting the final response.
- **Zero meta-leakage (Critical):** NEVER mention `"the rubric"`, `"SYNTHESIZE_TASK_JSON"`, `"teacher"`, `"training example"`, or `"gold answer"` inside `reasoning` or `answer`.

## 5. Publication-Grade Target Output (`answer`)
- **Complete & self-contained:** Satisfy 100% of the `rubric` criteria with explicit headings, exact section/document citations, and clean Markdown tables or redline blocks (`[Current Language]` -> `[Proposed Redline]` + `[Rationale]`).
- **No truncation or repetition:** Never trail off, summarize away required items (`"etc."`, `"and similar clauses"`), or repeat paragraphs.

## 6. Judge Grading Rules (`GRADE_RUBRIC_JSON`)
- Grade each criterion strictly (`1.0` if fully satisfied and faithful to the packet, `0.5` if partially satisfied, `0.0` if missed or contradicted by the packet).
- Penalize (`rubric_score <= 0.5`) any response that hallucinates non-packet facts, echoes meta-instructions, uses `[TBD]` placeholders, or gives a skeletal stub answer.
