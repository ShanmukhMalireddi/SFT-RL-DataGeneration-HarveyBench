"""Unified HTTP client supporting BOTH native Google Gemini generateContent API and OpenAI-compatible /chat/completions."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENV_FILE = Path(__file__).resolve().parent / "api_keys.env"


def load_env_file(env_path: Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        values[key] = val
    return values


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:240]}")
    return json.loads(text[start : end + 1])


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        api_format: str = "auto",
        mock: bool = False,
        timeout: int = 240,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = "gemini-4-argon" if model == "gemini-argon-4" else model
        self.api_format = api_format  # "auto", "gemini", or "openai"
        self.mock = mock
        self.timeout = timeout

    def _resolved_format(self) -> str:
        if self.api_format in ("gemini", "openai"):
            return self.api_format
        # Auto-detect: native Gemini REST vs OpenAI-compatible
        if "generativelanguage" in self.base_url and not self.base_url.endswith("/openai"):
            return "gemini"
        return "openai"

    @classmethod
    def from_env(
        cls,
        prefix: str,
        fallback_prefix: str | None = None,
        cli_base_url: str | None = None,
        cli_api_key: str | None = None,
        cli_model: str | None = None,
        api_format: str = "auto",
        mock: bool = False,
        env_file: Path = DEFAULT_ENV_FILE,
    ) -> "ChatClient":
        file_vars = load_env_file(env_file)

        def get_var(name: str) -> str:
            val = os.environ.get(name, "")
            if not val:
                val = file_vars.get(name, "")
            if val == "PASTE_YOUR_TEACHER_API_KEY_HERE":
                return ""
            return val

        def pick(suffix: str, default: str = "") -> str:
            val = get_var(f"{prefix}_{suffix}")
            if not val and fallback_prefix:
                val = get_var(f"{fallback_prefix}_{suffix}")
            return val or default

        resolved_fmt = api_format if api_format != "auto" else (get_var("API_FORMAT") or "auto")
        base_url = (
            cli_base_url
            or pick("BASE_URL")
            or get_var("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        api_key = (
            cli_api_key
            or pick("API_KEY")
            or get_var("GEMINI_API_KEY")
            or get_var("GOOGLE_API_KEY")
        )
        model = cli_model or pick("MODEL") or "gemini-4-argon"

        if not mock and not api_key:
            raise RuntimeError(
                f"Missing API key: edit api_keys.env ({env_file}) and set {prefix}_API_KEY "
                f"(or pass --teacher-api-key / --mock)."
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model if not mock else f"mock-{prefix.lower()}",
            api_format=resolved_fmt,
            mock=mock,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 8192,
    ) -> tuple[str, str]:
        if self.mock:
            return self._mock_complete(messages)
        if self._resolved_format() == "gemini":
            return self._complete_gemini_native(messages, temperature=temperature, max_tokens=max_tokens)
        return self._complete_openai(messages, temperature=temperature, max_tokens=max_tokens)

    def _complete_gemini_native(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        model_path = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = f"{self.base_url}/{model_path}:generateContent?key={urllib.parse.quote(self.api_key)}"

        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = str(m.get("content", ""))
            if role == "system":
                system_parts.append({"text": text})
            else:
                g_role = "model" if role == "assistant" else "user"
                contents.append({"role": g_role, "parts": [{"text": text}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for attempt in range(4):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parts = data["candidates"][0]["content"].get("parts", [])
                thoughts = [p.get("text", "") for p in parts if p.get("thought")]
                answers = [p.get("text", "") for p in parts if not p.get("thought")]
                return "\n".join(thoughts).strip(), "\n".join(answers).strip()
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
                if attempt == 3:
                    raise RuntimeError(
                        f"Gemini API HTTP {exc.code} on {self.base_url}/{model_path}: {err_body}"
                    ) from exc
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"Gemini API connection failed ({self.base_url}): {exc}") from exc
                time.sleep(2 ** attempt)
        return "", ""

    def _complete_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(4):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                content = msg.get("content") or ""
                return str(reasoning).strip(), str(content).strip()
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
                if attempt == 3:
                    raise RuntimeError(f"OpenAI-compatible HTTP {exc.code} at {url}: {err_body}") from exc
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"LLM call to {url} ({self.model}) failed: {exc}") from exc
                time.sleep(2 ** attempt)
        return "", ""

    def _mock_complete(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        user_text = messages[-1]["content"]
        if "GRADE_RUBRIC_JSON" in user_text:
            return (
                "Checking each rubric item against source packet.",
                json.dumps(
                    {
                        "criterion_scores": [1.0, 1.0, 1.0],
                        "rubric_score": 1.0,
                        "notes": "All criteria satisfied and grounded in provided source excerpts.",
                    }
                ),
            )
        if "SYNTHESIZE_TASK_JSON" in user_text:
            doc_match = re.findall(r"=== DOCUMENT (\d+): ([^\n=]+) ===", user_text)
            doc_label = doc_match[0][1].strip() if doc_match else "source_doc.txt"
            snippet_words = [
                w for w in re.findall(r"[A-Za-z]{5,}", user_text)
                if w.lower() not in {"document", "synthesize", "rubric"}
            ][:8]
            anchor = ", ".join(snippet_words[:4]) if snippet_words else "governing provisions"
            return (
                "Drafting grounded legal task, rubric, and gold response.",
                json.dumps(
                    {
                        "task_prompt": (
                            f"Based strictly on the provided legal packet ({doc_label}), prepare a structured "
                            f"legal analysis memorandum addressing the core obligations, exceptions, and procedural "
                            f"requirements concerning {anchor}."
                        ),
                        "rubric": [
                            {"criterion": f"Accurately cites facts and provisions from {doc_label}", "weight": 1.0},
                            {"criterion": "Analyzes primary legal obligations and exceptions", "weight": 1.0},
                            {"criterion": "Uses structured legal memorandum headings without unsupported assumptions", "weight": 1.0},
                        ],
                        "reasoning": (
                            f"1. Inspect {doc_label} to extract operative definitions and governing rules.\n"
                            f"2. Map obligations and exceptions around {anchor}.\n"
                            "3. Structure into Issue, Controlling Rules, Analysis, and Conclusion."
                        ),
                        "answer": (
                            "MEMORANDUM\n\n"
                            "1. Issue Presented\n"
                            f"This memorandum analyzes the legal obligations, exceptions, and procedural requirements "
                            f"set forth in {doc_label} regarding {anchor}.\n\n"
                            "2. Controlling Authorities and Packet Facts\n"
                            f"Under the provided text of {doc_label}, the operative terms establish explicit standards "
                            "governing compliance, notice, and potential remedies.\n\n"
                            "3. Legal Analysis\n"
                            "Applying the packet provisions strictly to the stated scope, parties must satisfy both the "
                            "threshold substantive requirements and the procedural conditions precedent identified in the record.\n\n"
                            "4. Conclusion\n"
                            f"Compliance under {doc_label} requires adherence to each enumerated condition without relying on extrinsic facts."
                        ),
                    }
                ),
            )
        return (
            "Brief review.",
            "Short summary: the document sets out general legal rules, but specific exceptions are omitted.",
        )
