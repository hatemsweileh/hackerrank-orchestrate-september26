"""Provider-agnostic LLM / VLM client.

Supports OpenAI, Anthropic and Google Gemini through their public REST APIs
using only the standard library (no SDK version pinning problems).

Selection (first match wins):
  * LLM_PROVIDER=openai|anthropic|gemini (explicit), otherwise
  * whichever of OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY is set.
  * LLM_MODEL overrides the default model for the chosen provider.
  * LLM_DISABLE=1 forces the offline path even when a key exists.

Every call returns parsed JSON (the prompts demand a JSON object) and is
recorded in a UsageTracker so evaluation/usage_report.md reflects the real
run. Responses are cached on disk keyed by a hash of provider, model, prompt
and image bytes, so re-running the pipeline does not re-bill identical calls.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
}

# USD per 1M tokens (input, output). Used only for the cost estimate in the
# usage report; update if provider pricing changes.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
}

ENV_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


@dataclass
class CallRecord:
    provider: str
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    cached: bool
    ok: bool
    retries: int = 0


@dataclass
class UsageTracker:
    calls: List[CallRecord] = field(default_factory=list)
    offline_extractions: Dict[str, int] = field(default_factory=dict)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def note_offline(self, purpose: str) -> None:
        self.offline_extractions[purpose] = self.offline_extractions.get(purpose, 0) + 1

    def summary(self) -> Dict[str, dict]:
        per: Dict[str, dict] = {}
        for c in self.calls:
            key = f"{c.provider}/{c.model}"
            s = per.setdefault(key, {"calls": 0, "live_calls": 0, "cache_hits": 0, "failures": 0,
                                     "input_tokens": 0, "output_tokens": 0, "retries": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["cache_hits"] += int(c.cached)
            s["live_calls"] += int(not c.cached)
            s["failures"] += int(not c.ok)
            s["retries"] += c.retries
            s["input_tokens"] += c.input_tokens
            s["output_tokens"] += c.output_tokens
            pin, pout = PRICING.get(c.model, (0.0, 0.0))
            s["cost_usd"] += (c.input_tokens * pin + c.output_tokens * pout) / 1_000_000
        return per


class LLMUnavailable(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


class LLMClient:
    def __init__(self, tracker: UsageTracker, cache_path: Optional[Path] = None,
                 timeout: float = 60.0, max_retries: int = 2):
        self.tracker = tracker
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider, self.api_key = self._select_provider()
        self.model = os.environ.get("LLM_MODEL") or (DEFAULT_MODELS.get(self.provider) if self.provider else None)
        self.cache_path = cache_path
        self.cache: Dict[str, dict] = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt cache must not stop the run
                self.cache = {}

    @staticmethod
    def _select_provider():
        if os.environ.get("LLM_DISABLE", "").strip() in {"1", "true", "yes"}:
            return None, None
        explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if explicit in ENV_KEYS:
            key = os.environ.get(ENV_KEYS[explicit], "").strip()
            return (explicit, key) if key else (None, None)
        for prov in ("openai", "anthropic", "gemini"):
            key = os.environ.get(ENV_KEYS[prov], "").strip()
            if key:
                return prov, key
        if os.environ.get("GOOGLE_API_KEY", "").strip():
            return "gemini", os.environ["GOOGLE_API_KEY"].strip()
        return None, None

    @property
    def available(self) -> bool:
        return self.provider is not None

    def save_cache(self) -> None:
        if self.cache_path and getattr(self, "_dirty", False):
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------ public
    def complete_json(self, system: str, user: str, purpose: str,
                      image_bytes: Optional[bytes] = None, max_tokens: int = 600) -> dict:
        if not self.available:
            raise LLMUnavailable("no LLM provider configured")
        h = hashlib.sha256()
        for part in (self.provider, self.model, system, user):
            h.update(part.encode("utf-8"))
        if image_bytes:
            h.update(image_bytes)
        key = h.hexdigest()
        if key in self.cache:
            hit = self.cache[key]
            self.tracker.add(CallRecord(self.provider, self.model, purpose, hit.get("in", 0), hit.get("out", 0),
                                        cached=True, ok=True))
            return hit["json"]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                text, tin, tout = getattr(self, f"_call_{self.provider}")(system, user, image_bytes, max_tokens)
                parsed = _extract_json(text)
                self.tracker.add(CallRecord(self.provider, self.model, purpose, tin, tout, cached=False,
                                            ok=True, retries=attempt))
                self.cache[key] = {"json": parsed, "in": tin, "out": tout}
                self._dirty = True
                return parsed
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                    break  # not retryable
                time.sleep(1.5 * (attempt + 1))
        self.tracker.add(CallRecord(self.provider, self.model, purpose, 0, 0, cached=False, ok=False,
                                    retries=self.max_retries))
        raise LLMUnavailable(f"{self.provider} call failed: {last_err}")

    # --------------------------------------------------------------- providers
    def _post(self, url: str, payload: dict, headers: Dict[str, str]) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _call_openai(self, system, user, image_bytes, max_tokens):
        content: list = [{"type": "text", "text": user}]
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        }
        data = self._post("https://api.openai.com/v1/chat/completions", payload,
                          {"Authorization": f"Bearer {self.api_key}"})
        usage = data.get("usage", {})
        return (data["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0))

    def _call_anthropic(self, system, user, image_bytes, max_tokens):
        content: list = []
        if image_bytes:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                         "data": base64.b64encode(image_bytes).decode("ascii")}})
        content.append({"type": "text", "text": user})
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                   "messages": [{"role": "user", "content": content}]}
        data = self._post("https://api.anthropic.com/v1/messages", payload,
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def _call_gemini(self, system, user, image_bytes, max_tokens):
        parts: list = [{"text": user}]
        if image_bytes:
            parts.append({"inline_data": {"mime_type": "image/png",
                                          "data": base64.b64encode(image_bytes).decode("ascii")}})
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
               f"?key={self.api_key}")
        data = self._post(url, payload, {})
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
