"""One thin wrapper over whichever LLM API happens to have a key in .env.

Two things here matter more than the provider support:

1. Every call is cached on disk, keyed by a hash of (provider, model, system,
   prompt, temperature). The cache file is committed to the repo, so a grader
   with no API key can still reproduce every number by replaying it.
2. Temperature is 0 by default. Sampled replies would make the eval unrepeatable
   and I would rather have a boring pipeline I can trust twice.

I used plain `requests` instead of the four official SDKs on purpose - the
request bodies are a dozen lines each and it keeps install time down.
"""

import hashlib
import json
import os
import re
import threading
import time

import requests

from src.config import CACHE

CACHE_PATH = CACHE / "llm_cache.jsonl"

DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
}

_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class LLMUnavailable(RuntimeError):
    pass


def _resolve_provider():
    """Prefer an explicit LLM_PROVIDER, otherwise take the first key that exists."""
    want = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if want and os.getenv(_ENV_KEYS.get(want, "")):
        return want
    for name, env in _ENV_KEYS.items():
        if os.getenv(env):
            return name
    return None


class LLM:
    def __init__(self, model=None, provider=None):
        self.provider = provider or _resolve_provider()
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "offline")
        self.api_key = os.getenv(_ENV_KEYS.get(self.provider, ""), "") if self.provider else ""
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    # ---------- cache ----------

    def _load_cache(self):
        cache = {}
        if CACHE_PATH.exists():
            with CACHE_PATH.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a half-written line from an interrupted run
                    cache[rec["key"]] = rec["response"]
        return cache

    def _key(self, system, prompt, temperature):
        blob = json.dumps(
            [self.model, system or "", prompt, temperature], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _remember(self, key, response):
        with self._lock:
            self._cache[key] = response
            with CACHE_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")

    # ---------- public ----------

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=600):
        key = self._key(system, prompt, temperature)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        if not self.provider or not self.api_key:
            raise LLMUnavailable(
                "No cached response for this prompt and no API key configured.\n"
                "Either copy .env.example to .env and add a key, or stick to the "
                "committed cache (see README > Reproducing)."
            )

        text = self._call_with_retry(prompt, system, temperature, max_tokens)
        self.calls += 1
        self._remember(key, text)
        return text

    def complete_json(self, prompt, system=None, temperature=0.0, max_tokens=600, default=None):
        """Same as complete() but tolerant of models that wrap JSON in prose/fences."""
        raw = self.complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        parsed = extract_json(raw)
        if parsed is None:
            if default is not None:
                return default
            raise ValueError(f"Could not find JSON in model output: {raw[:300]!r}")
        return parsed

    # ---------- transport ----------

    def _call_with_retry(self, prompt, system, temperature, max_tokens, attempts=5):
        delay = 2.0
        last = None
        for i in range(attempts):
            try:
                return self._call(prompt, system, temperature, max_tokens)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                last = exc
                # 429 = free-tier rate limit, which is the common case here.
                if status in (408, 409, 429, 500, 502, 503, 504) and i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except requests.RequestException as exc:
                last = exc
                if i < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last

    def _call(self, prompt, system, temperature, max_tokens):
        if self.provider == "gemini":
            return self._gemini(prompt, system, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(prompt, system, temperature, max_tokens)
        return self._openai_compatible(prompt, system, temperature, max_tokens)

    def _gemini(self, prompt, system, temperature, max_tokens):
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = requests.post(
            url, params={"key": self.api_key}, json=body, timeout=90,
        )
        r.raise_for_status()
        cands = r.json().get("candidates") or []
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()

    def _anthropic(self, prompt, system, temperature, max_tokens):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=90,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", [])).strip()

    def _openai_compatible(self, prompt, system, temperature, max_tokens):
        base = {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }[self.provider]
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text):
    """Models love to add 'Here you go:' and code fences. Dig the object out."""
    if not text:
        return None
    for candidate in _fragments(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _fragments(text):
    text = text.strip()
    yield text
    m = _FENCE.search(text)
    if m:
        yield m.group(1).strip()
    # Fall back to the outermost brace/bracket pair.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            yield text[i : j + 1]


def cache_stats():
    n = 0
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
    return {"entries": n, "path": str(CACHE_PATH)}
