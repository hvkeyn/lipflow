"""Turn raw uppercase lip-reading output into the sentence you meant.

Lip reading is ambiguous in ways speech isn't: p/b/m, f/v, t/d/n look identical on the
lips, so the model's guesses are often homophenes of the real words ("WALLET OFFICER"
for "while in office"). An LLM that sees the top hypotheses plus what you dictated just
before can usually recover the intended sentence.

Backends. You pick one in the menu (saved as "cleanup" in settings) or with --cleanup; "auto" uses
the first available of claude, deepseek, local, ollama, basic:
  claude   – ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) set
  codex    – your ChatGPT plan through the Codex CLI (`codex login`); LIPFLOW_CODEX_MODEL; only if chosen
  deepseek – DEEPSEEK_API_KEY set; LIPFLOW_DEEPSEEK_MODEL (deepseek-chat), LIPFLOW_DEEPSEEK_URL
  local    – a tiny on-device model via MLX (Qwen3-0.6B 4-bit, ~350 MB, ~0.2 s); LIPFLOW_LOCAL_MODEL
  ollama   – a local Ollama server on :11434 (LIPFLOW_OLLAMA_MODEL, default qwen3:4b); only if chosen
  basic    – offline casing + punctuation rules

None of these reads lips. They repair what a lip reader has already proposed, which is why they
cannot bring a language the visual model does not know: see the lip reader's own notes in vsr.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import requests

from . import lang


def _system() -> str:
    """The cleanup prompt for the language being read (English or Russian)."""
    return lang.system_prompt()


def _user_prompt(candidates: list[str], context: str, words: "list[str] | None" = None,
                 similar: "list[str] | None" = None) -> str:
    p = lang.profile()
    lines = []
    if similar:
        lines.append(p.label_similar + "\n" + "\n".join(f"- {s}" for s in similar) + "\n")
    if words:
        lines.append(p.label_words + ", ".join(words) + "\n")
    if context:
        lines.append(p.label_context.format(context=context) + "\n")
    lines.append(p.label_candidates)
    lines += [f"{i + 1}. {c}" for i, c in enumerate(candidates)]
    return "\n".join(lines)


def numbers_to_digits(t: str) -> str:
    """Spelled-out numbers in the active language -> digits ('nineteen forty three' -> 1943)."""
    return lang.numbers_to_digits(t)


# Tiny models copy whatever formatting they're shown, so they get lowercase guesses, a short
# instruction and worked examples instead of the long prompt above. Both live in lang.py now,
# per language; these two names stay as the English default for backwards compatibility.
SMALL_SYSTEM, SMALL_SHOTS, _ = lang.small_prompt("en")
LOCAL_MODEL = "mlx-community/Qwen3-0.6B-4bit"


def small_messages(candidates: list[str], context: str, words: "list[str] | None",
                   similar: "list[str] | None" = None, common: "list[str] | None" = None) -> list[dict]:
    system, shots, (l_names, l_common, l_before, l_guesses) = lang.small_prompt()
    u = (f"{l_names}: {', '.join(words)}\n" if words else "")
    u += (f"{l_common}: {', '.join(common)}\n" if common else "")
    u += (f"{l_before}:\n" + "\n".join("- " + s for s in similar) + "\n") if similar else ""
    u += f"{l_guesses}:\n" + "\n".join("- " + c.lower() for c in candidates)
    msgs = [{"role": "system", "content": system}]
    for a, b in shots:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    return msgs + [{"role": "user", "content": u}]


def fix_case(text: str) -> str:
    """Sentence-case start, and a capital I in English, whatever the model returned."""
    t = text.strip()
    if not t:
        return t
    return lang.sentence_case(lang.fix_pronouns(t))


def _norm_words(text: str) -> list[str]:
    return lang.words(numbers_to_digits(text.lower()))


def _edits(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def within_guesses(out: str, candidates: list[str], strict: bool, known=None, max_edits: "int | None" = None) -> bool:
    """Small models may only format, and pick words the lip-reader actually proposed.
    strict: same words as the top guess. loose: every word appears in some guess, or is a word
    `known(word)` says the user commonly uses; optionally at most `max_edits` word changes."""
    words = _norm_words(out)
    top = _norm_words(candidates[0])
    if strict:
        return words == top
    pool = {w for c in candidates for w in _norm_words(c)}
    if not words or not all(w in pool or (known and known(w)) for w in words):
        return False
    return max_edits is None or _edits(words, top) <= max_edits


def basic_cleanup(text: str) -> str:
    """Offline rules for the active language: digits, sentence case, closing mark, question mark."""
    return lang.base_cleanup(text)


# The cleanup menu, in order. "auto" picks for you (see the docstring above).
CHOICES = {"auto": "Automatic", "claude": "Claude", "codex": "ChatGPT (Codex CLI)", "deepseek": "DeepSeek",
           "local": "On-device model", "ollama": "Ollama", "basic": "Basic rules only"}


def unavailable(backend: str) -> "str | None":
    """Why this backend can't be used right now, or None. Cheap: runs when the menu is built."""
    if backend == "claude" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "set ANTHROPIC_API_KEY"
    if backend == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        return "set DEEPSEEK_API_KEY"
    if backend == "codex" and not shutil.which("codex"):
        return "install the Codex CLI"
    if backend == "local" and sys.platform != "darwin":
        return "Mac only"
    return None


class Cleaner:
    def __init__(self, backend: str = "auto"):
        self.backend = self._pick(backend)
        self.model = None
        self._client = None
        self._mlx = None
        self._words: list[str] = []
        self._similar: list[str] = []
        # Measured on real dictations (WER): free edits 31.5%, format-only 19.0%, one edit using only
        # the guesses' words or words you commonly use, with those words in the prompt: 17.9%.
        self.strict = os.environ.get("LIPFLOW_LOCAL_STRICT", "0") == "1"
        self.max_edits: "int | None" = 1
        self.prompt_common = True
        from .personal import Personal
        self.personal = Personal()
        if self.backend == "local":
            self.model = os.environ.get("LIPFLOW_LOCAL_MODEL", LOCAL_MODEL)
        elif self.backend == "claude":
            import anthropic
            self._client = anthropic.Anthropic(timeout=8.0, max_retries=1)
            self.model = os.environ.get("LIPFLOW_MODEL", "claude-opus-5-5")
        elif self.backend == "codex":
            self.model = os.environ.get("LIPFLOW_CODEX_MODEL")  # None: Codex's own default
        elif self.backend == "deepseek":
            self.model = os.environ.get("LIPFLOW_DEEPSEEK_MODEL", "deepseek-chat")
        elif self.backend == "ollama":
            self.model = os.environ.get("LIPFLOW_OLLAMA_MODEL", "qwen3:4b")

    @staticmethod
    def _ollama_up() -> bool:
        try:
            return requests.get("http://127.0.0.1:11434/api/tags", timeout=0.4).ok
        except requests.RequestException:
            return False

    def _pick(self, backend: str) -> str:
        if backend != "auto":
            why = unavailable(backend)
            if why is None:
                return backend
            print(f"[cleanup] can't use {backend} ({why}); picking automatically")
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "claude"
        if os.environ.get("DEEPSEEK_API_KEY"):
            return "deepseek"
        try:
            import mlx_lm  # noqa: F401  (Apple Silicon only)
            return "local"
        except ImportError:
            pass
        if self._ollama_up():
            return "ollama"
        return "basic"

    def warmup(self):
        """Load (and on first run download) the local model before the first dictation."""
        if self.backend == "local" and self._mlx is None:
            from mlx_lm import load
            self._mlx = load(self.model)
            self._local(["HELLO"], "")

    def describe(self) -> str:
        return f"{self.backend}" + (f" ({self.model})" if self.model else "")

    def is_common(self, word: str) -> bool:
        """Words name-snapping must never replace: ones you use often, plus basic function words."""
        w = word.lower()
        return w in lang.everyday() or (bool(self.personal) and self.personal.uni[w] >= 3)

    def __call__(self, candidates: list[str], context: str = "", words: "list[str] | None" = None,
                 names: "list[str] | None" = None) -> str:
        """names: extra names from what you're typing into (see context.py), for this dictation only."""
        from . import vocab
        from .visemes import snap_names
        words = vocab.load() if words is None else words
        names = [n for n in (names or []) if n.lower() not in {w.lower() for w in words}]
        words = words + names
        self._words = words
        candidates = [c for c in candidates if c.strip()]
        # names look like other words on the lips (Miguel → MCCALL); snap them before ranking
        candidates = list(dict.fromkeys(snap_names(c, words, self.is_common) for c in candidates))
        if self.personal:
            candidates = self.personal.rerank(candidates)
            self._similar = self.personal.similar(" ".join(candidates[:2]))
        else:
            self._similar = []
        candidates = vocab.rerank(candidates, words)
        if not candidates:
            return ""
        self._words = words or []
        try:
            if self.backend == "local":
                out = self._local(candidates, context)
            elif self.backend == "claude":
                out = self._claude(candidates, context)
            elif self.backend == "codex":
                out = self._codex(candidates, context)
            elif self.backend == "deepseek":
                out = self._deepseek(candidates, context)
            elif self.backend == "ollama":
                out = self._ollama(candidates, context)
            else:
                out = None
        except Exception as e:  # never lose a dictation to a network hiccup
            print(f"[cleanup] {self.backend} failed ({e.__class__.__name__}: {e}); using basic cleanup")
            out = None
        return vocab.apply_case((out or basic_cleanup(candidates[0])).strip(), words)

    def _claude(self, candidates: list[str], context: str) -> "str | None":
        resp = self._client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_system(),
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": _user_prompt(candidates, context, self._words, self._similar)}],
        )
        if resp.stop_reason == "refusal":
            return None
        return "".join(b.text for b in resp.content if b.type == "text") or None

    def _codex(self, candidates: list[str], context: str) -> "str | None":
        """One non-interactive `codex exec` turn on your ChatGPT plan. It runs read-only in an empty
        folder, so it can't touch your files or pick up a project's AGENTS.md, and leaves no session."""
        prompt = f"{_system()}\n\n{_user_prompt(candidates, context, self._words, self._similar)}"
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "reply.txt")
            args = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
                    "-c", 'model_reasoning_effort="low"', "-o", out]
            if self.model:
                args += ["-m", self.model]
            r = subprocess.run(args + ["-"], input=prompt, capture_output=True, text=True, timeout=20, cwd=tmp)
            if r.returncode != 0:
                raise RuntimeError(f"codex exited {r.returncode}: {r.stderr.strip()[-200:]}")
            text = open(out, encoding="utf-8").read().strip() if os.path.exists(out) else ""
        return text if text and not text.isupper() else None

    def _deepseek(self, candidates: list[str], context: str) -> "str | None":
        """DeepSeek's API is OpenAI-compatible, so this is one POST with a bearer token.

        DEEPSEEK_API_KEY is the key; LIPFLOW_DEEPSEEK_MODEL picks the model (deepseek-chat by
        default, deepseek-reasoner for the thinking one) and LIPFLOW_DEEPSEEK_URL an endpoint."""
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        r = requests.post(
            os.environ.get("LIPFLOW_DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"),
            timeout=20,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.model,
                "stream": False,
                "temperature": 0,
                "messages": [{"role": "system", "content": _system()},
                             {"role": "user", "content": _user_prompt(candidates, context, self._words, self._similar)}],
            },
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() or None

    def _local(self, candidates: list[str], context: str) -> "str | None":
        from mlx_lm import generate
        if self._mlx is None:
            self.warmup()
        model, tok = self._mlx
        common = self.personal.common_words() if (self.prompt_common and self.personal) else None
        prompt = tok.apply_chat_template(small_messages(candidates, context, self._words, self._similar, common), add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        out = generate(model, tok, prompt=prompt, max_tokens=160, verbose=False)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().split("\n")[0].strip()
        # a tiny model that invents words is worse than no model (measured on real dictations),
        # so it may only format and choose among the lip-reader's own words
        known = self.personal.knows if self.personal else None
        if not out or out.isupper() or not within_guesses(out, candidates, self.strict, known, self.max_edits):
            return None
        return fix_case(out)

    def _ollama(self, candidates: list[str], context: str) -> "str | None":
        r = requests.post("http://127.0.0.1:11434/api/chat", timeout=20, json={
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [{"role": "system", "content": _system()},
                         {"role": "user", "content": _user_prompt(candidates, context, self._words, self._similar)}],
            "options": {"temperature": 0},
        })
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() or None


if __name__ == "__main__":
    import sys
    c = Cleaner(sys.argv[1] if len(sys.argv) > 1 else "auto")
    print("backend:", c.describe())
    print(c(["TODAY AT GEORGE WASHINGTON PRESIDENTS HAVE DELIVERED SOME FORM OF FINAL MESSAGE WALLET OFFICER A FA WELL ADDRESSED TO THE AMERICAN PEOPLE"]))
    print(json.dumps(basic_cleanup("WHAT TIME IS IT I THINK")))
