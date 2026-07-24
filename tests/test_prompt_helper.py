"""CPU-only tests for the chat-template techniques in `PromptHelper`.

Everything runs on plain stubs — NO model, NO GPU, NO real tokenizer/lens. A
tiny fake tokenizer implements a Qwen-ish `apply_chat_template` (honoring
`add_generation_prompt`, `enable_thinking`, `continue_final_message`, `tools`),
and a fake lens returns deterministic logits so J-Lens readouts work.

Focus: the NEW universal-template features and the robustness fixes —
  * assistant-prefill via `continue_final_message` (NOT add_generation_prompt),
  * `tools=` threading + `poised_tool` scoring,
  * `enable_thinking=False` is honored / never silently swallowed,
  * template thinking-default detection,
  * the phantom empty-`<think>` guard.
"""
from __future__ import annotations

import re

import pytest
import torch

from jlensvl.prompt_helper import PromptHelper


# ── fixed vocab ──────────────────────────────────────────────────────────────
_TOKENS = [
    "<|unk|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "\n",
    "system", "user", "assistant", "tool",
    "You", "are", "a", "helpful", "assistant.", "hello", "world", "x",
    "severe", "critical", "minor", "safe", "danger", "risk",
    "get_weather", "send_email", "city", "to", "The", "answer", "is",
    "<tools>", "</tools>", "#", "Tools",
]
_ID = {t: i for i, t in enumerate(_TOKENS)}
_UNK = 0
_VOCAB = len(_TOKENS)

_SPLIT_RE = re.compile(r"<\|[^|]*\|>|</?think>|</?tools>|\n|[^\s]+")


def _split(text: str):
    return _SPLIT_RE.findall(text)


# ── fake tokenizer ───────────────────────────────────────────────────────────
class FakeTok:
    """A miniature Qwen-style chat tokenizer. Records the kwargs of the last
    apply_chat_template call so tests can assert which flags were used."""

    all_special_ids = [_ID["<|im_start|>"], _ID["<|im_end|>"]]

    def __init__(self, *, supports_thinking=True, default_thinking=True,
                 ignore_thinking=False):
        # ignore_thinking: accept the kwarg (no TypeError) but render as if it
        # were never passed — the "accepted-but-silently-ignored" failure mode.
        self.supports_thinking = supports_thinking
        self.default_thinking = default_thinking
        self.ignore_thinking = ignore_thinking
        self.last_kwargs = None

    # --- token <-> id ---
    def encode(self, text, add_special_tokens=False):
        return [_ID.get(t, _UNK) for t in _split(text)]

    def decode(self, ids):
        return "".join(_TOKENS[int(i)] if 0 <= int(i) < _VOCAB else "<|unk|>"
                       for i in ids)

    # --- chat template ---
    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=False,
                            continue_final_message=False, tools=None,
                            add_vision_id=None, **kw):
        # signature-level rejection when the flag is unsupported (old templates)
        if "enable_thinking" in kw and not self.supports_thinking:
            raise TypeError("apply_chat_template() got an unexpected keyword "
                            "argument 'enable_thinking'")
        enable_thinking = kw.get("enable_thinking", None)
        if self.ignore_thinking:
            enable_thinking = None  # accepted but ignored
        self.last_kwargs = dict(add_generation_prompt=add_generation_prompt,
                                continue_final_message=continue_final_message,
                                tools=tools, add_vision_id=add_vision_id,
                                enable_thinking=enable_thinking)
        parts = []
        if tools:
            parts.append("<tools>\n" + " ".join(self._tool_name(t) for t in tools)
                         + "\n</tools>\n")
        for i, m in enumerate(messages):
            role = m["role"]
            last = i == len(messages) - 1
            if last and continue_final_message and role == "assistant":
                # continue this turn: NO trailing <|im_end|>, keep it open
                body = self._assistant_body(m)
                parts.append(f"<|im_start|>{role}\n{body}")
                return "".join(parts)
            if role == "assistant":
                parts.append(f"<|im_start|>{role}\n{self._assistant_body(m)}"
                             f"<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{m.get('content','')}"
                             f"<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            thinking = self.default_thinking if enable_thinking is None else enable_thinking
            parts.append("<think>\n" if thinking else "<think>\n\n</think>\n\n")
        return "".join(parts)

    @staticmethod
    def _tool_name(t):
        if callable(t):
            return getattr(t, "__name__", "fn")
        fn = t.get("function", t) if isinstance(t, dict) else {}
        return fn.get("name", "fn")

    @staticmethod
    def _assistant_body(m):
        rc = m.get("reasoning_content")
        content = m.get("content", "") or ""
        if rc is not None:
            return f"<think>\n{rc}\n</think>\n\n{content}"
        return content


# ── fake lens + jl ───────────────────────────────────────────────────────────
class FakeLens:
    source_layers = [1, 2, 3]

    def __init__(self, tok):
        self.tok = tok

    def apply(self, lm, rendered, *, positions=None, layers=None, use_jacobian=True):
        ids = self.tok.encode(rendered)
        n = len(ids)
        g = torch.Generator().manual_seed(n * 131 + (ids[-1] if ids else 0))
        full = torch.randn(n, _VOCAB, generator=g)
        sel = list(range(n)) if positions is None else [range(n)[p] for p in positions]
        out = full[sel]
        ll = {L: out.clone() for L in (layers or self.source_layers)}
        return ll, None, torch.tensor(ids)


class FakeJL:
    def __init__(self, tok):
        self.tok = tok
        self.lm = object()
        self.lens = FakeLens(tok)

    def _require_lens(self):
        pass

    def _decode(self, ids):
        return [self.tok.decode([int(i)]).replace("\n", "\\n") for i in ids]

    def _word_ids(self, words):
        s = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                e = self.tok.encode(v, add_special_tokens=False)
                if len(e) == 1 and e[0] != _UNK:
                    s.add(e[0])
        return sorted(s)


def _helper(**tok_kw):
    return PromptHelper(FakeJL(FakeTok(**tok_kw)))


SENSES = {"high": ["severe", "critical"], "low": ["minor", "safe"]}
MSGS = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello world"}]


# ── _render + robustness ─────────────────────────────────────────────────────
def test_render_enable_thinking_false_is_honored():
    h = _helper(default_thinking=True)
    on = h._render(MSGS, enable_thinking=None)      # template default = ON
    off = h._render(MSGS, enable_thinking=False)
    assert on.endswith("<think>\n")                 # open think block
    assert "<think>\n\n</think>\n\n" in off          # phantom empty block
    assert on != off
    # the flag actually reached the tokenizer
    assert h.jl.tok.last_kwargs["enable_thinking"] is False


def test_render_unsupported_thinking_false_raises_not_swallowed():
    h = _helper(supports_thinking=False)
    # True but unsupported -> silently degrade to a plain render (template default);
    # the important part is it does NOT raise.
    assert "<|im_start|>assistant\n" in h._render(MSGS, enable_thinking=True)
    # False but unsupported -> must NOT be silently dropped
    with pytest.raises(TypeError, match="enable_thinking=False"):
        h._render(MSGS, enable_thinking=False)


def test_render_thinking_false_ignored_by_template_raises():
    # template ACCEPTS enable_thinking but silently ignores it (default ON):
    # False must not be honored in name only.
    h = _helper(default_thinking=True, ignore_thinking=True)
    with pytest.raises(TypeError, match="ignores/rejects the flag"):
        h._render(MSGS, enable_thinking=False)
    # ...and detection reports the default as moot (None), not a false "off".
    assert h._thinking_default() is None
    assert h._thinking_flag_effective() is False


def test_render_unrelated_typeerror_propagates():
    h = _helper()

    def boom(*a, **k):
        raise TypeError("something else entirely broke")

    h.jl.tok.apply_chat_template = boom
    with pytest.raises(TypeError, match="something else"):
        h._render(MSGS, enable_thinking=False)


def test_thinking_default_detection():
    assert _helper(default_thinking=True)._thinking_default() is True
    assert _helper(default_thinking=False)._thinking_default() is False
    assert _helper(supports_thinking=False)._thinking_default() is None


def test_render_threads_tools_and_vision():
    h = _helper()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    r = h._render(MSGS, tools=tools, add_vision_id=True)
    assert "get_weather" in r
    assert h.jl.tok.last_kwargs["tools"] == tools
    assert h.jl.tok.last_kwargs["add_vision_id"] is True


# ── poised_continuation (assistant prefill) ──────────────────────────────────
def test_poised_continuation_uses_continue_not_generation_prompt():
    h = _helper()
    out = h.poised_continuation(MSGS, "The answer is", senses=SENSES)
    kw = h.jl.tok.last_kwargs
    assert kw["continue_final_message"] is True
    assert kw["add_generation_prompt"] is False        # mutually exclusive
    # rendered ends with the open (continued) assistant turn, no <|im_end|>
    assert out["rendered"].endswith("The answer is")
    assert not out["rendered"].endswith("<|im_end|>\n")
    assert set(out["senses"]) == {"high", "low"}
    assert len(out["tokens"]) == 6 and isinstance(out["margin"], float)


def test_poised_continuation_reasoning_prefill_goes_in_think():
    h = _helper()
    out = h.poised_continuation(MSGS, "let me think", reasoning=True)
    assert out["reasoning"] is True
    # reasoning prefill lands inside a (closed) think block
    assert "<think>\nlet me think\n</think>" in out["rendered"]


# ── tools scoring ────────────────────────────────────────────────────────────
def test_poised_tool_scores_tools_and_args():
    h = _helper()
    tools = [{"type": "function", "function": {"name": "get_weather"}},
             {"type": "function", "function": {"name": "send_email"}}]
    out = h.poised_tool(MSGS, tools, arg_names=["city", "to"])
    assert set(out["tools"]) == {"get_weather", "send_email"}
    assert out["top_tool"] in out["tools"]
    assert set(out["args"]) == {"city", "to"}
    assert out["top_arg"] in out["args"]
    assert all(isinstance(v, float) for v in out["tools"].values())


def test_poised_tool_noop_without_tools():
    assert _helper().poised_tool(MSGS, None) == {}
    assert _helper().poised_tool(MSGS, []) == {}


def test_tool_names_extraction_handles_shapes():
    def get_weather():
        pass
    tools = [{"type": "function", "function": {"name": "a"}},
             {"name": "b"}, get_weather]
    assert PromptHelper._tool_names(tools) == ["a", "b", "get_weather"]


# ── phantom empty-<think> guard ──────────────────────────────────────────────
def test_phantom_think_positions_detects_empty_block():
    toks = ["<|im_start|>", "assistant", "\n", "<think>", "\n", "\n",
            "</think>", "\n", "\n", "answer"]
    ph = PromptHelper._phantom_think_positions(toks)
    assert ph == {3, 4, 5, 6}
    # a NON-empty think block is not flagged
    toks2 = ["<think>", "\n", "hello", "\n", "</think>"]
    assert PromptHelper._phantom_think_positions(toks2) == set()


def test_trace_rendered_flags_phantom_and_scores_senses():
    h = _helper(default_thinking=True)
    tr = h.trace_rendered(MSGS, senses=SENSES, enable_thinking=False)
    # thinking-off render injects the phantom empty block -> flagged
    assert tr["phantom_think"]
    for p in tr["phantom_think"]:
        assert tr["per"][p]["phantom"] and tr["per"][p]["special"]
    assert set(tr["senses"]) == {"high", "low"}
    assert tr["answer"] == len(tr["tokens"]) - 1


def test_trace_rendered_threads_tools():
    h = _helper()
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    tr = h.trace_rendered(MSGS, tools=tools)
    assert "get_weather" in tr["rendered"]
