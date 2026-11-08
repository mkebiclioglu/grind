"""Model adapters. Internally every conversation uses the OpenAI chat format:

  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [...], "reasoning": str,
   "_anthropic": [raw content blocks, replayed verbatim to Anthropic]}
  {"role": "tool", "tool_call_id": str, "content": str}

Each adapter streams Events and returns the final assistant message.
"""

import json
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class Event:
    kind: str  # "text" | "reasoning" | "tool_start" | "retry"
    text: str = ""


@dataclass
class Result:
    message: dict
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens
    stop: str = ""


class ContextOverflow(Exception):
    pass


def make(profile):
    kind = profile.get("provider", "openai")
    if kind == "openai":
        return OpenAIProvider(profile)
    if kind == "anthropic":
        return AnthropicProvider(profile)
    raise ValueError(f"unknown provider {kind!r}")


def _looks_like_overflow(err):
    s = str(err).lower()
    return any(p in s for p in ("context length", "context_length", "maximum context", "too long", "prompt is too long", "max_model_len"))


def _with_retries(fn, on_event, attempts=6):
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except ContextOverflow:
            raise
        except Exception as e:  # noqa: BLE001 - classify below
            status = getattr(e, "status_code", None)
            if _looks_like_overflow(e):
                raise ContextOverflow(str(e)) from e
            retryable = status is None or status in (408, 409, 429) or status >= 500
            if not retryable or i == attempts - 1:
                raise
            on_event(Event("retry", f"{type(e).__name__}: {str(e)[:160]} (retry in {delay:.0f}s)"))
            time.sleep(delay)
            delay = min(delay * 2, 60)


class OpenAIProvider:
    """Any OpenAI-compatible /v1/chat/completions server."""

    def __init__(self, p):
        from openai import OpenAI

        self.p = p
        self.client = OpenAI(base_url=p.get("base_url") or None, api_key=config.api_key(p) or "none",
                             timeout=600, max_retries=0)

    def _messages(self, system, messages):
        out = [{"role": "system", "content": system}]
        keep_reasoning = self.p.get("preserve_reasoning", False)
        for m in messages:
            m2 = {k: v for k, v in m.items() if not k.startswith("_") and k != "reasoning"}
            if m["role"] == "assistant":
                if keep_reasoning and m.get("reasoning"):
                    m2["reasoning_content"] = m["reasoning"]
                if not m2.get("content") and not m2.get("tool_calls"):
                    m2["content"] = "(no output)"
            out.append(m2)
        return out

    def _params(self):
        p = self.p
        extra = dict(p.get("extra_body") or {})
        kw = {}
        for k in ("temperature", "top_p", "presence_penalty"):
            if k in p:
                kw[k] = p[k]
        style, effort = p.get("effort_style"), p.get("effort")
        if style == "qwen" and effort:
            ctk = dict(extra.get("chat_template_kwargs") or {})
            if effort == "off":
                ctk["enable_thinking"] = False
            else:
                ctk["enable_thinking"] = True
                ctk["reasoning_effort"] = effort
            extra["chat_template_kwargs"] = ctk
        elif style == "openai" and effort:
            kw["reasoning_effort"] = effort
        if p.get("max_tokens"):
            kw["max_tokens"] = p["max_tokens"]
        return kw, extra

    def complete(self, system, messages, tools, on_event):
        kw, extra = self._params()

        def run():
            stream = self.client.chat.completions.create(
                model=self.p["model"], messages=self._messages(system, messages),
                tools=tools or None, stream=True, stream_options={"include_usage": True},
                extra_body=extra or None, **kw)
            text, reasoning, calls, usage, stop = [], [], {}, {}, ""
            for chunk in stream:
                if chunk.usage:
                    usage = {"input_tokens": chunk.usage.prompt_tokens, "output_tokens": chunk.usage.completion_tokens}
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                d = ch.delta
                stop = ch.finish_reason or stop
                extra_fields = getattr(d, "model_extra", None) or {}
                r = extra_fields.get("reasoning_content") or extra_fields.get("reasoning")
                if r:
                    reasoning.append(r)
                    on_event(Event("reasoning", r))
                if d.content:
                    text.append(d.content)
                    on_event(Event("text", d.content))
                for tc in d.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": []})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                        on_event(Event("tool_start", slot["name"]))
                    if tc.function and tc.function.arguments:
                        slot["args"].append(tc.function.arguments)
            msg = {"role": "assistant", "content": "".join(text)}
            if reasoning:
                msg["reasoning"] = "".join(reasoning)
            if calls:
                msg["tool_calls"] = [
                    {"id": c["id"] or f"call_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": "".join(c["args"]) or "{}"}}
                    for i, c in sorted(calls.items())]
            return Result(msg, usage, stop)

        return _with_retries(run, on_event)


class AnthropicProvider:
    """Anthropic Messages API. Thinking blocks are replayed unchanged."""

    def __init__(self, p):
        import anthropic

        self.p = p
        self.client = anthropic.Anthropic(api_key=config.api_key(p) or None, max_retries=0,
                                          base_url=p.get("base_url") or None)

    @staticmethod
    def _tools(tools):
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in tools]

    @staticmethod
    def _messages(messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                block = {"type": "text", "text": m["content"]}
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant":
                if m.get("_anthropic"):
                    content = m["_anthropic"]
                else:  # history produced by another provider
                    content = []
                    if m.get("content"):
                        content.append({"type": "text", "text": m["content"]})
                    for tc in m.get("tool_calls") or []:
                        try:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        content.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args})
                    if not content:
                        content = [{"type": "text", "text": "(no output)"}]
                out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if m.get("_is_error"):
                    block["is_error"] = True
                if out and out[-1]["role"] == "user" and all(b["type"] == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system, messages, tools, on_event):
        p = self.p
        kw = {}
        if p.get("effort"):
            kw["output_config"] = {"effort": p["effort"]}

        def run():
            with self.client.messages.stream(
                model=p["model"], max_tokens=p.get("max_tokens", 64000), system=system,
                messages=self._messages(messages), tools=self._tools(tools),
                thinking={"type": "adaptive", "display": "summarized"},
                cache_control={"type": "ephemeral"}, **kw,
            ) as stream:
                for ev in stream:
                    if ev.type == "content_block_start" and ev.content_block.type == "tool_use":
                        on_event(Event("tool_start", ev.content_block.name))
                    elif ev.type == "content_block_delta":
                        if ev.delta.type == "text_delta":
                            on_event(Event("text", ev.delta.text))
                        elif ev.delta.type == "thinking_delta":
                            on_event(Event("reasoning", ev.delta.thinking))
                final = stream.get_final_message()
            text, reasoning, calls = [], [], []
            for b in final.content:
                if b.type == "text":
                    text.append(b.text)
                elif b.type == "thinking":
                    reasoning.append(b.thinking or "")
                elif b.type == "tool_use":
                    calls.append({"id": b.id, "type": "function",
                                  "function": {"name": b.name, "arguments": json.dumps(b.input)}})
            msg = {"role": "assistant", "content": "".join(text),
                   "_anthropic": [b.model_dump(exclude_none=True) for b in final.content]}
            if reasoning:
                msg["reasoning"] = "\n".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            u = final.usage
            inp = (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            if final.stop_reason == "refusal":
                msg["content"] += "\n[model declined this request]"
            return Result(msg, {"input_tokens": inp, "output_tokens": u.output_tokens}, final.stop_reason or "")

        return _with_retries(run, on_event)
