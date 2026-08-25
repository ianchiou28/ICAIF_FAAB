"""Multi-vendor model channel for the FAAB baselines (OpenAI-compatible).

Reviewers asked that the main conclusions not rest on a single model, so "model" is decoupled
from the global provider selection in agents/llm.py: this module resolves vendor, base_url and
key environment variable from the model name, each baseline names its model explicitly with
--model, and result files are archived separately per model.

Adding a second vendor takes one extra line in VENDORS; putting the key in .env is enough for
it to be picked up automatically.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "deepseek-chat"


def _load_dotenv() -> None:
    """Project-root .env -> os.environ (never overriding real environment variables), following
    the same convention as agents/llm.py."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# vendor -> (key environment variable, base_url, model-name prefixes). Prefixes drive automatic
# routing; "vendor:model" names a vendor explicitly.
VENDORS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "prefixes": ("deepseek-",),
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "prefixes": ("gpt-", "o1", "o3", "o4"),
    },
    "dashscope": {
        "env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "prefixes": ("qwen", "qwq"),
    },
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "prefixes": ("kimi", "moonshot"),
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "prefixes": ("glm",),
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "prefixes": (),
    },
    # Local CLI proxy (OpenAI-compatible). No prefixes, so it must be named as local:<model>.
    "local": {
        "env": "LOCAL_API_KEY",
        "base_url": "http://127.0.0.1:8318/v1",
        "prefixes": (),
    },
}

# Reasoning models: they reject temperature, and their chain of thought consumes the output budget.
_REASONING = re.compile(r"(reasoner|^o[134]|^gpt-5|thinking|qwq|-r1)", re.I)
# Model families that take max_completion_tokens rather than max_tokens.
_COMPLETION_BUDGET = re.compile(r"^(gpt-5|o[134])", re.I)


def vendor_of(model: str) -> str:
    """Resolve the vendor from "vendor:model" or from a model-name prefix; openrouter is the
    explicit fallback."""
    if ":" in model and model.split(":", 1)[0] in VENDORS:
        return model.split(":", 1)[0]
    name = model.lower()
    for vendor, spec in VENDORS.items():
        if any(name.startswith(p) for p in spec["prefixes"]):
            return vendor
    raise ValueError(
        f"cannot infer the vendor for model {model}; name it explicitly as 'vendor:model', "
        f"choosing from {sorted(VENDORS)}")


def bare_name(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model and model.split(":", 1)[0] in VENDORS else model


def is_reasoning(model: str) -> bool:
    return bool(_REASONING.search(bare_name(model)))


def available(model: str) -> bool:
    try:
        return bool(os.environ.get(VENDORS[vendor_of(model)]["env"]))
    except ValueError:
        return False


def slug(model: str) -> str:
    """Result-file suffix: the default model gets no suffix, keeping the frozen v1.4 filenames."""
    if model == DEFAULT_MODEL:
        return ""
    return "__" + re.sub(r"[^a-zA-Z0-9._-]+", "-", model)


class ToolCall:
    """Normalized tool call that hides the field differences between chat.completions and responses."""

    __slots__ = ("id", "name", "arguments")

    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.name = name
        self.arguments = arguments


class Reply:
    """Normalized model reply. `raw` keeps the vendor-native entry for writing back into history."""

    __slots__ = ("content", "tool_calls", "raw", "parsed_json")

    def __init__(self, content: str | None, tool_calls: list[ToolCall], raw: Any):
        self.content = content
        self.tool_calls = tool_calls
        self.raw = raw
        self.parsed_json: dict[str, Any] | None = None


class ChatModel:
    """OpenAI-compatible channel. Function calling with reasoning on the gpt-5 family is only
    available on /v1/responses, so the vendor decides between chat.completions and responses and
    this class owns the conversation-history format."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 600):
        from openai import OpenAI

        self.spec_name = model
        self.vendor = vendor_of(model)
        self.model_name = bare_name(model)
        spec = VENDORS[self.vendor]
        key = os.environ.get(spec["env"])
        if not key:
            raise RuntimeError(f"{model} requires the environment variable {spec['env']}")
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if spec["base_url"]:
            kwargs["base_url"] = spec["base_url"]
        self.client = OpenAI(**kwargs)
        self.reasoning = is_reasoning(model)
        # The local CLI proxy exposes only chat.completions, not /v1/responses.
        self.api = ("responses" if self.vendor == "openai"
                    and _COMPLETION_BUDGET.match(self.model_name) else "chat")

    # --- Conversation construction: callers use only these four methods and never assemble
    # --- provider fields themselves ------------------------------------------------------

    def seed(self, system: str, user: str) -> list[Any]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def add_user(self, conversation: list[Any], text: str) -> None:
        conversation.append({"role": "user", "content": text})

    def add_reply(self, conversation: list[Any], reply: Reply) -> None:
        if self.api == "responses":
            conversation.extend(reply.raw)
        else:
            conversation.append(reply.raw)

    def add_tool_result(self, conversation: list[Any], call: ToolCall, output: str) -> None:
        if self.api == "responses":
            conversation.append({"type": "function_call_output",
                                 "call_id": call.id, "output": output})
        else:
            conversation.append({"role": "tool", "tool_call_id": call.id,
                                 "content": output})

    def tool_schema(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """TOOL_SPECS are declared in chat.completions shape; responses needs them flattened."""
        if self.api != "responses":
            return tools
        return [{"type": "function", **t["function"]} for t in tools]

    # --- Invocation ----------------------------------------------------------

    def _budget(self, max_tokens: int) -> int:
        # Chain of thought counts against the output budget; without headroom a reasoning model
        # gets truncated during the reasoning phase.
        return max(max_tokens, 8192) if self.reasoning else max_tokens

    def complete(
        self,
        conversation: list[Any],
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        json_object: bool = False,
    ) -> tuple[Reply, dict[str, int]]:
        if self.api == "responses":
            return self._complete_responses(conversation, max_tokens, tools, json_object)
        return self._complete_chat(conversation, max_tokens, tools, json_object)

    def _complete_chat(self, conversation, max_tokens, tools, json_object):
        kwargs: dict[str, Any] = {
            "model": self.model_name, "messages": conversation,
            "max_tokens": self._budget(max_tokens),
        }
        if not self.reasoning:
            kwargs["temperature"] = 0
        if tools:
            kwargs["tools"] = self.tool_schema(tools)
            kwargs["tool_choice"] = "auto"
        if json_object and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # Some reasoning models reject response_format/temperature.
            text = str(exc).lower()
            if not any(t in text for t in ("response_format", "temperature",
                                           "json_object", "max_tokens")):
                raise
            kwargs.pop("response_format", None)
            kwargs.pop("temperature", None)
            if "max_tokens" in text and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments)
                 for c in (message.tool_calls or [])]
        raw: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if calls:
            raw["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        usage_obj = response.usage
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }
        return Reply(message.content, calls, raw), usage

    def _complete_responses(self, conversation, max_tokens, tools, json_object):
        kwargs: dict[str, Any] = {
            "model": self.model_name, "input": conversation,
            "max_output_tokens": self._budget(max_tokens),
        }
        if tools:
            kwargs["tools"] = self.tool_schema(tools)
        if json_object and not tools:
            kwargs["text"] = {"format": {"type": "json_object"}}
        response = self.client.responses.create(**kwargs)
        items = [item.model_dump(exclude_none=True) for item in response.output]
        calls = [ToolCall(item["call_id"], item["name"], item.get("arguments") or "{}")
                 for item in items if item.get("type") == "function_call"]
        usage_obj = response.usage
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
        }
        return Reply(response.output_text, calls, items), usage


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: str | None) -> dict[str, Any]:
    """Extract one JSON object from model text, tolerating code fences, chain-of-thought tags
    and surrounding prose."""
    if not text:
        raise ValueError("empty response")
    body = _THINK.sub("", text).strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    start = body.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(body)):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = body.find("{", start + 1)
    raise ValueError("no JSON object in response")
