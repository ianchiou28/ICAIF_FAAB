"""FAAB 基线的多厂商模型通道（OpenAI 兼容）。

审稿意见要求主结论不能只建立在一个模型上，因此把「模型」从 agents/llm.py 的
全局 provider 选择中解耦出来：这里按模型名解析厂商、base_url 与 key 环境变量，
各基线用 --model 显式指定，结果文件按模型名分开归档。

新增第二厂商只需在 VENDORS 里加一行：把 key 写进 .env 即可自动识别。
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
    """项目根 .env → os.environ（不覆盖真实环境变量），与 agents/llm.py 同约定。"""
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

# vendor -> (key 环境变量, base_url, 模型名前缀)。前缀用于自动路由，也可用
# "vendor:model" 显式指定。
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
    # 本地 CLI Proxy（OpenAI 兼容）。无前缀，必须写成 local:<model> 显式指定。
    "local": {
        "env": "LOCAL_API_KEY",
        "base_url": "http://127.0.0.1:8318/v1",
        "prefixes": (),
    },
}

# 推理型模型：不接受 temperature，且思维链会占用输出预算。
_REASONING = re.compile(r"(reasoner|^o[134]|^gpt-5|thinking|qwq|-r1)", re.I)
# 用 max_completion_tokens 而非 max_tokens 的模型族。
_COMPLETION_BUDGET = re.compile(r"^(gpt-5|o[134])", re.I)


def vendor_of(model: str) -> str:
    """按 "vendor:model" 或模型名前缀解析厂商；openrouter 作为显式兜底。"""
    if ":" in model and model.split(":", 1)[0] in VENDORS:
        return model.split(":", 1)[0]
    name = model.lower()
    for vendor, spec in VENDORS.items():
        if any(name.startswith(p) for p in spec["prefixes"]):
            return vendor
    raise ValueError(
        f"无法识别模型 {model} 的厂商；用 'vendor:model' 显式指定，"
        f"可选 {sorted(VENDORS)}")


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
    """结果文件后缀：默认模型不加后缀，保持已冻结的 v1.4 文件名不变。"""
    if model == DEFAULT_MODEL:
        return ""
    return "__" + re.sub(r"[^a-zA-Z0-9._-]+", "-", model)


class ToolCall:
    """归一化的工具调用，屏蔽 chat.completions 与 responses 的字段差异。"""

    __slots__ = ("id", "name", "arguments")

    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.name = name
        self.arguments = arguments


class Reply:
    """归一化的模型回复。raw 保留厂商原生条目，供写回会话历史。"""

    __slots__ = ("content", "tool_calls", "raw", "parsed_json")

    def __init__(self, content: str | None, tool_calls: list[ToolCall], raw: Any):
        self.content = content
        self.tool_calls = tool_calls
        self.raw = raw
        self.parsed_json: dict[str, Any] | None = None


class ChatModel:
    """OpenAI 兼容通道。gpt-5 系带推理的 function calling 只在 /v1/responses 可用，
    因此按厂商选择 chat.completions 或 responses，并由本类负责会话历史格式。"""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 600):
        from openai import OpenAI

        self.spec_name = model
        self.vendor = vendor_of(model)
        self.model_name = bare_name(model)
        spec = VENDORS[self.vendor]
        key = os.environ.get(spec["env"])
        if not key:
            raise RuntimeError(f"{model} 需要环境变量 {spec['env']}")
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if spec["base_url"]:
            kwargs["base_url"] = spec["base_url"]
        self.client = OpenAI(**kwargs)
        self.reasoning = is_reasoning(model)
        # 本地 CLI Proxy 只暴露 chat.completions，没有 /v1/responses。
        self.api = ("responses" if self.vendor == "openai"
                    and _COMPLETION_BUDGET.match(self.model_name) else "chat")

    # --- 会话构造：调用方只用这四个方法，不直接拼 provider 字段 ---------------

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
        """TOOL_SPECS 以 chat.completions 形状声明；responses 需要摊平。"""
        if self.api != "responses":
            return tools
        return [{"type": "function", **t["function"]} for t in tools]

    # --- 调用 ---------------------------------------------------------------

    def _budget(self, max_tokens: int) -> int:
        # 思维链计入输出预算；不给推理型模型留额度会截断在 reasoning 阶段。
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
        except Exception as exc:  # 部分推理型模型拒绝 response_format/temperature
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
    """从模型文本里取一个 JSON 对象：容忍代码围栏、思维链标签与前后散文。"""
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
