"""Utilities for rendering OpenAI traces into training messages."""

import importlib
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from light_scale.data import Message


_BLOCK_PATTERN = re.compile(
    r"(?P<full>(?P<prefix>(?:\n)?<\|im_start\|>(?P<role>[^\n]+)\n)(?P<content>.*?<\|im_end\|>))",
    re.S,
)
_TOKENIZER_CACHE: Dict[Tuple[str, bool], Any] = {}
_CHAT_TEMPLATE_CACHE: Dict[str, str] = {}
_CACHE_LOCK = Lock()


def get_cached_template_artifacts(
    *,
    tokenizer_path: str,
    trust_remote_code: bool,
    chat_template_path: Optional[str],
    owner_name: str,
):
    cache_key = (str(Path(tokenizer_path).expanduser()), trust_remote_code)
    template_key = None if chat_template_path is None else str(Path(chat_template_path).expanduser().resolve())
    with _CACHE_LOCK:
        tokenizer = _TOKENIZER_CACHE.get(cache_key)
        if tokenizer is None:
            try:
                transformers = importlib.import_module("transformers")
            except ImportError as err:
                raise ImportError(f"{owner_name} requires transformers to load chat templates") from err
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                cache_key[0],
                trust_remote_code=trust_remote_code,
            )
            _TOKENIZER_CACHE[cache_key] = tokenizer

        chat_template = None
        if template_key is not None:
            chat_template = _CHAT_TEMPLATE_CACHE.get(template_key)
            if chat_template is None:
                chat_template = Path(template_key).read_text(encoding="utf-8")
                _CHAT_TEMPLATE_CACHE[template_key] = chat_template

    return tokenizer, chat_template


def normalize_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized_tools.append(deepcopy(tool))
        else:
            normalized_tools.append({"type": "function", "function": deepcopy(tool)})
    return normalized_tools


def normalize_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized_message: Dict[str, Any] = {
            "role": str(message.get("role", "")).lower(),
            "content": "" if message.get("content") is None else str(message.get("content")),
        }
        if message.get("tool_calls") is not None:
            normalized_message["tool_calls"] = deepcopy(message.get("tool_calls"))
        if message.get("tool_call_id") is not None:
            normalized_message["tool_call_id"] = str(message.get("tool_call_id"))
        if message.get("name") is not None:
            normalized_message["name"] = str(message.get("name"))
        normalized_messages.append(normalized_message)
    return normalized_messages


def render_chat(
    *,
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    chat_template: Optional[str],
    add_generation_prompt: bool,
) -> str:
    render_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if chat_template is not None:
        render_kwargs["chat_template"] = chat_template
    if tools:
        try:
            return tokenizer.apply_chat_template(messages, tools=tools, **render_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, custom_tools=tools, **render_kwargs)
    return tokenizer.apply_chat_template(messages, **render_kwargs)


def parse_rendered_messages(rendered_text: str) -> List[Message]:
    rendered_text = rendered_text.rstrip("\n")
    matches = list(_BLOCK_PATTERN.finditer(rendered_text))
    if not matches:
        raise RuntimeError(f"Failed to parse rendered chat blocks: {rendered_text!r}")

    rendered_messages: List[Message] = []
    for match in matches:
        role = match.group("role").strip().lower()
        if role == "assistant":
            rendered_messages.append(Message(content=match.group("prefix"), is_masked=True))
            rendered_messages.append(Message(content=match.group("content"), is_masked=False))
        else:
            rendered_messages.append(Message(content=match.group("full"), is_masked=True))

    if "".join(message.content for message in rendered_messages) != rendered_text:
        raise RuntimeError(f"Failed to reconstruct rendered chat blocks exactly: {rendered_text!r}")
    return rendered_messages


def convert_openai_trace_to_messages(
    *,
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    chat_template: Optional[str],
    add_generation_prompt: bool,
) -> List[Message]:
    if not messages:
        return []
    rendered_text = render_chat(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    return parse_rendered_messages(rendered_text)


def extract_compat_response(normalized_messages: List[Dict[str, Any]]) -> str:
    for message in reversed(normalized_messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def count_tokens(tokenizer, messages: List[Message]) -> Tuple[int, int]:
    completion_tokens = 0
    total_tokens = 0
    for message in messages:
        token_count = len(tokenizer.encode(message.content or "", add_special_tokens=False))
        total_tokens += token_count
        if not message.is_masked:
            completion_tokens += token_count
    return completion_tokens, total_tokens
