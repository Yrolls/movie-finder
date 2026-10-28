"""The only module that talks to the LLM proxy.

- `chat()`       : plain chat completion (optionally with tools), temperature 0 by default
- `chat_json()`  : ask for JSON, parse it, validate against a Pydantic schema, and on failure
                   send the validation error back to the model for a limited number of repairs
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import get_settings

if TYPE_CHECKING:  # imported lazily so unit tests run without the SDK or a key
    from openai import OpenAI

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMOutputError(RuntimeError):
    """The model did not return valid output after all repair attempts."""


@lru_cache(maxsize=1)
def get_client() -> "OpenAI":
    from openai import OpenAI

    s = get_settings()
    s.require_llm()
    return OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    client: "OpenAI | None" = None,
):
    """Return the raw first choice message from the proxy."""
    client = client or get_client()
    kwargs: dict[str, Any] = dict(
        model=get_settings().llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model reply (handles ```json fences and chatter around it)."""
    if text is None:
        raise ValueError("empty reply")
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost {...} block
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        return json.loads(candidate[start : end + 1])
    raise ValueError("no JSON object found in reply")


def chat_json(
    messages: list[dict[str, Any]],
    schema: type[T],
    *,
    retries: int = 2,
    client: "OpenAI | None" = None,
) -> T:
    """Ask for JSON matching `schema`; repair up to `retries` times, then raise LLMOutputError."""
    convo = list(messages)
    last_err = ""
    for _ in range(retries + 1):
        msg = chat(convo, client=client)
        reply = msg.content or ""
        try:
            return schema.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as e:
            last_err = str(e)
            convo += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        "Your reply was not valid. Error:\n"
                        f"{last_err[:800]}\n"
                        "Reply again with ONLY one JSON object that matches the schema."
                    ),
                },
            ]
    raise LLMOutputError(f"invalid JSON after {retries + 1} attempts: {last_err[:300]}")
