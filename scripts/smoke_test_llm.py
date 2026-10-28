"""Check the class proxy before building anything on top of it.

    python scripts/smoke_test_llm.py

Runs three probes and prints PASS / FAIL for each:
  1. plain chat in Thai              -> does the model understand Thai descriptions?
  2. JSON output + schema validation -> can we rely on structured output?
  3. native tool calling             -> does the proxy accept `tools=`? (decides the agent mode)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel  # noqa: E402

from movie_finder.config import get_settings  # noqa: E402
from movie_finder.llm import chat, chat_json  # noqa: E402


class MovieGuess(BaseModel):
    title: str
    year: int
    reason_th: str


def probe(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s)\n       {out}\n")
        return True
    except Exception as e:  # noqa: BLE001 -- a smoke test should report, not crash
        print(f"[FAIL] {name}: {type(e).__name__}: {e}\n")
        return False


def thai_chat() -> str:
    msg = chat([{"role": "user", "content": "สรุปเรื่องย่อหนัง Titanic (1997) ใน 2 ประโยค เป็นภาษาไทย"}])
    return (msg.content or "").strip()[:300]


def json_output() -> str:
    guess = chat_json(
        [
            {
                "role": "system",
                "content": "Reply with ONLY a JSON object: "
                '{"title": str, "year": int, "reason_th": str (Thai, one sentence)}',
            },
            {"role": "user", "content": "หนังที่ผู้ชายตื่นมาแล้วเจอวันเดิมซ้ำทุกวัน ในเมืองเล็ก ๆ ช่วงหน้าหนาว"},
        ],
        MovieGuess,
    )
    return guess.model_dump_json()


def tool_calling() -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_movies",
                "description": "Search movie plots by description.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    msg = chat(
        [{"role": "user", "content": "Find the movie where dreams are nested inside dreams. Use the tool."}],
        tools=tools,
    )
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        raise RuntimeError(f"no tool_calls returned; content was: {(msg.content or '')[:200]!r}")
    c = calls[0].function
    return f"{c.name}({json.loads(c.arguments)})"


def main() -> None:
    s = get_settings()
    s.require_llm()
    print(f"model={s.llm_model}  base_url={s.openai_base_url}\n")
    results = [
        probe("1. Thai chat", thai_chat),
        probe("2. JSON + schema", json_output),
        probe("3. native tool calling", tool_calling),
    ]
    if not results[2]:
        print("Note: if (3) fails but (2) passes, the agent will use JSON-action mode instead "
              "of native tool calls -- tell me and I'll switch it.")
    sys.exit(0 if all(results[:2]) else 1)


if __name__ == "__main__":
    main()
