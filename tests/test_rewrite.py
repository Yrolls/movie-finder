import importlib.util
import json
from pathlib import Path

from movie_finder.evaluation import GoldQuery
from movie_finder.retrieval.rewrite import rewrite_query

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_rewrite_query_parses_and_repairs(fake_client):
    client = fake_client(["sure!", json.dumps({"plot": "A weatherman relives the same winter day."})])
    assert rewrite_query("ผู้ชายเจอวันเดิมซ้ำ", client=client).startswith("A weatherman")
    assert client.calls[0]["temperature"] == 0.0


def test_title_leak_check():
    gold = [GoldQuery("q1", "orig", "The Truman Show", 1998, []),
            GoldQuery("q2", "orig2", "Up", 2009, [])]
    rewrites = {"orig": "This is The Truman Show: a man's life is a TV show.",
                "orig2": "An old man ties balloons to his house and floats up."}
    assert run_eval.title_leaks(gold, rewrites) == ["q1"]           # one-word titles ignored
