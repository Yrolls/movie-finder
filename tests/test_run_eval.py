"""End-to-end check of the retrieval eval + report on the tiny fixture corpus (no LLM)."""
import importlib.util
from pathlib import Path

from movie_finder.evaluation import GoldQuery
from tests.fixtures import make_retriever

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

GOLD = [
    GoldQuery("q1", "man relives the same winter day", "Loop Day", 1993, ["en"], "m00000"),
    GoldQuery("q2", "thief steals secrets inside dreams", "Dream Heist", 2010, ["en"], "m00001"),
    GoldQuery("q3", "quantum chromodynamics lattice", None, None, ["unanswerable"]),
]


def test_retrieval_eval_and_report():
    r = make_retriever()
    out = run_eval.eval_retrieval(r, GOLD, ["bm25", "dense"])
    assert out["dense"]["overall"]["recall@1"] == 1.0
    sweep = run_eval.sweep_min_score(r, GOLD)
    assert sweep[0]["abstain_unanswerable"] == 0.0          # no floor -> junk query gets results
    assert sweep[-1]["abstain_unanswerable"] == 1.0         # high floor -> nothing returned

    report = run_eval.write_report({
        "generated": "now", "n_answerable": 2, "n_unanswerable": 1,
        "config": {"llm_model": "m", "embed_model": "e"},
        "retrieval": [{"index": "data/index", "meta": {"chunk_words": 12}, "modes": out}],
        "sweep": sweep,
    })
    assert "| bm25 (baseline) |" in report and "min_score" in report
