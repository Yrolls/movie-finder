"""Chunk the corpus and embed it into data/index/.

    python scripts/build_index.py                         # uses CHUNK_WORDS / CHUNK_OVERLAP from .env
    python scripts/build_index.py --chunk-words 100 --chunk-overlap 25 --out data/index_c100
    python scripts/build_index.py --limit 500             # quick dev run on the first 500 movies
    python scripts/build_index.py --embed-model BAAI/bge-small-en-v1.5 --out data/index_bge

Separate --out folders let eval/run_eval.py compare chunk sizes side by side.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_finder.config import INDEX_DIR, PROCESSED_DIR, PROJECT_ROOT, get_settings  # noqa: E402
from movie_finder.data.corpus import chunk_corpus, load_movies  # noqa: E402
from movie_finder.retrieval.embedder import SentenceTransformerEmbedder  # noqa: E402
from movie_finder.retrieval.vector_store import VectorStore  # noqa: E402


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-words", type=int, default=s.chunk_words)
    ap.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--out", type=Path, default=INDEX_DIR)
    ap.add_argument("--limit", type=int, help="only index the first N movies (dev)")
    ap.add_argument("--embed-model", default=s.embed_model,
                    help="sentence-transformers model, e.g. BAAI/bge-small-en-v1.5 (default: EMBED_MODEL)")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    movies = load_movies(PROCESSED_DIR / "movies.jsonl")
    if args.limit:
        movies = movies[: args.limit]
    chunks = chunk_corpus(movies, words=args.chunk_words, overlap=args.chunk_overlap)
    print(f"[index] {len(movies):,} movies -> {len(chunks):,} chunks "
          f"(words={args.chunk_words}, overlap={args.chunk_overlap})")

    t0 = time.perf_counter()
    emb = SentenceTransformerEmbedder(args.embed_model)
    vectors = emb.embed_passages([c.text for c in chunks])
    print(f"[index] embedded in {time.perf_counter() - t0:.0f}s with {args.embed_model}")

    VectorStore(
        vectors,
        chunks,
        meta={
            "embed_model": args.embed_model,
            "chunk_words": args.chunk_words,
            "chunk_overlap": args.chunk_overlap,
            "n_movies": len(movies),
            "n_chunks": len(chunks),
        },
    ).save(out)
    print(f"[index] saved -> {out}")


if __name__ == "__main__":
    main()
