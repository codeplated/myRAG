"""Measure how well the pipeline finds the right passages and how faithful its answers are.

Each test question names the document that holds the answer (or several, when
more than one document states the fact) and one or more short evidence phrases
copied from it. A retrieved chunk counts as relevant when it comes from one of
those documents and contains one of the phrases. Matching on
phrases instead of chunk ids keeps the test set valid when chunking changes.

Retrieval metrics (per configuration, over the top k chunks):
  hit rate  share of questions with at least one relevant chunk in the top k
  MRR       mean of 1 / rank of the first relevant chunk (0 when none is found)

Answer metric (optional, needs an LLM):
  faithfulness  share of the claims in an answer that the retrieved chunks support,
                judged by an LLM and averaged over the questions

Usage:
    python -m rag_pipeline.evaluate --questions eval/questions.jsonl --answers
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import Settings, get_settings
from .llm import LLM, get_llm
from .query import SYSTEM_PROMPT, build_context, build_prompt
from .reranker import rerank
from .retrieval import RetrievedChunk, retrieve

# name -> (search mode, rerank mode)
CONFIGS: dict[str, tuple[str, str]] = {
    "vector": ("vector", "none"),
    "keyword": ("keyword", "none"),
    "hybrid": ("hybrid", "none"),
    "hybrid+rerank": ("hybrid", "llm"),
}

_JUDGE_SYSTEM = "You check whether statements are supported by source text. Reply with JSON only."


@dataclass
class EvalQuestion:
    id: str
    question: str
    book_ids: List[str]  # documents that contain the answer; usually one
    evidence: List[str]
    reference: str = ""


@dataclass
class QuestionResult:
    id: str
    rank: int | None  # 1-based rank of the first relevant chunk, None if not in the top k
    seconds: float
    answer: str | None = None
    faithfulness: float | None = None
    retrieved: List[str] = field(default_factory=list)


def load_questions(path: Path) -> List[EvalQuestion]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not row.get("evidence"):
            raise ValueError(f"{path}:{line_number}: question '{row.get('id')}' has no evidence phrases")
        questions.append(
            EvalQuestion(
                id=str(row["id"]),
                question=row["question"],
                book_ids=[row["book_id"]] if isinstance(row["book_id"], str) else list(row["book_id"]),
                evidence=list(row["evidence"]),
                reference=row.get("reference", ""),
            )
        )
    return questions


def normalize(text: str) -> str:
    """Lower-case and reduce every run of non-alphanumeric characters to one space."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_relevant(chunk: RetrievedChunk, question: EvalQuestion) -> bool:
    if chunk.book_id not in question.book_ids:
        return False
    content = normalize(chunk.content)
    return any(normalize(phrase) in content for phrase in question.evidence)


def first_relevant_rank(chunks: List[RetrievedChunk], question: EvalQuestion, k: int) -> int | None:
    for rank, chunk in enumerate(chunks[:k], start=1):
        if is_relevant(chunk, question):
            return rank
    return None


def retrieval_metrics(ranks: List[int | None]) -> dict[str, float]:
    if not ranks:
        return {"hit_rate": 0.0, "mrr": 0.0}
    hits = [r for r in ranks if r is not None]
    return {
        "hit_rate": len(hits) / len(ranks),
        "mrr": sum(1.0 / r for r in hits) / len(ranks),
    }


def parse_judgement(reply: str) -> float | None:
    """Share of supported claims in the judge's reply, or None if it lists no claims."""
    verdicts: List[bool] = []
    match = re.search(r"\{.*\}", reply or "", flags=re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group(0)).get("claims", [])
            verdicts = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
        except (json.JSONDecodeError, AttributeError):
            verdicts = []
    if not verdicts:  # the reply was not valid JSON, so read the verdicts directly
        verdicts = [v == "true" for v in re.findall(r'"supported"\s*:\s*(true|false)', reply or "")]
    if not verdicts:
        return None
    return sum(verdicts) / len(verdicts)


def judge_faithfulness(llm: LLM, answer: str, chunks: List[RetrievedChunk]) -> float | None:
    """Ask the LLM which claims in the answer are backed by the retrieved chunks."""
    prompt = (
        "Sources:\n" + build_context(chunks) + "\n\n"
        "Answer to check:\n" + answer + "\n\n"
        "List every factual claim the answer makes. For each one, decide whether the sources "
        "support it. Ignore citation numbers and statements that the answer was not found. "
        'Reply as JSON: {"claims": [{"claim": "...", "supported": true}]}'
    )
    return parse_judgement(llm.generate(prompt, system=_JUDGE_SYSTEM))


def evaluate_config(
    settings: Settings,
    questions: List[EvalQuestion],
    config: str,
    k: int,
    llm: LLM | None,
    with_answers: bool,
) -> dict:
    """Run every question through one configuration and return metrics plus per-question details."""
    search_mode, rerank_mode = CONFIGS[config]
    cfg = replace(settings, rerank_mode=rerank_mode, rerank_top_k=k)
    results: List[QuestionResult] = []

    for q in questions:
        start = time.perf_counter()
        candidates = retrieve(cfg, q.question, mode=search_mode)
        chunks = rerank(cfg, q.question, candidates, llm=llm)[:k]
        result = QuestionResult(
            id=q.id,
            rank=first_relevant_rank(chunks, q, k),
            seconds=round(time.perf_counter() - start, 3),
            retrieved=[f"{c.book_id} p{c.page_start}-{c.page_end}" for c in chunks],
        )
        if with_answers and llm is not None and chunks:
            result.answer = llm.generate(build_prompt(cfg, q.question, chunks), system=SYSTEM_PROMPT)
            result.faithfulness = judge_faithfulness(llm, result.answer, chunks)
        results.append(result)

    metrics = retrieval_metrics([r.rank for r in results])
    metrics["avg_search_seconds"] = round(statistics.mean(r.seconds for r in results), 3) if results else 0.0
    judged = [r.faithfulness for r in results if r.faithfulness is not None]
    if judged:
        metrics["faithfulness"] = sum(judged) / len(judged)
        metrics["answers_judged"] = len(judged)
    return {"config": config, "k": k, "metrics": metrics, "questions": [r.__dict__ for r in results]}


def format_table(runs: List[dict], k: int) -> str:
    """Results as a Markdown table."""
    lines = [
        f"| Configuration | Hit rate@{k} | MRR@{k} | Faithfulness | Avg search time |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        m = run["metrics"]
        faithfulness = f"{m['faithfulness']:.2f}" if "faithfulness" in m else "not measured"
        lines.append(
            f"| {run['config']} | {m['hit_rate']:.2f} | {m['mrr']:.2f} | {faithfulness} "
            f"| {m['avg_search_seconds']:.2f} s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument("--questions", default="eval/questions.jsonl", help="JSONL file with test questions.")
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help=f"Comma-separated configurations to compare. Available: {', '.join(CONFIGS)}.",
    )
    parser.add_argument("--k", type=int, default=5, help="How many top chunks count (default 5).")
    parser.add_argument(
        "--answers",
        action="store_true",
        help="Also generate answers and judge their faithfulness (slower, uses the LLM).",
    )
    parser.add_argument(
        "--answer-configs",
        default="",
        help="With --answers: only generate answers for these configurations (default: all selected).",
    )
    parser.add_argument("--out", default="eval/results", help="Folder for the JSON result file.")
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown configuration(s): {', '.join(unknown)}")

    settings = get_settings()
    questions = load_questions(Path(args.questions))
    needs_llm = args.answers or any(CONFIGS[c][1] == "llm" for c in configs)
    llm = get_llm(settings) if needs_llm else None

    answer_configs = [c.strip() for c in args.answer_configs.split(",") if c.strip()]
    runs = []
    for config in configs:
        print(f"Running '{config}' on {len(questions)} questions...", flush=True)
        with_answers = args.answers and (not answer_configs or config in answer_configs)
        runs.append(evaluate_config(settings, questions, config, args.k, llm, with_answers))

    table = format_table(runs, args.k)
    print("\n" + table)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "created_at": stamp,
                "questions_file": args.questions,
                "num_questions": len(questions),
                "embedding_model": settings.embedding_model,
                "llm": f"{settings.llm_provider}/{settings.active_llm_model}" if llm else None,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDetails saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
