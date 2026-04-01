import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import bm25s
from datasets import Dataset, load_dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.models import Normalize, Pooling, Transformer
from sentence_transformers.training_args import BatchSamplers
from tqdm.auto import tqdm


logging.getLogger("bm25s").setLevel(logging.WARNING)
ENABLE_TQDM = sys.stderr.isatty()


GREENNODE_BASE_URL = "https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="QuangDuy/bert-tiny-stage2-hf"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/bert-tiny-stage2-static-bm25-vn",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=768)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--loss_mini_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_hard_negatives", type=int, default=7)
    parser.add_argument("--bm25_candidate_pool", type=int, default=64)
    parser.add_argument("--bm25_query_batch_size", type=int, default=2048)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    parser.add_argument("--max_eval_queries", type=int, default=2048)
    parser.add_argument("--greennode_eval_queries", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25_cache_dir", type=str, default=None)
    parser.add_argument("--overwrite_bm25_cache", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def serialize_document(title: str | None, text: str) -> str:
    title = clean_text(title or "")
    text = clean_text(text)
    if not title or title.lower() == "none":
        return text
    return f"{title}\n{text}" if text else title


def build_sentence_transformer(
    model_name: str, max_seq_length: int
) -> SentenceTransformer:
    transformer = Transformer(model_name, max_seq_length=max_seq_length)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    normalize = Normalize()
    return SentenceTransformer(modules=[transformer, pooling, normalize])


def load_json_dataset(url: str) -> Dataset:
    return load_dataset("json", data_files=url, split="train")


def deduplicate_pairs(
    queries: list[str], positives: list[str]
) -> tuple[list[str], list[str]]:
    seen = set()
    dedup_queries = []
    dedup_positives = []
    for query, positive in zip(queries, positives):
        key = (query, positive)
        if key in seen:
            continue
        seen.add(key)
        dedup_queries.append(query)
        dedup_positives.append(positive)
    return dedup_queries, dedup_positives


def load_greennode_resources() -> tuple[
    dict[str, str], dict[str, str], dict[str, dict[str, int]]
]:
    corpus_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/corpus.jsonl")
    queries_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/queries.jsonl")
    qrels_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/qrels/train.jsonl")

    corpus_by_id = {
        row["_id"]: serialize_document(row.get("title"), row["text"])
        for row in corpus_ds
    }
    query_by_id = {row["_id"]: clean_text(row["text"]) for row in queries_ds}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)

    for row in qrels_ds:
        if row.get("score", 0) <= 0:
            continue
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        if query_id not in query_by_id or corpus_id not in corpus_by_id:
            continue
        qrels_by_query[query_id][corpus_id] = int(row.get("score", 1))

    return corpus_by_id, query_by_id, qrels_by_query


def build_greennode_train_and_evaluator(
    eval_query_count: int,
    seed: int,
) -> tuple[
    list[str], list[str], list[str], InformationRetrievalEvaluator, dict[str, int]
]:
    corpus_by_id, query_by_id, qrels_by_query = load_greennode_resources()
    all_query_ids = list(qrels_by_query)
    rng = random.Random(seed)
    eval_query_count = min(eval_query_count, len(all_query_ids))
    eval_query_ids = set(rng.sample(all_query_ids, eval_query_count))

    train_queries = []
    train_positives = []
    relevant_docs = {}
    eval_queries = {}

    for query_id, doc_scores in qrels_by_query.items():
        query_text = query_by_id[query_id]
        if query_id in eval_query_ids:
            eval_queries[query_id] = query_text
            relevant_docs[query_id] = {corpus_id: 1 for corpus_id in doc_scores}
            continue

        for corpus_id in doc_scores:
            train_queries.append(query_text)
            train_positives.append(corpus_by_id[corpus_id])

    evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=corpus_by_id,
        relevant_docs=relevant_docs,
        name="greennode-train-dev",
        show_progress_bar=False,
    )
    stats = {
        "greennode_train_pairs": len(train_queries),
        "greennode_eval_queries": len(eval_queries),
        "greennode_corpus_docs": len(corpus_by_id),
    }

    return train_queries, train_positives, list(corpus_by_id.values()), evaluator, stats


def load_viquad_training_data() -> tuple[list[str], list[str], list[str]]:
    dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="train")

    queries = []
    positives = []
    corpus = set()
    for row in dataset:
        if row.get("is_impossible"):
            continue
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue
        query_text = clean_text(row["question"])
        positive_text = serialize_document(row.get("title"), row["context"])
        if not query_text or not positive_text:
            continue
        queries.append(query_text)
        positives.append(positive_text)
        corpus.add(positive_text)

    return queries, positives, list(corpus)


def build_train_data_and_evaluator(
    max_train_pairs: int | None,
    viquad_eval_queries: int,
    greennode_eval_queries: int,
    seed: int,
) -> tuple[Dataset, list[str], dict[str, int], SequentialEvaluator]:
    gn_queries, gn_positives, gn_corpus, gn_evaluator, gn_stats = (
        build_greennode_train_and_evaluator(
            eval_query_count=greennode_eval_queries,
            seed=seed,
        )
    )
    vq_queries, vq_positives, vq_corpus = load_viquad_training_data()

    queries = gn_queries + vq_queries
    positives = gn_positives + vq_positives
    queries, positives = deduplicate_pairs(queries, positives)

    if max_train_pairs is not None:
        queries = queries[:max_train_pairs]
        positives = positives[:max_train_pairs]

    corpus = list(dict.fromkeys(gn_corpus + vq_corpus + positives))
    train_dataset = Dataset.from_dict({"query": queries, "positive": positives})
    viquad_evaluator = build_viquad_dev_evaluator(
        max_eval_queries=viquad_eval_queries,
        seed=seed,
    )
    combined_evaluator = SequentialEvaluator(
        [viquad_evaluator, gn_evaluator],
        main_score_function=lambda scores: sum(scores) / len(scores),
    )
    stats = {
        **gn_stats,
        "viquad_pairs": len(vq_queries),
        "train_pairs": len(train_dataset),
        "corpus_size": len(corpus),
    }
    return train_dataset, corpus, stats, combined_evaluator


def build_viquad_dev_evaluator(
    max_eval_queries: int, seed: int
) -> InformationRetrievalEvaluator:
    dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="validation")

    rows = []
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        if row.get("is_impossible") or not answers:
            continue
        rows.append(
            {
                "title": row.get("title"),
                "question": clean_text(row["question"]),
                "context": serialize_document(row.get("title"), row["context"]),
                "answer": clean_text(answers[0]),
            }
        )

    rng = random.Random(seed)
    if max_eval_queries is not None and len(rows) > max_eval_queries:
        rows = rng.sample(rows, max_eval_queries)

    queries = {}
    corpus = {}
    relevant_docs = {}
    text_to_id = {}
    next_id = 0

    for row in rows:
        query_id = str(next_id)
        queries[query_id] = row["question"]
        next_id += 1

        relevant_docs[query_id] = {}
        for text in [row["context"], row["answer"]]:
            if not text:
                continue
            if text not in text_to_id:
                corpus_id = str(next_id)
                text_to_id[text] = corpus_id
                corpus[corpus_id] = {
                    "title": clean_text(row.get("title") or ""),
                    "text": text,
                }
                next_id += 1
            relevant_docs[query_id][text_to_id[text]] = 1

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="uit-viquad2-validation",
        show_progress_bar=False,
    )


def get_cache_dir(output_dir: Path, args: argparse.Namespace) -> Path:
    if args.bm25_cache_dir:
        return Path(args.bm25_cache_dir)
    return output_dir / "bm25_static_cache"


def group_positives_by_query(train_dataset: Dataset) -> dict[str, set[str]]:
    query_to_positives: dict[str, set[str]] = defaultdict(set)
    for row in train_dataset:
        query_to_positives[row["query"]].add(row["positive"])
    return query_to_positives


def iter_static_bm25_rows(
    train_dataset: Dataset,
    negative_lookup: dict[str, list[str]],
) -> dict[str, str]:
    progress_bar = tqdm(
        train_dataset,
        total=len(train_dataset),
        desc="Cache rows",
        unit="row",
        mininterval=2.0,
        dynamic_ncols=True,
        disable=not ENABLE_TQDM,
    )
    for row in progress_bar:
        record = {
            "query": row["query"],
            "positive": row["positive"],
        }
        for idx, negative in enumerate(negative_lookup[row["query"]], start=1):
            record[f"negative_{idx}"] = negative

        yield record


def build_static_bm25_dataset(
    train_dataset: Dataset,
    corpus: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Dataset, dict[str, int]]:
    cache_dir = get_cache_dir(output_dir, args)
    if cache_dir.exists() and not args.overwrite_bm25_cache:
        cached_dataset = load_from_disk(str(cache_dir))
        cached_stats = {
            "bm25_candidate_pool": args.bm25_candidate_pool,
            "num_hard_negatives": args.num_hard_negatives,
            "fallback_queries": -1,
            "cached_train_rows": len(cached_dataset),
        }
        return cached_dataset, cached_stats

    query_to_positives = group_positives_by_query(train_dataset)
    unique_queries = list(query_to_positives)
    candidate_pool = min(
        len(corpus),
        max(args.bm25_candidate_pool, args.num_hard_negatives + 8),
    )

    print(f"[info] Building BM25 index over {len(corpus)} documents...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords=[], show_progress=False)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=False)

    rng = random.Random(args.seed)
    negative_lookup: dict[str, list[str]] = {}
    fallback_queries = 0
    batch_size = max(1, args.bm25_query_batch_size)

    print(
        f"[info] Retrieving BM25 candidates for {len(unique_queries)} unique queries "
        f"in batches of {batch_size}..."
    )
    total_batches = (len(unique_queries) + batch_size - 1) // batch_size
    batch_iterator = tqdm(
        range(0, len(unique_queries), batch_size),
        total=total_batches,
        desc="BM25 batches",
        unit="batch",
        mininterval=2.0,
        dynamic_ncols=True,
        disable=not ENABLE_TQDM,
    )

    for batch_start in batch_iterator:
        batch_end = min(batch_start + batch_size, len(unique_queries))
        batch_queries = unique_queries[batch_start:batch_end]
        batch_iterator.set_postfix_str(
            f"queries {batch_start + 1}-{batch_end}",
            refresh=False,
        )
        query_tokens = bm25s.tokenize(
            batch_queries,
            stopwords=[],
            show_progress=False,
        )
        doc_ids, _ = retriever.retrieve(
            query_tokens,
            k=candidate_pool,
            show_progress=False,
        )

        for batch_offset, query in enumerate(batch_queries):
            positives = query_to_positives[query]
            negatives: list[str] = []

            for corpus_idx in doc_ids[batch_offset]:
                candidate = corpus[int(corpus_idx)]
                if candidate in positives or candidate in negatives:
                    continue
                negatives.append(candidate)
                if len(negatives) == args.num_hard_negatives:
                    break

            if len(negatives) < args.num_hard_negatives:
                fallback_queries += 1
                max_attempts = max(len(corpus) * 2, 1000)
                attempts = 0
                while (
                    len(negatives) < args.num_hard_negatives and attempts < max_attempts
                ):
                    candidate = corpus[rng.randrange(len(corpus))]
                    attempts += 1
                    if candidate in positives or candidate in negatives:
                        continue
                    negatives.append(candidate)

            if len(negatives) != args.num_hard_negatives:
                raise RuntimeError(
                    f"Could not mine enough negatives for query: {query}"
                )

            negative_lookup[query] = negatives

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = cache_dir.parent / "bm25_static_cache.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    print(
        f"[info] BM25 retrieval complete. Building static training dataset for {len(train_dataset)} rows..."
    )
    print(f"[info] Writing mined examples to {jsonl_path}...")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in iter_static_bm25_rows(train_dataset, negative_lookup):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[info] Converting JSONL cache to Hugging Face Dataset...")
    static_dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"[info] Saving static BM25 cache to {cache_dir}...")
    static_dataset.save_to_disk(str(cache_dir))
    print("[info] Static BM25 cache saved.")
    jsonl_path.unlink(missing_ok=True)

    stats = {
        "bm25_candidate_pool": candidate_pool,
        "num_hard_negatives": args.num_hard_negatives,
        "fallback_queries": fallback_queries,
        "cached_train_rows": len(static_dataset),
    }
    return static_dataset, stats


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    data_stats: dict[str, int],
    mining_stats: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "data": data_stats,
        "mining": mining_stats,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_evaluator_label(evaluator: InformationRetrievalEvaluator) -> str:
    name = evaluator.name.lower()
    if "viquad" in name:
        return "viquad"
    if "greennode" in name:
        return "greennode"
    return name.replace("-", "_")


def build_eval_record(
    evaluator: InformationRetrievalEvaluator | SequentialEvaluator,
    metrics: dict[str, float],
    stage_name: str,
    output_path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage_name,
        "output_path": str(output_path),
        "sequential_score": metrics.get("sequential_score"),
    }

    subevaluators = (
        list(evaluator.evaluators)
        if isinstance(evaluator, SequentialEvaluator)
        else [evaluator]
    )
    for subevaluator in subevaluators:
        label = get_evaluator_label(subevaluator)
        primary_metric = getattr(subevaluator, "primary_metric", None)
        if primary_metric and primary_metric in metrics:
            record[f"{label}_primary_metric"] = primary_metric
            record[f"{label}_primary_score"] = metrics[primary_metric]

    return record


def append_eval_summary(output_dir: Path, record: dict[str, Any]) -> None:
    jsonl_path = output_dir / "evaluation_summary.jsonl"
    markdown_path = output_dir / "evaluation_summary.md"

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    headers = [
        "stage",
        "sequential_score",
        "viquad_primary_score",
        "greennode_primary_score",
        "viquad_primary_metric",
        "greennode_primary_metric",
    ]
    lines = [
        "# Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in records:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_eval_summary(record: dict[str, Any]) -> None:
    parts = [f"stage={record['stage']}"]
    if record.get("viquad_primary_score") is not None:
        parts.append(f"viquad={record['viquad_primary_score']:.6f}")
    if record.get("greennode_primary_score") is not None:
        parts.append(f"greennode={record['greennode_primary_score']:.6f}")
    if record.get("sequential_score") is not None:
        parts.append(f"combined={record['sequential_score']:.6f}")
    print("[eval] " + " | ".join(parts))


def train_one_epoch(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: Path,
    args: argparse.Namespace,
    epoch_index: int,
) -> None:
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints" / f"epoch_{epoch_index}"),
        num_train_epochs=1,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to="none",
        run_name=f"static-bm25-vn-retrieval-epoch-{epoch_index}",
    )
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=args.loss_mini_batch_size,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()


def run_dev_evaluation(
    evaluator: InformationRetrievalEvaluator | SequentialEvaluator,
    model: SentenceTransformer,
    output_dir: Path,
    stage_name: str,
    output_path: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = evaluator(model, output_path=str(output_path))
    record = build_eval_record(evaluator, metrics, stage_name, output_path)
    append_eval_summary(output_dir, record)
    print_eval_summary(record)
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    model = build_sentence_transformer(args.model_name, args.max_seq_length)

    base_train_dataset, corpus, data_stats, dev_evaluator = (
        build_train_data_and_evaluator(
            max_train_pairs=args.max_train_pairs,
            viquad_eval_queries=args.max_eval_queries,
            greennode_eval_queries=args.greennode_eval_queries,
            seed=args.seed,
        )
    )
    static_train_dataset, mining_stats = build_static_bm25_dataset(
        base_train_dataset,
        corpus,
        output_dir,
        args,
    )
    save_run_config(output_dir, args, data_stats, mining_stats)

    print(
        json.dumps(
            {"data": data_stats, "mining": mining_stats},
            indent=2,
            ensure_ascii=False,
        )
    )

    print("[info] Running initial retrieval evaluation on ViQuAD + GreenNode dev...")
    run_dev_evaluation(
        dev_evaluator,
        model,
        output_dir,
        "initial",
        output_dir / "eval_initial",
    )

    print("[info] Starting training with static BM25 hard negatives...")
    for epoch_index in range(1, args.epochs + 1):
        print(f"[info] Training epoch {epoch_index}/{args.epochs}...")
        train_one_epoch(model, static_train_dataset, output_dir, args, epoch_index)
        print(f"[info] Running retrieval evaluation after epoch {epoch_index}...")
        run_dev_evaluation(
            dev_evaluator,
            model,
            output_dir,
            f"epoch_{epoch_index}",
            output_dir / f"eval_epoch_{epoch_index}",
        )

    print("[info] Running final retrieval evaluation...")
    run_dev_evaluation(
        dev_evaluator,
        model,
        output_dir,
        "final",
        output_dir / "eval_final",
    )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[done] Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
