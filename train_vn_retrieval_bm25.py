import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import bm25s
from datasets import Dataset, load_dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.models import Normalize, Pooling, Transformer
from sentence_transformers.training_args import BatchSamplers


logging.getLogger("bm25s").setLevel(logging.WARNING)


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


def load_greennode_training_data() -> tuple[list[str], list[str], list[str]]:
    corpus_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/corpus.jsonl")
    queries_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/queries.jsonl")
    qrels_ds = load_json_dataset(f"{GREENNODE_BASE_URL}/qrels/train.jsonl")

    corpus_by_id = {
        row["_id"]: serialize_document(row.get("title"), row["text"])
        for row in corpus_ds
    }
    query_by_id = {row["_id"]: clean_text(row["text"]) for row in queries_ds}

    queries = []
    positives = []
    for row in qrels_ds:
        if row.get("score", 0) <= 0:
            continue
        query_text = query_by_id.get(row["query-id"])
        positive_text = corpus_by_id.get(row["corpus-id"])
        if not query_text or not positive_text:
            continue
        queries.append(query_text)
        positives.append(positive_text)

    return queries, positives, list(corpus_by_id.values())


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


def build_train_pairs(
    max_train_pairs: int | None = None,
) -> tuple[Dataset, list[str], dict[str, int]]:
    gn_queries, gn_positives, gn_corpus = load_greennode_training_data()
    vq_queries, vq_positives, vq_corpus = load_viquad_training_data()

    queries = gn_queries + vq_queries
    positives = gn_positives + vq_positives
    queries, positives = deduplicate_pairs(queries, positives)

    if max_train_pairs is not None:
        queries = queries[:max_train_pairs]
        positives = positives[:max_train_pairs]

    corpus = list(dict.fromkeys(gn_corpus + vq_corpus + positives))
    train_dataset = Dataset.from_dict({"query": queries, "positive": positives})
    stats = {
        "greennode_pairs": len(gn_queries),
        "viquad_pairs": len(vq_queries),
        "train_pairs": len(train_dataset),
        "corpus_size": len(corpus),
    }
    return train_dataset, corpus, stats


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
        show_progress_bar=True,
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
    corpus_tokens = bm25s.tokenize(corpus, stopwords=[])
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)

    rng = random.Random(args.seed)
    negative_lookup: dict[str, list[str]] = {}
    fallback_queries = 0
    batch_size = max(1, args.bm25_query_batch_size)

    print(
        f"[info] Retrieving BM25 candidates for {len(unique_queries)} unique queries "
        f"in batches of {batch_size}..."
    )

    for batch_start in range(0, len(unique_queries), batch_size):
        batch_end = min(batch_start + batch_size, len(unique_queries))
        batch_queries = unique_queries[batch_start:batch_end]
        print(
            f"[info] BM25 batch {batch_start // batch_size + 1}/"
            f"{(len(unique_queries) + batch_size - 1) // batch_size}: "
            f"queries {batch_start + 1}-{batch_end}"
        )
        query_tokens = bm25s.tokenize(batch_queries, stopwords=[])
        doc_ids, _ = retriever.retrieve(query_tokens, k=candidate_pool)

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

    dataset_dict = {"query": [], "positive": []}
    for idx in range(args.num_hard_negatives):
        dataset_dict[f"negative_{idx + 1}"] = []

    print(
        f"[info] BM25 retrieval complete. Building static training dataset for {len(train_dataset)} rows..."
    )
    for row_index, row in enumerate(train_dataset, start=1):
        dataset_dict["query"].append(row["query"])
        dataset_dict["positive"].append(row["positive"])
        negatives = negative_lookup[row["query"]]
        for idx, negative in enumerate(negatives, start=1):
            dataset_dict[f"negative_{idx}"].append(negative)
        if row_index % 10000 == 0 or row_index == len(train_dataset):
            print(f"[info] Materialized {row_index}/{len(train_dataset)} training rows")

    print("[info] Converting mined examples to Hugging Face Dataset...")
    static_dataset = Dataset.from_dict(dataset_dict)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(f"[info] Saving static BM25 cache to {cache_dir}...")
    static_dataset.save_to_disk(str(cache_dir))
    print("[info] Static BM25 cache saved.")

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


def train(
    model: SentenceTransformer,
    train_dataset: Dataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
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
        run_name="static-bm25-vn-retrieval",
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    model = build_sentence_transformer(args.model_name, args.max_seq_length)

    base_train_dataset, corpus, data_stats = build_train_pairs(
        max_train_pairs=args.max_train_pairs
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

    dev_evaluator = build_viquad_dev_evaluator(args.max_eval_queries, args.seed)
    print("[info] Running initial ViQuAD validation evaluation...")
    dev_evaluator(model, output_path=str(output_dir / "eval_initial"))

    print("[info] Starting training with static BM25 hard negatives...")
    train(model, static_train_dataset, output_dir, args)

    print("[info] Running final ViQuAD validation evaluation...")
    dev_evaluator(model, output_path=str(output_dir / "eval_final"))

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"[done] Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
