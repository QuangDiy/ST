# Vietnamese Retrieval Training

Scripts in this folder train a single-vector Vietnamese retrieval model from `QuangDuy/bert-tiny-stage2-hf` with static BM25 hard negatives.

- train pairs come from `GreenNode/GreenNode-Table-Markdown-Retrieval-VN` and `taidng/UIT-ViQuAD2.0`
- hard negatives are mined once with BM25 and then reused for the full run
- training uses `CachedMultipleNegativesRankingLoss`
- final evaluation runs on `VieQuADRetrieval` and `GreenNodeTableMarkdownRetrieval`

## Install

The project now pins package versions in `pyproject.toml`, including `transformers==4.51.3`.

If you need a CUDA-specific PyTorch build, install that first from the PyTorch index that matches your GPU, then install the project:

```bash
pip install -e .
```

## Train

```bash
python train_vn_retrieval_bm25.py \
  --model_name QuangDuy/bert-tiny-stage2-hf \
  --output_dir output/bert-tiny-stage2-static-bm25-vn \
  --epochs 3 \
  --num_hard_negatives 5 \
  --bm25_candidate_pool 64 \
  --per_device_train_batch_size 64 \
  --loss_mini_batch_size 16
```

Useful flags:

- `--max_seq_length 768` keeps more GreenNode context
- `--max_train_pairs N` is handy for smoke tests
- `--overwrite_bm25_cache` rebuilds the static BM25 negatives
- `--fp16` or `--bf16` depends on your GPU

The script uses:

- `GreenNode` qrels train split for supervised retrieval pairs
- `UIT-ViQuAD2.0` train split with `question -> context` positives
- BM25 negatives from the merged retrieval corpus
- `UIT-ViQuAD2.0` validation split for quick dev retrieval evaluation

## Evaluate

```bash
python eval_vn_retrieval_mteb.py \
  --model output/bert-tiny-stage2-static-bm25-vn/final \
  --output_dir output/bert-tiny-stage2-static-bm25-vn/mteb
```

The evaluation script runs MTEB on:

- `VieQuADRetrieval`
- `GreenNodeTableMarkdownRetrieval`

## Notes

- `GreenNode` has a slightly awkward Hub layout, so the training script reads the raw `corpus.jsonl`, `queries.jsonl`, and `qrels/train.jsonl` files directly.
- The model is wrapped as `Transformer + mean pooling + Normalize`, so it becomes a proper single-vector sentence-transformer model.
- BM25 negatives are mined once and cached to disk in `output/.../bm25_static_cache` by default, which keeps repeated experiments much faster.
