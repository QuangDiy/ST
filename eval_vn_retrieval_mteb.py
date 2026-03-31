import argparse
import json
from pathlib import Path

from mteb import MTEB
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mteb_results")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["VieQuADRetrieval", "GreenNodeTableMarkdownRetrieval"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def collect_main_scores(output_dir: Path, tasks: list[str]) -> dict:
    summary = {}
    for task in tasks:
        task_dir = output_dir / task
        if not task_dir.exists():
            continue
        result_files = sorted(task_dir.glob("*.json"))
        if not result_files:
            continue
        payload = json.loads(result_files[0].read_text(encoding="utf-8"))
        split_scores = payload.get("scores", {})
        for split_name, split_metrics in split_scores.items():
            hf_subset = (
                split_metrics[0]
                if isinstance(split_metrics, list) and split_metrics
                else split_metrics
            )
            main_score = hf_subset.get("main_score")
            if main_score is None:
                main_score = hf_subset.get("ndcg_at_10")
            summary[f"{task}:{split_name}"] = main_score
    return summary


def main() -> None:
    args = parse_args()

    model = SentenceTransformer(args.model)
    evaluation = MTEB(tasks=args.tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation.run(model, output_folder=str(output_dir), batch_size=args.batch_size)
    summary = collect_main_scores(output_dir, args.tasks)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
