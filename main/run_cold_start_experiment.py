import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_SEED = 622768884
DEFAULT_DATASET_PATH = r"D:\Code\LAMRec-RAGBK\mimic-iii-1.4\zip"
DEFAULT_OUTPUT_SUBDIR = "cold_start_experiment_622768884"

# Keep PyHealth/model cache writes inside this project instead of the user profile.
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_HOME = PROJECT_ROOT / "output" / DEFAULT_OUTPUT_SUBDIR / "cache_home"
os.environ["USERPROFILE"] = str(LOCAL_CACHE_HOME)
os.environ["HOME"] = str(LOCAL_CACHE_HOME)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.model_MLPMixerBK3 import LAMRec
from pyhealth.custom_dataset import PreprocessedDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.utils import set_seed


METRICS = [
    "jaccard_samples",
    "f1_samples",
    "pr_auc_samples",
    "roc_auc_samples",
    "ddi_score",
    "avg_med",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LAMRec medication-history cold-start behavior."
    )
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--gru_hidden_size", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_false",
        dest="save_predictions",
        help="Do not save y_true/y_prob arrays for every evaluation group.",
    )
    parser.set_defaults(save_predictions=True)
    return parser.parse_args()


def resolve_checkpoint(project_root: Path, seed: int, checkpoint: str = None) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"Specified checkpoint does not exist: {checkpoint_path}")

    preferred = [
        project_root / "output" / str(seed) / "best.ckpt",
        project_root
        / "output"
        / f"{seed}DIAGNOSES_Official512filtered_drug_embeddings_768"
        / "best.ckpt",
    ]
    for checkpoint_path in preferred:
        if checkpoint_path.exists():
            return checkpoint_path

    candidates = sorted(
        (project_root / "output").glob(f"**/{seed}*/best.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidate_text = "\n".join(str(path) for path in candidates[:20])
    raise FileNotFoundError(
        "No default checkpoint was found. Provide --checkpoint explicitly."
        + (f"\nAvailable seed-matched candidates:\n{candidate_text}" if candidate_text else "")
    )


def is_first_visit(sample: Dict) -> bool:
    visit_id = str(sample["visit_id"])
    visit_suffix_is_zero = visit_id.rsplit("_", 1)[-1] == "0"
    no_drug_history = sample.get("drugs_hist") == [[]]
    return visit_suffix_is_zero or no_drug_history


def build_test_groups(test_samples: List[Dict]) -> Tuple[Dict[str, List[Dict]], Dict]:
    first_visit_samples = [sample for sample in test_samples if is_first_visit(sample)]
    follow_up_samples = [sample for sample in test_samples if not is_first_visit(sample)]
    masked_samples = []
    for sample in test_samples:
        masked_sample = copy.deepcopy(sample)
        masked_sample["drugs_hist"] = [[]]
        masked_samples.append(masked_sample)

    groups = {
        "Overall": test_samples,
        "First-visit": first_visit_samples,
        "Follow-up": follow_up_samples,
        "All-history-masked": masked_samples,
    }
    stats = {
        "overall_samples": len(test_samples),
        "first_visit_samples": len(first_visit_samples),
        "follow_up_samples": len(follow_up_samples),
        "first_visit_ratio": len(first_visit_samples) / len(test_samples),
        "masked_samples": len(masked_samples),
        "test_patients": len({sample["patient_id"] for sample in test_samples}),
    }
    return groups, stats


def load_dataset(dataset_path: str) -> PreprocessedDataset:
    records_path = os.path.join(dataset_path, "records_final.pkl")
    voc_path = os.path.join(dataset_path, "voc_final.pkl")
    if not os.path.exists(records_path) or not os.path.exists(voc_path):
        raise FileNotFoundError(
            "records_final.pkl and voc_final.pkl must exist in --dataset_path."
        )
    return PreprocessedDataset(records_path=records_path, voc_path=voc_path)


def build_model(args: argparse.Namespace, sample_dataset: PreprocessedDataset) -> LAMRec:
    model = LAMRec(
        sample_dataset,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        alpha=args.alpha,
        bce_weight=args.bce_weight,
        gru_hidden_size=args.gru_hidden_size,
        gru_layers=args.gru_layers,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def smoke_test(model: torch.nn.Module, samples: List[Dict], batch_size: int, device: str, name: str) -> float:
    if not samples:
        raise ValueError(f"Smoke test group is empty: {name}")
    dataloader = get_dataloader(samples, batch_size=min(batch_size, len(samples)), shuffle=False)
    batch = next(iter(dataloader))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    return float(output["loss"].detach().cpu().item())


def evaluate_group(
    model: torch.nn.Module,
    samples: List[Dict],
    batch_size: int,
    device: str,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    if not samples:
        raise ValueError("Cannot evaluate an empty group.")

    dataloader = get_dataloader(samples, batch_size=batch_size, shuffle=False)
    loss_all = []
    y_true_all = []
    y_prob_all = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(**batch)
            loss_all.append(float(output["loss"].detach().cpu().item()))
            y_true_all.append(output["y_true"].detach().cpu().numpy())
            y_prob_all.append(output["y_prob"].detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    scores = multilabel_metrics_fn(y_true, y_prob, metrics=METRICS)
    scores["loss"] = float(np.mean(loss_all))
    scores = {key: float(value) for key, value in scores.items()}
    return scores, {"y_true": y_true, "y_prob": y_prob}


def sanitize_group_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def save_outputs(
    output_dir: Path,
    metrics_rows: List[Dict],
    split_stats: Dict,
    predictions: Dict[str, Dict[str, np.ndarray]],
    save_predictions: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "cold_start_metrics.csv", index=False)

    with open(output_dir / "cold_start_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    with open(output_dir / "cold_start_split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    if save_predictions:
        arrays = {}
        for group, group_predictions in predictions.items():
            prefix = sanitize_group_name(group)
            arrays[f"{prefix}_y_true"] = group_predictions["y_true"]
            arrays[f"{prefix}_y_prob"] = group_predictions["y_prob"]
        np.savez_compressed(output_dir / "cold_start_predictions.npz", **arrays)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "output" / f"cold_start_experiment_{args.seed}"
    )

    set_seed(args.seed)
    checkpoint_path = resolve_checkpoint(project_root, args.seed, args.checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using device: {args.device}")

    sample_dataset = load_dataset(args.dataset_path)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    del train_dataset, val_dataset

    test_indices = [int(index) for index in test_dataset.indices]
    test_samples = [sample_dataset.samples[index] for index in test_indices]
    groups, split_stats = build_test_groups(test_samples)
    split_stats.update(
        {
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "dataset_path": str(args.dataset_path),
            "batch_size": args.batch_size,
            "device": args.device,
        }
    )

    print("Split stats:")
    print(json.dumps(split_stats, indent=2))
    if split_stats["overall_samples"] != 1488:
        print("Warning: expected 1488 test samples for the reference split.")
    if split_stats["first_visit_samples"] != 635:
        print("Warning: expected 635 first-visit samples for the reference split.")
    if split_stats["follow_up_samples"] != 853:
        print("Warning: expected 853 follow-up samples for the reference split.")

    model = build_model(args, sample_dataset).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)

    smoke_results = {
        "First-visit": smoke_test(
            model, groups["First-visit"], args.batch_size, args.device, "First-visit"
        ),
        "All-history-masked": smoke_test(
            model,
            groups["All-history-masked"],
            args.batch_size,
            args.device,
            "All-history-masked",
        ),
    }
    split_stats["smoke_test_loss"] = smoke_results
    print("Smoke test loss:")
    print(json.dumps(smoke_results, indent=2))

    metrics_rows = []
    predictions = {}
    for group_name in ["Overall", "First-visit", "Follow-up", "All-history-masked"]:
        print(f"Evaluating {group_name} ({len(groups[group_name])} samples)...")
        scores, group_predictions = evaluate_group(
            model, groups[group_name], args.batch_size, args.device
        )
        row = {
            "group": group_name,
            "n_samples": len(groups[group_name]),
        }
        for metric in METRICS + ["loss"]:
            row[metric] = scores[metric]
        metrics_rows.append(row)
        predictions[group_name] = group_predictions
        print(json.dumps(row, indent=2))

    save_outputs(output_dir, metrics_rows, split_stats, predictions, args.save_predictions)
    print(f"Saved cold-start experiment outputs to: {output_dir}")
    print(pd.DataFrame(metrics_rows).to_string(index=False))


if __name__ == "__main__":
    main()
