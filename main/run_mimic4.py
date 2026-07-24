import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import torch
from pandarallel import pandarallel
from torch.utils.data import Subset

pandarallel.initialize(nb_workers=16, progress_bar=False)

sys.path.append("..")

from models.model_MLPMixerMIMIC3 import LAMRec
from pyhealth.custom_dataset import PreprocessedDataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.utils import set_seed
from trainerLogDrug import Trainer


def is_first_visit(sample: Dict) -> bool:
    return sample["visit_id"].endswith("_0") or sample.get("drugs_hist") == [[]]


def dataset_statistics(dataset: PreprocessedDataset) -> Dict[str, float]:
    samples = dataset.samples
    patients = {sample["patient_id"] for sample in samples}
    visit_counts: Dict[str, int] = {}
    for sample in samples:
        visit_counts[sample["patient_id"]] = visit_counts.get(sample["patient_id"], 0) + 1

    conditions = set()
    procedures = set()
    drugs = set()
    med_counts = []
    first_visits = 0
    for sample in samples:
        for visit_codes in sample["conditions"]:
            conditions.update(visit_codes)
        for visit_codes in sample["procedures"]:
            procedures.update(visit_codes)
        drugs.update(sample["drugs"])
        med_counts.append(len(set(sample["drugs"])))
        first_visits += int(is_first_visit(sample))

    return {
        "n_patients": len(patients),
        "n_visits": len(samples),
        "n_diagnoses": len(conditions),
        "n_procedures": len(procedures),
        "n_medications": len(drugs),
        "avg_visits_per_patient": len(samples) / len(patients),
        "avg_medications_per_visit": sum(med_counts) / len(med_counts),
        "n_first_visits": first_visits,
        "first_visit_ratio": first_visits / len(samples),
        "min_visits_per_patient": min(visit_counts.values()),
        "max_visits_per_patient": max(visit_counts.values()),
    }


def subset_statistics(dataset: PreprocessedDataset, subset: Subset) -> Dict[str, float]:
    samples = [dataset.samples[i] for i in subset.indices]
    patients = {sample["patient_id"] for sample in samples}
    med_counts = [len(set(sample["drugs"])) for sample in samples]
    first_visits = sum(1 for sample in samples if is_first_visit(sample))
    return {
        "n_samples": len(samples),
        "n_patients": len(patients),
        "avg_medications_per_visit": sum(med_counts) / len(med_counts),
        "n_first_visits": first_visits,
        "first_visit_ratio": first_visits / len(samples),
    }


def all_first_batch_offsets(
    dataset: PreprocessedDataset, indices: Iterable[int], batch_size: int
) -> List[int]:
    indices = list(indices)
    offsets = []
    for offset in range(0, len(indices), batch_size):
        chunk = indices[offset : offset + batch_size]
        if chunk and all(is_first_visit(dataset.samples[i]) for i in chunk):
            offsets.append(offset)
    return offsets


def mix_first_followup_indices(dataset: PreprocessedDataset, indices: Iterable[int]) -> List[int]:
    first = [i for i in indices if is_first_visit(dataset.samples[i])]
    follow = [i for i in indices if not is_first_visit(dataset.samples[i])]

    mixed = []
    while first or follow:
        if follow:
            mixed.append(follow.pop(0))
        if first:
            mixed.append(first.pop(0))
    return mixed


def stabilize_eval_subset(
    name: str, dataset: PreprocessedDataset, subset: Subset, batch_size: int
) -> Tuple[Subset, Dict]:
    before_offsets = all_first_batch_offsets(dataset, subset.indices, batch_size)
    if not before_offsets:
        return subset, {
            "name": name,
            "reordered": False,
            "all_first_batch_offsets_before": [],
            "all_first_batch_offsets_after": [],
        }

    mixed_indices = mix_first_followup_indices(dataset, subset.indices)
    after_offsets = all_first_batch_offsets(dataset, mixed_indices, batch_size)
    if after_offsets:
        raise RuntimeError(
            f"{name} still has all-first-visit batches after reordering: {after_offsets[:10]}"
        )

    return Subset(dataset, mixed_indices), {
        "name": name,
        "reordered": True,
        "all_first_batch_offsets_before": before_offsets,
        "all_first_batch_offsets_after": after_offsets,
    }


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAMRec MIMIC-IV training")

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=4006862222)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--run_suffix", type=str, default="clean20")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--gru_hidden_size", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"D:\Code\LAMRec-RAGBK\mimic-iv-2.2\CIDGMed",
    )

    args = parser.parse_args()
    seed = args.seed
    set_seed(seed)

    if args.gru_hidden_size is None:
        args.gru_hidden_size = args.embedding_dim

    records_file_path = os.path.join(args.dataset_path, "records_final.pkl")
    voc_file_path = os.path.join(args.dataset_path, "voc_final.pkl")
    if not os.path.exists(records_file_path) or not os.path.exists(voc_file_path):
        raise FileNotFoundError(
            "records_final.pkl and voc_final.pkl must exist in dataset_path: "
            f"{args.dataset_path}"
        )

    print(f"Using MIMIC-IV preprocessed records: {records_file_path}")
    print(f"Using MIMIC-IV preprocessed vocab: {voc_file_path}")
    print(f"Using seed: {seed}")
    print(f"Using DDI loss alpha: {args.alpha}")

    sample_dataset = PreprocessedDataset(
        records_path=records_file_path,
        voc_path=voc_file_path,
    )
    sample_dataset.stat()

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    project_root = r"D:\Code\LAMRec-RAGBK"
    run_name = f"mimic4_{seed}_alpha{str(args.alpha).replace('.', 'p')}_epochs{args.epochs}"
    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"
    run_dir = os.path.join(project_root, "output", run_name)
    os.makedirs(run_dir, exist_ok=True)

    val_dataset, val_batch_info = stabilize_eval_subset(
        "val", sample_dataset, val_dataset, args.batch_size
    )
    test_dataset, test_batch_info = stabilize_eval_subset(
        "test", sample_dataset, test_dataset, args.batch_size
    )

    stats_payload = {
        "args": vars(args),
        "run_name": run_name,
        "dataset": dataset_statistics(sample_dataset),
        "splits": {
            "train": subset_statistics(sample_dataset, train_dataset),
            "val": subset_statistics(sample_dataset, val_dataset),
            "test": subset_statistics(sample_dataset, test_dataset),
        },
        "batch_checks": {
            "val": val_batch_info,
            "test": test_batch_info,
        },
        "paper_reference": {
            "n_patients": 60125,
            "n_visits": 156810,
            "n_diagnoses": 2000,
            "n_procedures": 1500,
            "n_medications": 131,
            "avg_visits_per_patient": 2.61,
            "avg_medications_per_visit": 6.66,
        },
    }
    write_json(os.path.join(run_dir, "mimic4_dataset_stats.json"), stats_payload)

    print("Dataset statistics saved to:", os.path.join(run_dir, "mimic4_dataset_stats.json"))
    print("MIMIC-IV stats:", stats_payload["dataset"])
    print("Split stats:", stats_payload["splits"])
    print("Eval batch checks:", stats_payload["batch_checks"])

    train_dataloader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = get_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_dataloader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

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

    prediction_file = os.path.join(run_dir, "predictions_log.csv")
    trainer = Trainer(
        model=model,
        metrics=[
            "jaccard_samples",
            "pr_auc_samples",
            "f1_samples",
            "ddi_score",
            "roc_auc_samples",
            "avg_med",
        ],
        device=args.device,
        seed=run_name,
        output_file=prediction_file,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        monitor="jaccard_samples",
        lr=args.lr,
        weight_decay=1e-4,
    )
