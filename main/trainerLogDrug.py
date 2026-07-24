import logging
import os
import time
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd  # Make sure pandas is imported
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn, regression_metrics_fn)
from pyhealth.utils import create_directory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return


def get_metrics_fn(mode: str) -> Callable:
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    elif mode == "multilabel":
        return multilabel_metrics_fn
    elif mode == "regression":
        return regression_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


class Trainer:
    """
    Main trainer class.

    Args:
        model: A PyTorch model.
        checkpoint_path: Path to a checkpoint to load.
        metrics: A list of metrics to compute.
        device: The device to use.
        enable_logging: Whether to enable logging.
        output_file: Path to a CSV file to save predictions for each epoch.
        seed: The random seed.
    """

    def __init__(
            self,
            model: nn.Module,
            checkpoint_path: Optional[str] = None,
            metrics: Optional[List[str]] = None,
            device: Optional[str] = None,
            enable_logging: bool = True,
            output_file: Optional[str] = None,  # <-- NEW: Add output_file argument
            seed: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.metrics = metrics
        self.device = device
        self.enable_logging = enable_logging
        self.output_file = output_file  # <-- NEW: Store output_file path
        self.seed = seed

        # set logger
        if enable_logging:
            output_path = os.path.join(os.getcwd(), "output")
            self.exp_path = os.path.join(output_path, str(seed))
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        # NEW: Initialize the output file with headers
        if self.output_file is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            # Write header
            pd.DataFrame(columns=["epoch", "patient_index", "y_true", "y_pred"]).to_csv(
                self.output_file, index=False
            )
            logger.info(f"Predictions will be saved to {self.output_file}")

        # set device
        self.model.to(self.device)

        # logging
        logger.info(self.model)
        logger.info(f"Metrics: {self.metrics}")
        logger.info(f"Device: {self.device}")

        # load checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.load_ckpt(checkpoint_path)

        logger.info("")
        return

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            epochs: int = 5,
            optimizer_class: Type[Optimizer] = torch.optim.Adam,
            optimizer_params: Optional[Dict[str, object]] = None,
            steps_per_epoch: int = None,
            weight_decay: float = 0.0,
            max_grad_norm: float = None,
            monitor: Optional[str] = None,
            monitor_criterion: str = "max",
            load_best_model_at_last: bool = True,
            lr: float = 1e-3,
    ):
        if optimizer_params is None:
            optimizer_params = {"lr": lr}

        # logging omitted for brevity...

        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0

        for epoch in range(epochs):
            # training loop omitted for brevity...
            training_loss = []
            self.model.zero_grad()
            self.model.train()
            logger.info("")
            for _ in trange(
                    steps_per_epoch,
                    desc=f"Training Epoch {epoch} / {epochs}",
                    smoothing=0.05,
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)
                data = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }
                output = self.model(**data)
                loss = output["loss"]
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1

            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")
            if self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))

            # validation
            if val_dataloader is not None:
                # <-- MODIFIED: pass epoch number to evaluate
                scores = self.evaluate(val_dataloader, epoch=epoch)
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(
                            f"New best {monitor} score ({score:.4f}) "
                            f"at epoch-{epoch}, step-{global_step}"
                        )
                        best_score = score
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
                os.path.join(self.exp_path, "best.ckpt")
        ):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

        # test
        if test_dataloader is not None:
            # <-- MODIFIED: pass "test" as epoch label
            scores = self.evaluate(test_dataloader, epoch="test")
            logger.info(f"--- Test ---")
            for key in scores.keys():
                logger.info("{}: {:.4f}".format(key, scores[key]))
        return

    def evaluate(self, dataloader, epoch: Optional[int or str] = None) -> Dict[str, float]:
        """
        Evaluates the model on a dataset.

        Args:
            dataloader: A PyTorch DataLoader.
            epoch: The current epoch number (or a string like 'test').
                   Used for logging predictions.
        """
        if self.model.mode is not None:
            all_outputs = self.inference(dataloader)
            y_true_all = all_outputs["y_true_all"]
            y_prob_all = all_outputs["y_prob_all"]
            loss_mean = all_outputs["loss_mean"]

            mode = self.model.mode
            metrics_fn = get_metrics_fn(mode)
            scores = metrics_fn(y_true_all, y_prob_all, metrics=self.metrics)
            scores["loss"] = loss_mean

            # NEW: Save predictions if epoch and output file are specified
            if epoch is not None and self.output_file is not None:
                self._save_predictions(
                    epoch=epoch,
                    trues=all_outputs["y_true_indices_all"],
                    preds=all_outputs["y_pred_indices_all"],
                )
        else:
            # This part remains unchanged
            loss_all = []
            for data in tqdm(dataloader, desc="Evaluation"):
                self.model.eval()
                data = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }
                with torch.no_grad():
                    output = self.model(**data)
                    loss = output["loss"]
                    loss_all.append(loss.item())
            loss_mean = sum(loss_all) / len(loss_all)
            scores = {"loss": loss_mean}
        return scores

    def inference(self, dataloader) -> Dict:
        """
        Runs inference on a dataset.

        Returns:
            A dictionary containing all model outputs.
        """
        self.model.eval()
        loss_all = []
        y_true_all, y_prob_all = [], []

        # NEW: Lists to store predicted and true drug indices
        y_pred_indices_all, y_true_indices_all = [], []

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Inference"):
                data = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                }
                output = self.model(**data)

                loss_all.append(output["loss"].item())
                y_true_all.append(output["y_true"].cpu().numpy())
                y_prob_all.append(output["y_prob"].cpu().numpy())

                # NEW: Collect the prediction and true indices from the model output
                y_pred_indices_all.extend(output["y_pred_indices"])
                y_true_indices_all.extend(output["y_true_indices"])

        # a dictionary of all collected lists
        results = {
            "loss_mean": sum(loss_all) / len(loss_all),
            "y_true_all": np.concatenate(y_true_all, axis=0),
            "y_prob_all": np.concatenate(y_prob_all, axis=0),
            "y_pred_indices_all": y_pred_indices_all,
            "y_true_indices_all": y_true_indices_all,
        }
        return results

    def _save_predictions(
            self,
            epoch: int or str,
            trues: List[List[int]],
            preds: List[List[int]],
    ):
        """
        Saves predictions for an epoch to the output CSV file.
        """
        # A simple placeholder for patient index within the file
        patient_indices = range(len(trues))

        df = pd.DataFrame(
            {
                "epoch": epoch,
                "patient_index": patient_indices,
                "y_true": trues,
                "y_pred": preds,
            }
        )

        # Append to the CSV file without writing the header again
        df.to_csv(self.output_file, mode="a", header=False, index=False)
        logger.info(f"Saved {len(df)} predictions for epoch '{epoch}' to {self.output_file}")

    # save_ckpt and load_ckpt methods remain unchanged...
    def save_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = self.model.state_dict()
        torch.save(state_dict, ckpt_path)
        return

    def load_ckpt(self, ckpt_path: str) -> None:
        """Saves the model checkpoint."""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return