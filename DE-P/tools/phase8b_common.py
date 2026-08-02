"""Shared immutable-sample and metric helpers for bounded Phase-8B tools."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss.dynamic_types import DynamicLossConfig, DynamicObjectiveConfig, RiskMetricsConfig
from policy.dep_trainer import DepTrainer
from policy.dynamic_collate import dynamic_sequence_collate
from policy.dynamic_sequence_dataset import DynamicSequenceDataset
from policy.dynamic_training_config import DynamicTrainingConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    return YAML(typ="safe").load(path)


def make_trainer(config, *, seed=None, objective=None, loss_weight=None,
                 attention=True, freeze_policy=None, data_root=None, map_catalog=None):
    seed = int(config["random_seed"] if seed is None else seed)
    seed_everything(seed)
    training = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
    loss = DynamicLossConfig.from_mapping(config["dynamic_loss"])
    risk = RiskMetricsConfig.from_mapping(config["risk_metrics"])
    objective = objective or DynamicObjectiveConfig.from_mapping(config["dynamic_objective"])
    trainer = DepTrainer(
        learning_rate=float(config["learning_rate"]), batch_size=int(config["batch_size"]),
        loss_weight=loss_weight or config.get("loss_weight", [1.0, 1.0]),
        tensorboard_path=str(Path("/tmp") / f"dep-phase8b-{seed}"),
        checkpoint_path=config["initialization_checkpoint"], backbone_variant="corrected",
        dataset_mode="dynamic", dynamic_data_root=data_root or config["dataset_root"],
        freeze_policy=freeze_policy or config.get("freeze_policy", "late"), num_workers=0,
        random_seed=seed, training_config_override=training,
        dynamic_loss_config_override=loss, dynamic_objective_config_override=objective,
        risk_metrics_config_override=risk,
        dynamic_map_catalog_override=(
            map_catalog or config.get("dynamic_map_catalog")
        ),
    )
    if not attention:
        trainer.policy.dynamic_config = trainer.policy.dynamic_config.__class__(**{
            **trainer.policy.dynamic_config.__dict__, "use_attention": False,
        })
    return trainer


def dataset_index(dataset, sequence_id, frame_index):
    for index, (sequence, current, _category) in enumerate(dataset._windows):
        _directory, _metadata, frames = dataset._sequences[sequence]
        if sequence == sequence_id and int(frames[current]["frame_index"]) == int(frame_index):
            return index
    raise KeyError(f"window not found: {sequence_id}/{frame_index}")


def load_fixed_samples(config, fixed_path, *, training_only=False):
    payload = json.loads(Path(fixed_path).read_text(encoding="utf-8"))
    cache, samples = {}, []
    for entry in payload["windows"]:
        if training_only and entry["split"] != "train":
            continue
        root = str(Path(entry["dataset_root"]).resolve())
        key = (root, entry["split"])
        if key not in cache:
            training = DynamicTrainingConfig.from_mapping(config["dynamic_training"])
            cache[key] = DynamicSequenceDataset(root, entry["split"], training_config=training)
        dataset = cache[key]
        sample = dataset[dataset_index(dataset, entry["sequence_id"], entry["frame_index"])]
        sample["fixed_entry"] = entry
        samples.append(sample)
    return samples


def collate(samples):
    return dynamic_sequence_collate(samples)


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _finite_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values < np.finfo(np.float32).max / 2)]
    return float(values.min()) if values.size else None


def evaluate_batch(trainer, batch):
    trainer.policy.eval()
    with torch.inference_mode():
        details = trainer.compute_batch("dynamic", batch)
    raw = details["candidate_dynamic_cost_raw"].reshape(-1, 15).detach().cpu().numpy()
    score = details["predicted_score"].reshape(-1, 15).detach().cpu().numpy()
    diagnostics = details["dynamic_diagnostics"]
    clearance = diagnostics.candidate_min_clearance.detach().cpu().numpy()
    safe = diagnostics.candidate_safe_mask.detach().cpu().numpy()
    fraction = trainer.risk_metrics_config.cvar_fraction
    top_count = max(1, int(np.ceil(raw.shape[1] * fraction)))
    cvar = np.sort(raw, axis=1)[:, -top_count:].mean(axis=1)
    selected = score.argmin(axis=1)
    rows = np.arange(raw.shape[0])
    correlations = []
    kendalls = []
    for predicted, risk in zip(score, raw):
        correlations.append(float(np.nan_to_num(spearmanr(predicted, risk).statistic)))
        kendalls.append(float(np.nan_to_num(kendalltau(predicted, risk).statistic)))
    expected = (torch.softmax(-details["predicted_score"].reshape(-1, 15), dim=1)
                * details["candidate_dynamic_cost_raw"].reshape(-1, 15)).sum(1)
    return {
        "raw_dynamic_mean": float(raw.mean()),
        "weighted_dynamic_mean": scalar(details["dynamic_safety_loss"]),
        "dynamic_weight": trainer.dynamic_loss_config.weight,
        "dynamic_cvar": float(cvar.mean()),
        "dynamic_median": float(np.median(raw)),
        "dynamic_p75": float(np.quantile(raw, 0.75)),
        "dynamic_p90": float(np.quantile(raw, 0.90)),
        "safe_candidate_fraction": float(safe.mean()),
        "safe_candidate_count_mean": float(safe.sum(axis=1).mean()),
        "high_risk_candidate_count_mean": float(
            diagnostics.candidate_high_risk_mask.sum(1).float().mean().cpu()
        ),
        "violation_candidate_fraction": float((clearance < 0).mean()),
        "minimum_clearance": _finite_min(clearance),
        "minimum_distance": _finite_min(diagnostics.candidate_min_distance.cpu().numpy()),
        "top1_raw_risk": float(raw[rows, selected].mean()),
        "top1_collision_fraction": float((clearance[rows, selected] < 0).mean()),
        "top1_min_clearance": _finite_min(clearance[rows, selected]),
        "oracle_regret": float((raw[rows, selected] - raw.min(axis=1)).mean()),
        "score_weighted_expected_risk": float(expected.mean().detach().cpu()),
        "spearman": float(np.mean(correlations)),
        "kendall": float(np.mean(kendalls)),
        "static": scalar(details["static_safety_loss"]),
        "guidance": scalar(details["guidance_loss"]),
        "score_loss": scalar(details["score_loss"]),
        "trajectory_loss": scalar(details["trajectory_loss"]),
        "raw": raw,
        "score": score,
        "clearance": clearance,
        "details": details,
    }


def jsonable(metrics):
    return {key: value for key, value in metrics.items()
            if key not in {"raw", "score", "clearance", "details"}}
