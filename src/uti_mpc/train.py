from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from uti_mpc.config import load_config, require_keys, save_config
from uti_mpc.engine.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from uti_mpc.engine.features import extract_embeddings
from uti_mpc.engine.runtime import build_loaders, load_dataset_and_split
from uti_mpc.losses import ProtoMarginLoss
from uti_mpc.metrics.open_set import compute_prototypes, squared_distances
from uti_mpc.models import UTIMPC
from uti_mpc.utils import (
    append_jsonl,
    choose_amp_dtype,
    cosine_warmup_factor,
    seed_everything,
    select_single_device,
)


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _checkpoint_payload(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "stage": "warmup" if epoch < int(config["train"]["stage1_epochs"]) else "formal",
        "best_validation": best_validation,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "known_classes": list(config["split"]["known_classes"]),
        "unknown_classes": list(config["split"]["unknown_classes"]),
    }


@torch.no_grad()
def _known_validation_accuracy(
    model: torch.nn.Module,
    loaders: dict,
    known_classes: list[int],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    train_features, train_labels, _, _ = extract_embeddings(
        model, loaders["train_eval"], device, amp_dtype
    )
    validation_features, validation_labels, _, _ = extract_embeddings(
        model, loaders["validation"], device, amp_dtype
    )
    prototypes, classes = compute_prototypes(train_features, train_labels, known_classes)
    predicted = classes[squared_distances(validation_features, prototypes).argmin(dim=1)]
    return float((predicted == validation_labels).float().mean())


def train(config_path: str | Path, resume: str | Path | None = None) -> Path:
    config = load_config(config_path)
    require_keys(
        config,
        "data.cache_dir",
        "model",
        "split.known_classes",
        "split.unknown_classes",
        "train.output_dir",
        "train.epochs",
    )
    output_dir = Path(config["train"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    seed = int(config["train"].get("seed", 42))
    seed_everything(seed, bool(config["train"].get("deterministic", True)))
    device = select_single_device(str(config["train"].get("device", "cuda:0")))
    amp_dtype = choose_amp_dtype(device, str(config["train"].get("amp", "bf16")))
    print(
        f"device={device}; amp={amp_dtype}; single-device mode; "
        f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
    )
    dataset, split, split_path = load_dataset_and_split(config, output_dir)
    loaders = build_loaders(dataset, split, config)
    raw_model = UTIMPC(config["model"]).to(device)
    model: torch.nn.Module = raw_model
    if bool(config["train"].get("compile", False)):
        model = torch.compile(raw_model)
    criterion = ProtoMarginLoss(
        triplet_margin=float(config["loss"]["triplet_margin"]),
        prototype_margin=float(config["loss"]["prototype_margin"]),
        lambda_intra=float(config["loss"]["lambda_intra"]),
        lambda_inter=float(config["loss"]["lambda_inter"]),
        known_classes=list(config["split"]["known_classes"]),
        embedding_dim=int(config["model"].get("embedding_dim", 128)),
        lambda_arcface=float(config["loss"].get("lambda_arcface", 0.0)),
        arcface_scale=float(config["loss"].get("arcface_scale", 30.0)),
        arcface_margin=float(config["loss"].get("arcface_margin", 0.2)),
    ).to(device)
    trainable_parameters = [*raw_model.parameters(), *criterion.parameters()]
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_warmup_factor(
            epoch, int(config["train"]["warmup_epochs"]), epochs
        ),
    )
    use_fp16_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    start_epoch = 0
    best_validation = float("-inf")
    if resume:
        checkpoint = load_checkpoint(resume, device)
        raw_model.load_state_dict(checkpoint["model"])
        if "criterion" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion"])
        elif criterion.arcface is not None:
            raise ValueError("ArcFace checkpoint is missing the criterion state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        restore_rng_state(checkpoint["rng_state"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_validation", best_validation))
    writer = SummaryWriter(output_dir / "tensorboard")
    metrics_path = output_dir / "training.jsonl"
    eval_every = int(config["train"].get("evaluate_every", 5))
    stage1_epochs = int(config["train"]["stage1_epochs"])
    gradient_clip = float(config["train"].get("gradient_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        sampler = loaders["train"].batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        stage = "warmup" if epoch < stage1_epochs else "formal"
        totals = {
            "total": 0.0,
            "triplet": 0.0,
            "intra": 0.0,
            "inter": 0.0,
            "arcface": 0.0,
        }
        batches = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["label"].to(device, non_blocking=True)
            with _autocast(device, amp_dtype):
                embeddings = model(
                    batch["byte_tokens"].to(device, non_blocking=True),
                    batch["length_direction"].to(device, non_blocking=True),
                    batch["byte_mask"].to(device, non_blocking=True),
                    batch["length_mask"].to(device, non_blocking=True),
                )
                losses = criterion(embeddings, labels, stage)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batches}")
            if use_fp16_scaler:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_parameters, gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                clip_grad_norm_(trainable_parameters, gradient_clip)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
            batches += 1
        if batches == 0:
            raise RuntimeError("Training loader produced no batches")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "stage": stage,
            "learning_rate": learning_rate,
            "seconds": time.time() - epoch_start,
            "batches": batches,
            "batch_size": int(config["train"]["classes_per_batch"])
            * int(config["train"]["samples_per_class"]),
            **{key: value / batches for key, value in totals.items()},
        }
        record["samples_per_second"] = record["batches"] * record["batch_size"] / record["seconds"]
        if device.type == "cuda":
            record["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        should_validate = (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        if should_validate:
            validation_accuracy = _known_validation_accuracy(
                model,
                loaders,
                list(config["split"]["known_classes"]),
                device,
                amp_dtype,
            )
            record["known_validation_accuracy"] = validation_accuracy
            if validation_accuracy > best_validation:
                best_validation = validation_accuracy
                payload = _checkpoint_payload(
                    raw_model,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_validation,
                    config,
                )
                save_checkpoint(output_dir / "best.pt", payload, device)
        payload = _checkpoint_payload(
            raw_model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_validation,
            config,
        )
        save_checkpoint(output_dir / "last.pt", payload, device)
        append_jsonl(metrics_path, record)
        for key, value in record.items():
            if isinstance(value, (int, float)) and key not in {"epoch"}:
                writer.add_scalar(f"train/{key}", value, epoch)
        writer.flush()
        print(record)
    writer.close()
    print(f"split={split_path}; best_validation={best_validation:.6f}")
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UTI-MPC on one logical CUDA device")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
