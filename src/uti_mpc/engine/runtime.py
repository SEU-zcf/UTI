from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from uti_mpc.data.buckets import packet_counts_to_buckets
from uti_mpc.data.dataset import TrafficDataset
from uti_mpc.data.sampler import PKBatchSampler
from uti_mpc.data.splits import (
    OpenSetSplit,
    build_grouped_open_set_split,
    build_open_set_split,
)


def load_dataset_and_split(
    config: dict[str, Any], output_dir: Path
) -> tuple[TrafficDataset, OpenSetSplit, Path]:
    manifest = Path(config["data"]["cache_dir"]) / "manifest.json"
    dataset = TrafficDataset(manifest)
    split_path = output_dir / "splits.npz"
    split_config = config["split"]
    expected_known = tuple(sorted(int(value) for value in split_config["known_classes"]))
    expected_unknown = tuple(sorted(int(value) for value in split_config["unknown_classes"]))
    if split_path.exists():
        split = OpenSetSplit.load(split_path)
        if split.known_classes != expected_known or split.unknown_classes != expected_unknown:
            raise ValueError(f"Existing split does not match configuration: {split_path}")
        if split.cache_fingerprint and split.cache_fingerprint != dataset.fingerprint:
            raise ValueError(f"Existing split was created for a different cache: {split_path}")
        split.validate_groups()
    else:
        split_builder = (
            build_grouped_open_set_split
            if bool(split_config.get("group_by_capture", False))
            else build_open_set_split
        )
        split = split_builder(
            dataset.labels(),
            *((dataset.captures(),) if split_builder is build_grouped_open_set_split else ()),
            expected_known,
            expected_unknown,
            seed=int(config["train"]["seed"]),
            test_fraction=float(split_config.get("test_fraction", 0.2)),
            validation_fraction_of_development=float(
                split_config.get("validation_fraction_of_development", 0.1)
            ),
            **(
                {"cache_fingerprint": dataset.fingerprint}
                if split_builder is build_grouped_open_set_split
                else {}
            ),
        )
        split.save(split_path)
    return dataset, split, split_path


def _loader_options(config: dict[str, Any], workers: int, batch_size: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": bool(config["train"].get("pin_memory", True)),
        "persistent_workers": bool(config["train"].get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = int(config["train"].get("prefetch_factor", 2))
    return options


def build_loaders(
    dataset: TrafficDataset, split: OpenSetSplit, config: dict[str, Any]
) -> dict[str, DataLoader]:
    workers = int(config["train"].get("num_workers", 8))
    p = int(config["train"]["classes_per_batch"])
    q = int(config["train"]["samples_per_class"])
    train_subset = Subset(dataset, split.train.tolist())
    train_labels = [dataset.get_label(int(index)) for index in split.train]
    train_packet_counts = torch.tensor(
        [dataset.get_packet_count(int(index)) for index in split.train], dtype=torch.long
    )
    bucket_edges = config["data"].get("flow_length_bucket_edges", [1, 2, 8])
    train_buckets = packet_counts_to_buckets(train_packet_counts, bucket_edges).tolist()
    sampler = PKBatchSampler(
        train_labels,
        p,
        q,
        seed=int(config["train"]["seed"]),
        batches_per_epoch=config["train"].get("batches_per_epoch"),
        buckets=train_buckets,
        stratify_by_bucket=bool(config["train"].get("stratify_by_flow_length", False)),
    )
    train_options = _loader_options(config, workers, p * q)
    train_options.pop("batch_size")
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(train_subset, batch_sampler=sampler, **train_options),
    }
    evaluation_batch = int(config["evaluation"].get("batch_size", 512))
    evaluation_options = _loader_options(config, workers, evaluation_batch)
    loaders["train_eval"] = DataLoader(train_subset, shuffle=False, **evaluation_options)
    loaders["validation"] = DataLoader(
        Subset(dataset, split.validation.tolist()), shuffle=False, **evaluation_options
    )
    loaders["test"] = DataLoader(
        Subset(dataset, split.test.tolist()), shuffle=False, **evaluation_options
    )
    return loaders
