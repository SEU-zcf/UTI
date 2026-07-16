import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from uti_mpc.evaluate import evaluate
from uti_mpc.train import train


def _write_manifest(root: Path):
    shard = root / "shards" / "synthetic"
    shard.mkdir(parents=True)
    samples_per_class = 12
    labels = np.repeat(np.array([1, 2, 3]), samples_per_class)
    count = len(labels)
    arrays = {
        "byte_tokens": np.random.default_rng(1).integers(0, 256, (count, 4, 32), dtype=np.uint8),
        "byte_mask": np.ones((count, 4), dtype=np.bool_),
        "length_direction": np.random.default_rng(2).normal(size=(count, 4)).astype(np.float32),
        "length_mask": np.ones((count, 4), dtype=np.bool_),
        "labels": labels,
    }
    paths = {}
    for name, array in arrays.items():
        path = shard / f"{name}.npy"
        np.save(path, array)
        paths[name] = path.relative_to(root).as_posix()
    manifest = {"shards": [{"id": "synthetic", "count": count, **paths}]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_one_epoch_train_resume_calibrate_and_evaluate(tmp_path: Path):
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    _write_manifest(cache)
    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"""
data:
  cache_dir: {cache}
  flow_length_bucket_edges: [1, 2, 8]
model:
  hierarchical_bgi: true
  cross_modal_fusion: true
  byte_embedding_dim: 8
  branch_channels: 4
  byte_dim: 16
  time_dim: 16
  embedding_dim: 8
  se_reduction: 4
  attention_heads: 4
  windows: [2, 4]
  ffn_expansion: 2
  dropout: 0.0
  max_length: 8
  byte_attention_heads: 4
  byte_packet_layers: 1
  cross_modal_dim: 16
  cross_attention_heads: 4
  max_packets: 8
split:
  known_classes: [1, 2]
  unknown_classes: [3]
  test_fraction: 0.2
  validation_fraction_of_development: 0.2
loss:
  triplet_margin: 0.2
  prototype_margin: 1.0
  lambda_intra: 0.5
  lambda_inter: 0.3
  lambda_arcface: 0.1
  arcface_scale: 16.0
  arcface_margin: 0.2
  subcenters_per_class: 3
  lambda_diversity: 0.1
  subcenter_diversity_margin: 0.2
  loss_weighting: ema
train:
  output_dir: {output}
  seed: 42
  device: cpu
  amp: none
  deterministic: true
  compile: false
  epochs: 1
  stage1_epochs: 1
  classes_per_batch: 2
  samples_per_class: 2
  batches_per_epoch: 1
  learning_rate: 0.001
  weight_decay: 0.0
  warmup_epochs: 0
  gradient_clip: 1.0
  evaluate_every: 1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
evaluation:
  batch_size: 8
  threshold_quantile: 0.95
  threshold_source: validation
  minimum_threshold_samples: 1
  use_train_threshold_floor: true
  length_conditioned_prototypes: true
""",
        encoding="utf-8",
    )
    best = train(config)
    assert best.exists()
    assert (output / "last.pt").exists()
    config.write_text(
        config.read_text(encoding="utf-8").replace("epochs: 1", "epochs: 2", 1),
        encoding="utf-8",
    )
    resumed = train(config, output / "last.pt")
    assert resumed.exists()
    metrics = evaluate(config, best)
    assert {
        "PR",
        "KCA",
        "UDR",
        "closed_set_KCA",
        "known_rejection_rate",
        "accepted_known_accuracy",
    }.issubset(metrics)
    assert metrics["threshold_source"] == "validation"
    assert metrics["length_conditioned_prototypes"] is True
    assert (output / "evaluation" / "flow_length_metrics.csv").exists()
    assert (output / "evaluation" / "decision_mode_comparison.csv").exists()

    config.write_text(
        config.read_text(encoding="utf-8")
        .replace("length_conditioned_prototypes: true", "length_conditioned_prototypes: false")
        .replace(
            "use_train_threshold_floor: true",
            "use_train_threshold_floor: true\n  compare_auxiliary_rejection: true\n"
            "  auxiliary_threshold_quantile: 0.95\n  knn_neighbors: 1\n"
            "  minimum_auxiliary_threshold_samples: 1\n  knn_chunk_size: 8",
        ),
        encoding="utf-8",
    )
    auxiliary_metrics = evaluate(config, best)
    assert set(auxiliary_metrics["decision_mode_comparison"]) == {
        "prototype_only",
        "prototype_knn",
        "prototype_knn_margin",
    }
    assert (output / "evaluation" / "auxiliary_predictions.csv").exists()

    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "knn_chunk_size: 8", "knn_chunk_size: 8\n  subprototypes_per_class: 2"
        ),
        encoding="utf-8",
    )
    subprototype_metrics = evaluate(config, best)
    assert subprototype_metrics["subprototypes_per_class"] == 2
    assert all(
        set(class_thresholds) == {"subprototype_1", "subprototype_2"}
        for class_thresholds in subprototype_metrics["thresholds"].values()
    )


def _write_v3_manifest(root: Path):
    rng = np.random.default_rng(7)
    shards = []
    shard_index = 0
    for label, captures in ((1, 3), (2, 3), (3, 2)):
        for capture_index in range(captures):
            shard = root / "shards" / f"capture_{shard_index}"
            shard.mkdir(parents=True)
            count = 4
            packet_mask = np.ones((count, 3), dtype=np.bool_)
            arrays = {
                "payload_tokens": rng.integers(0, 256, (count, 3, 8), dtype=np.uint8),
                "payload_mask": np.ones((count, 3, 8), dtype=np.bool_),
                "packet_features": rng.random((count, 3, 16), dtype=np.float32),
                "packet_mask": packet_mask,
                "burst_features": rng.random((count, 2, 8), dtype=np.float32),
                "burst_mask": np.ones((count, 2), dtype=np.bool_),
                "labels": np.full(count, label, dtype=np.int64),
            }
            paths = {}
            for name, array in arrays.items():
                path = shard / f"{name}.npy"
                np.save(path, array)
                paths[name] = path.relative_to(root).as_posix()
            shards.append(
                {
                    "id": f"capture_{shard_index}",
                    "capture": f"class_{label}/capture_{capture_index}.pcap",
                    "label": label,
                    "count": count,
                    **paths,
                }
            )
            shard_index += 1
    manifest = {
        "version": 2,
        "representation": "uti_mpc_v3",
        "shards": shards,
        "total_samples": sum(int(shard["count"]) for shard in shards),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_v3_one_epoch_grouped_train_and_evaluate(tmp_path: Path):
    cache = tmp_path / "v3_cache"
    output = tmp_path / "v3_output"
    cache.mkdir()
    _write_v3_manifest(cache)
    config = tmp_path / "v3_smoke.yaml"
    config.write_text(
        f"""
data:
  cache_dir: {cache}
  flow_length_bucket_edges: [1, 2, 8]
model:
  name: uti_mpc_v3
  payload_bytes: 8
  max_packets: 3
  max_bursts: 2
  byte_embedding_dim: 4
  packet_dim: 8
  packet_heads: 2
  packet_layers: 1
  burst_dim: 8
  burst_heads: 2
  burst_layers: 1
  embedding_dim: 4
  subprototypes_per_class: 2
  dropout: 0.0
split:
  known_classes: [1, 2]
  unknown_classes: [3]
  group_by_capture: true
  test_fraction: 0.2
  validation_fraction_of_development: 0.1
loss:
  lambda_contrastive: 1.0
  lambda_reconstruction: 0.1
  lambda_prototype: 1.0
  lambda_compact: 0.5
  lambda_separation: 0.1
  lambda_overlap: 0.1
  lambda_radius: 0.01
  lambda_diversity: 0.01
  lambda_pseudo_unknown: 0.1
train:
  output_dir: {output}
  seed: 42
  device: cpu
  amp: none
  deterministic: true
  compile: false
  epochs: 1
  stage1_epochs: 0
  classes_per_batch: 2
  samples_per_class: 2
  batches_per_epoch: 1
  learning_rate: 0.001
  weight_decay: 0.0
  warmup_epochs: 0
  gradient_clip: 1.0
  evaluate_every: 1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  augmentation:
    packet_drop: 0.0
    payload_mask_fraction: 0.0
    stats_mask_fraction: 0.5
    iat_jitter: 0.0
evaluation:
  batch_size: 8
  coverage: 0.95
  minimum_subprototype_samples: 1
  minimum_class_samples: 1
""",
        encoding="utf-8",
    )
    checkpoint = train(config)
    metrics = evaluate(config, checkpoint)
    assert checkpoint.exists()
    assert {"AUROC", "OSCR", "known_macro_F1", "open_macro_F1"}.issubset(metrics)
    assert (output / "evaluation" / "open_set_artifacts.pt").exists()
    assert (output / "evaluation" / "capture_metrics.csv").exists()
