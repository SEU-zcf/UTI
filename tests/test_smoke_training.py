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
