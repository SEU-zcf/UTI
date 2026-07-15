# UTI-MPC

PyTorch reproduction of **UTI-MPC: A Multi-View Contrastive Learning Method for Unknown Encrypted Traffic Detection**.

The implementation preserves the paper's BGI-CNN byte branch, TWT length-direction branch, adaptive modality gate, unit-sphere embedding, two-stage ProtoMargin objective, and class-specific 95th-percentile rejection thresholds.

## Environment

The target environment is Python 3.12.3 with an existing `torch 2.11.0+cu130` installation. Install this project without replacing that CUDA build:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e . --no-deps
```

Verify the supplied PyTorch installation separately:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Single-H200 rule

This project deliberately does not implement DDP or DataParallel. Select exactly one physical H200 before starting Python:

```bash
CUDA_VISIBLE_DEVICES=3 python -m uti_mpc.train --config configs/iscxvpn2016_ur40.yaml
```

Inside the process, the selected card is always logical `cuda:0`. The code never enumerates or initializes the other seven cards. The default precision is BF16 and the fixed training batch is `P=4, Q=32` (128 flows).

## Expected raw data layout

Place ISCXVPN2016 PCAP or PCAPNG captures under one root directory. Labels are inferred conservatively from capture paths. For non-standard names, supply a CSV instead of relying on inference:

```csv
path,label
NonVPN/aim_chat_3a.pcap,Chat
VPN/vpn_ftps_A.pcap,VPN-FileTransfer
custom/capture.pcap,10
```

Both numeric IDs and the following names are accepted:

| ID | Class | ID | Class |
|---:|---|---:|---|
| 1 | Chat | 6 | VPN-Chat |
| 2 | Email | 7 | VPN-Email |
| 3 | FileTransfer | 8 | VPN-FileTransfer |
| 4 | Streaming | 9 | VPN-Streaming |
| 5 | VoIP | 10 | VPN-VoIP |

## Preprocessing

One preprocessing pass can be reused by all three unknown-ratio experiments:

```bash
python -m uti_mpc.preprocess \
  --config configs/iscxvpn2016_ur40.yaml \
  --pcap-root /path/to/ISCXVPN2016/PCAPs \
  --label-map /path/to/labels.csv
```

The cache defaults to `data/processed/iscxvpn2016`. It contains memory-mapped NumPy shards and `manifest.json`. Processing keeps IPv4 TCP/UDP flows only, aggregates bidirectional five-tuples with a 60-second idle timeout, and records one label per capture.

## Training

Run each paper split independently:

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train --config configs/iscxvpn2016_ur20.yaml
CUDA_VISIBLE_DEVICES=1 python -m uti_mpc.train --config configs/iscxvpn2016_ur40.yaml
CUDA_VISIBLE_DEVICES=2 python -m uti_mpc.train --config configs/iscxvpn2016_ur60.yaml
```

These are three independent single-GPU processes. A single training process always uses only one selected card.

### Enhanced single-card configurations

The `*_enhanced.yaml` configurations retain the paper's BGI-CNN, TWT, adaptive
modality gate, unit-sphere embedding, and ProtoMargin pipeline. They add one
BGI residual refinement block, a second TWT block with shifted local windows,
and a residual path around the gated fusion transform. Their P×Q samplers cover
every known class in each batch: UR20 uses `8×64=512`, UR40 uses `6×80=480`,
and UR60 uses `4×128=512`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur40_enhanced.yaml
```

These are new model structures and must start from scratch; do not resume a
baseline checkpoint into an enhanced configuration. The enhanced output paths
are separate, so baseline artifacts remain intact.

Resume without changing the configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20.yaml \
  --resume outputs/iscxvpn2016_ur20/last.pt
```

Each output directory contains the resolved configuration, deterministic split, TensorBoard logs, JSONL training log, `last.pt`, and `best.pt`.
Training JSONL records also include batch count, effective batch size,
samples/second, and peak allocated GPU memory in GiB.

## Calibration and evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.evaluate \
  --config configs/iscxvpn2016_ur20.yaml \
  --checkpoint outputs/iscxvpn2016_ur20/best.pt
```

The evaluation directory contains prototypes and thresholds, PR/KCA/UDR, confusion matrix, and per-flow predictions. Unknown traffic is represented by prediction `-1`.
It also contains `raw_class_confusion.csv`, which preserves the original labels
of unknown classes rather than merging them into one row, and
`class_distance_diagnostics.json`, with per-class acceptance, nearest-prototype,
distance, and threshold-ratio distributions.
`capture_prediction_breakdown.csv` further aggregates these diagnostics by source
PCAP capture, preserving the original class and final prediction.

### Flow-length conditional experiments

The `*_length_conditional.yaml` configurations retain every short flow. They
stratify each P×Q batch by four observed packet-count buckets (`1`, `2`, `3-8`,
and `>=9` packets), then calibrate a separate prototype and rejection threshold
for every known-class/bucket pair. Evaluation writes `flow_length_metrics.csv`
with PR/KCA/UDR for each bucket. These configurations use separate output paths
and must be trained from scratch because their sampler differs from the standard
enhanced experiment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20_length_conditional.yaml
```

## CPU smoke tests

CPU mode is intended for verification, not full training. Tests override the CUDA/BF16 settings with a small synthetic configuration:

```bash
pytest
```

The tests cover packet feature construction, bidirectional flow aggregation, deterministic open-set splits, model shapes and gradients, ProtoMargin, adaptive thresholds, metrics, checkpointing, and a one-epoch end-to-end run.
