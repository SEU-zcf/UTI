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

### Protocol-sanitized preprocessing

The `*_sanitized.yaml` configurations create a separate
`data/processed/iscxvpn2016_sanitized` cache. They remove only explicitly
configured infrastructure traffic (DNS, DHCP, NetBIOS, SSDP, WS-Discovery,
mDNS, LLMNR, IPv4 multicast, and limited broadcast) before capture-level labels
are assigned to model samples. Ports are used only for this audit-driven cleanup
and are not model inputs. The original cache remains unchanged.

```bash
python -m uti_mpc.preprocess \
  --config configs/iscxvpn2016_ur20_sanitized.yaml \
  --pcap-root /path/to/ISCX-VPN-NonVPN-2016
```

The sanitized cache contains `sanitization_audit.json` with kept/dropped flow,
packet, and byte counts for every capture and exclusion reason. One preprocessing
pass is shared by UR20, UR40, and UR60 sanitized experiments.

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

The sanitized configurations use held-out known validation embeddings for
class-specific threshold calibration while always computing prototypes from the
training set. A class with fewer than `minimum_threshold_samples` correctly
classified validation examples falls back to its train-derived threshold; the
train threshold is also retained as a lower bound. `metrics.json` records the
effective source and sample count for every threshold, plus pre-rejection
`closed_set_KCA`, `known_rejection_rate`, and `accepted_known_accuracy`.
This evaluation-only change can be applied to an existing checkpoint without
retraining.

Sanitized evaluation also compares three rejection rules using the same
checkpoint: `prototype_only`, `prototype_knn`, and
`prototype_knn_margin`. The kNN score is the mean squared distance to the five
nearest training embeddings of the assigned class. The margin score is the
nearest/second-nearest prototype distance ratio. Both thresholds are calibrated
per class from correctly classified known validation samples only. Results are
written to `decision_mode_comparison.csv`; `auxiliary_predictions.csv` contains
the scores, thresholds, and predictions needed for detailed error analysis.

For the fixed-update-budget UR20 experiment, use
`iscxvpn2016_ur20_sanitized_steps.yaml`. It preserves the sanitized enhanced
model and 8x64 batch but fixes each epoch at 64 balanced batches, giving 6,400
optimizer updates over 100 epochs. Its output directory is separate from the
original sanitized run.

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20_sanitized_steps.yaml
```

The follow-up `iscxvpn2016_ur20_sanitized_arcface.yaml` experiment keeps the
same 6,400-update budget and inference architecture. During the formal training
stage it adds a normalized ArcFace classifier with weight `0.3`, scale `30`,
and angular margin `0.2`. The classifier state is checkpointed for exact resume
but is not used for prototype calibration or evaluation.

### UTI-MPC V2 architecture

`iscxvpn2016_ur20_v2.yaml` keeps the existing sanitized cache and command-line
interfaces while replacing three representation bottlenecks. Its byte branch
encodes bytes within each packet before applying a masked packet Transformer;
padding is removed before convolution. Byte and TWT packet tokens then interact
through bidirectional cross-attention and a two-way reliability gate with no
ungated residual bypass. Training uses three learnable subcenters per class and
EMA-normalized loss weights. Evaluation independently builds deterministic
spherical k-means subprototypes from training embeddings and calibrates a
threshold for every subprototype.

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20_v2.yaml

CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.evaluate \
  --config configs/iscxvpn2016_ur20_v2.yaml \
  --checkpoint outputs/iscxvpn2016_ur20_v2/best.pt
```

#### Rich packet/temporal preprocessing

`iscxvpn2016_ur20_v2_rich.yaml` is the main V2 preprocessing upgrade. Each
flow now keeps 64 packets, 32 application-payload bytes per packet in a
64-byte semantic row, and 13 temporal/protocol features per packet: normalized
log length, direction, normalized log inter-arrival time, payload ratio,
TCP/UDP type, and eight TCP flag bits. Source/destination IP addresses and
transport ports remain excluded from model input. This cache is intentionally
separate from every earlier experiment.

```bash
python -m uti_mpc.preprocess \
  --config configs/iscxvpn2016_ur20_v2_rich.yaml \
  --pcap-root /path/to/ISCX-VPN-NonVPN-2016

CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20_v2_rich.yaml
```

For a controlled comparison, `iscxvpn2016_ur20_v2_rich_raw.yaml` uses the
same features and model but disables infrastructure-flow filtering. It writes
both cache and checkpoints to different paths. Treat it as an ablation: those
extra flows contain very few packets/bytes and inherit capture-level labels,
so its headline accuracy is not directly comparable to the sanitized result.

### UTI-MPC V3 information–geometry–boundary model

V3 is an independent open-set pipeline. It removes endpoint and volatile header
fingerprints from model inputs, stores masked application-payload bytes plus
packet/burst statistics, and uses capture-disjoint train/validation/test splits.
Its inference head is a union of learned hyperspherical subprototype regions;
each region has its own trainable and known-only calibrated radius. Real unknown
classes are never used for training, checkpoint selection, or calibration.

Create the separate V3 cache and run UR20:

```bash
python -m uti_mpc.preprocess \
  --config configs/iscxvpn2016_ur20_v3.yaml \
  --pcap-root /path/to/ISCX-VPN-NonVPN-2016

CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.train \
  --config configs/iscxvpn2016_ur20_v3.yaml

CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.evaluate \
  --config configs/iscxvpn2016_ur20_v3.yaml \
  --checkpoint outputs/iscxvpn2016_ur20_v3_seed42/best.pt
```

Run the three configured seeds sequentially on one selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m uti_mpc.benchmark \
  --configs \
    configs/iscxvpn2016_ur20_v3.yaml \
    configs/iscxvpn2016_ur20_v3_seed43.yaml \
    configs/iscxvpn2016_ur20_v3_seed44.yaml
```

V3 additionally reports AUROC, AUPR-Out, FPR95, OSCR, known/open macro-F1,
capture-macro accuracy, learned/calibrated radii, and a continuous unknown score
for every flow. Existing V1/V2 caches, configurations, and checkpoints remain
supported. Use `configs/iscxvpn2016_ur20_v2_rich_grouped.yaml` when comparing
V2-rich with V3 under the same capture-disjoint split protocol.

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
