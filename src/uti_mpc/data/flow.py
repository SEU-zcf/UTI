from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

import dpkt
import numpy as np

from uti_mpc.data.sanitization import BackgroundFlowFilter, FlowAudit


@dataclass
class FlowFeatures:
    flow_id: str
    byte_tokens: np.ndarray
    byte_mask: np.ndarray
    length_direction: np.ndarray
    length_mask: np.ndarray
    payload_tokens: np.ndarray | None = None
    payload_mask: np.ndarray | None = None
    packet_features: np.ndarray | None = None
    packet_mask: np.ndarray | None = None
    burst_features: np.ndarray | None = None
    burst_mask: np.ndarray | None = None


@dataclass
class _FlowState:
    key: tuple
    origin: tuple[bytes, int, bytes, int]
    start: float
    last: float
    byte_rows: list[bytes] = field(default_factory=list)
    lengths: list[float | np.ndarray] = field(default_factory=list)
    packets: int = 0
    byte_count: int = 0
    background_reason: str | None = None
    payload_rows: list[bytes] = field(default_factory=list)
    packet_metadata: list[tuple[float, int, int, float, bool, int]] = field(
        default_factory=list
    )


def _reader(handle: BinaryIO):
    position = handle.tell()
    try:
        return dpkt.pcap.Reader(handle)
    except (ValueError, dpkt.dpkt.NeedData):
        handle.seek(position)
        return dpkt.pcapng.Reader(handle)


def _network_packet(buffer: bytes, datalink: int):
    if datalink == dpkt.pcap.DLT_EN10MB:
        return dpkt.ethernet.Ethernet(buffer).data
    # libpcap's historical DLT_RAW constant is 12, while PCAP files using the
    # modern LINKTYPE_RAW registry encode raw IPv4/IPv6 packets as 101.  The
    # official ISCX VPN captures use 101.
    if datalink in {getattr(dpkt.pcap, "DLT_RAW", 12), 101}:
        return dpkt.ip.IP(buffer)
    if datalink == getattr(dpkt.pcap, "DLT_LINUX_SLL", 113):
        return dpkt.sll.SLL(buffer).data
    raise ValueError(f"Unsupported PCAP datalink type: {datalink}")


def _canonical_key(
    src: bytes, sport: int, dst: bytes, dport: int, protocol: int
) -> tuple:
    left = (src, sport)
    right = (dst, dport)
    return (left, right, protocol) if left <= right else (right, left, protocol)


def packet_feature(
    ip: dpkt.ip.IP, payload_bytes: int = 3, byte_width: int = 32
) -> tuple[bytes, int, int, bytes, int, int, int, bool] | None:
    """Return a semantic byte row plus endpoints and temporal metadata."""
    if payload_bytes < 0 or byte_width < 28 + payload_bytes:
        raise ValueError("byte_width must fit 28 header bytes and the configured payload")
    raw_ip = bytes(ip)
    if len(raw_ip) < 20 or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        return None
    transport = bytes(ip.data)
    if isinstance(ip.data, dpkt.tcp.TCP):
        if len(transport) < 20:
            return None
        offset = max(20, ((transport[12] >> 4) & 0x0F) * 4)
        control = transport[4:20]
        payload_data = transport[offset:]
        tcp_flags = int(ip.data.flags)
        is_tcp = True
    else:
        if len(transport) < 8:
            return None
        control = transport[4:8] + bytes(12)
        payload_data = transport[8:]
        tcp_flags = 0
        is_tcp = False
    payload = payload_data[:payload_bytes]
    row = (
        raw_ip[:12]
        + control.ljust(16, b"\x00")[:16]
        + payload.ljust(payload_bytes, b"\x00")
    ).ljust(byte_width, b"\x00")[:byte_width]
    total_length = int(ip.len) if int(ip.len) > 0 else len(raw_ip)
    return (
        row,
        int(ip.data.sport),
        int(ip.data.dport),
        ip.src,
        total_length,
        len(payload_data),
        tcp_flags,
        is_tcp,
    )


def payload_feature(
    ip: dpkt.ip.IP,
) -> tuple[bytes, int, int, bytes, int, int, int, bool] | None:
    """Return application payload and stable packet metadata.

    Endpoint information is returned only for bidirectional flow aggregation and
    protocol sanitization. It is never copied into the V3 model arrays.
    """
    raw_ip = bytes(ip)
    if len(raw_ip) < 20 or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        return None
    transport = bytes(ip.data)
    if isinstance(ip.data, dpkt.tcp.TCP):
        if len(transport) < 20:
            return None
        offset = max(20, ((transport[12] >> 4) & 0x0F) * 4)
        payload = transport[offset:]
        flags = int(ip.data.flags)
        is_tcp = True
    else:
        if len(transport) < 8:
            return None
        payload = transport[8:]
        flags = 0
        is_tcp = False
    total_length = int(ip.len) if int(ip.len) > 0 else len(raw_ip)
    return (
        payload,
        int(ip.data.sport),
        int(ip.data.dport),
        ip.src,
        total_length,
        len(payload),
        flags,
        is_tcp,
    )


def _v3_packet_vector(
    direction: float,
    packet_index: int,
    total_length: int,
    payload_length: int,
    iat: float,
    elapsed: float,
    is_tcp: bool,
    tcp_flags: int,
    max_packets: int,
    iat_clip: float,
) -> np.ndarray:
    flags = [float(bool(tcp_flags & (1 << bit))) for bit in range(8)]
    return np.asarray(
        [
            direction,
            packet_index / max(max_packets - 1, 1),
            np.log1p(min(total_length, 1500)) / np.log1p(1500),
            np.log1p(min(payload_length, 1500)) / np.log1p(1500),
            min(payload_length / max(total_length, 1), 1.0),
            np.log1p(min(iat, iat_clip)) / np.log1p(iat_clip),
            np.log1p(min(elapsed, iat_clip)) / np.log1p(iat_clip),
            float(is_tcp),
            *flags,
        ],
        dtype=np.float32,
    )


def _v3_bursts(
    metadata: list[tuple[float, int, int, float, bool, int]],
    max_bursts: int,
    iat_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((max_bursts, 8), dtype=np.float32)
    mask = np.zeros(max_bursts, dtype=np.bool_)
    if not metadata:
        return features, mask
    groups: list[list[tuple[float, int, int, float, bool, int]]] = []
    for item in metadata:
        if not groups or groups[-1][0][0] != item[0]:
            groups.append([item])
        else:
            groups[-1].append(item)
    byte_scale = np.log1p(64 * 1500)
    time_scale = np.log1p(iat_clip)
    for index, group in enumerate(groups[:max_bursts]):
        direction = group[0][0]
        total_bytes = sum(item[1] for item in group)
        payload_bytes = sum(item[2] for item in group)
        iats = np.asarray([item[3] for item in group], dtype=np.float32)
        duration = float(iats[1:].sum()) if len(iats) > 1 else 0.0
        features[index] = np.asarray(
            [
                direction,
                np.log1p(len(group)) / np.log1p(64),
                np.log1p(total_bytes) / byte_scale,
                np.log1p(payload_bytes) / byte_scale,
                np.log1p(min(duration, iat_clip)) / time_scale,
                np.log1p(min(float(iats.mean()), iat_clip)) / time_scale,
                np.log1p(min(float(iats.std()), iat_clip)) / time_scale,
                min(payload_bytes / max(total_bytes, 1), 1.0),
            ],
            dtype=np.float32,
        )
        mask[index] = True
    return features, mask


def _finalize_v3(
    state: _FlowState,
    npackets: int,
    payload_bytes: int,
    max_bursts: int,
    iat_clip: float,
) -> FlowFeatures:
    payload_tokens = np.zeros((npackets, payload_bytes), dtype=np.uint8)
    payload_mask = np.zeros((npackets, payload_bytes), dtype=np.bool_)
    packet_features = np.zeros((npackets, 16), dtype=np.float32)
    packet_mask = np.zeros(npackets, dtype=np.bool_)
    for index, (payload, metadata) in enumerate(
        zip(state.payload_rows[:npackets], state.packet_metadata[:npackets], strict=True)
    ):
        width = min(len(payload), payload_bytes)
        if width:
            payload_tokens[index, :width] = np.frombuffer(payload[:width], dtype=np.uint8)
            payload_mask[index, :width] = True
        direction, total_length, payload_length, iat, is_tcp, flags = metadata
        elapsed = max(0.0, state.last - state.start) if index == len(state.packet_metadata) - 1 else sum(
            item[3] for item in state.packet_metadata[: index + 1]
        )
        packet_features[index] = _v3_packet_vector(
            direction,
            index,
            total_length,
            payload_length,
            iat,
            elapsed,
            is_tcp,
            flags,
            npackets,
            iat_clip,
        )
        packet_mask[index] = True
    burst_features, burst_mask = _v3_bursts(
        state.packet_metadata[:npackets], max_bursts, iat_clip
    )
    digest = hashlib.sha1(repr((state.key, state.start)).encode("utf-8")).hexdigest()[:20]
    # Legacy arrays remain valid empty placeholders so FlowFeatures has one
    # stable type; V3 preprocessing writes only the explicit V3 arrays.
    return FlowFeatures(
        digest,
        np.zeros((npackets, 1), dtype=np.uint8),
        packet_mask.copy(),
        packet_features.copy(),
        packet_mask.copy(),
        payload_tokens,
        payload_mask,
        packet_features,
        packet_mask,
        burst_features,
        burst_mask,
    )


def _finalize(
    state: _FlowState,
    npackets: int,
    nlengths: int,
    byte_width: int,
    temporal_features: int,
) -> FlowFeatures:
    byte_tokens = np.zeros((npackets, byte_width), dtype=np.uint8)
    byte_mask = np.zeros(npackets, dtype=np.bool_)
    for index, row in enumerate(state.byte_rows[:npackets]):
        byte_tokens[index] = np.frombuffer(row, dtype=np.uint8)
        byte_mask[index] = True
    length_direction = np.zeros(
        (nlengths, temporal_features) if temporal_features > 1 else nlengths,
        dtype=np.float32,
    )
    length_mask = np.zeros(nlengths, dtype=np.bool_)
    count = min(len(state.lengths), nlengths)
    if count:
        length_direction[:count] = np.asarray(state.lengths[:count], dtype=np.float32)
        length_mask[:count] = True
    digest = hashlib.sha1(repr((state.key, state.start)).encode("utf-8")).hexdigest()[:20]
    return FlowFeatures(digest, byte_tokens, byte_mask, length_direction, length_mask)


def _finish_state(
    state: _FlowState,
    npackets: int,
    nlengths: int,
    min_packets: int,
    audit: FlowAudit | None,
    byte_width: int,
    temporal_features: int,
    representation: str = "legacy",
    payload_bytes: int = 3,
    max_bursts: int = 32,
    iat_clip: float = 60.0,
) -> FlowFeatures | None:
    reason = state.background_reason
    if state.packets < min_packets:
        reason = "min_packets"
    if audit is not None:
        audit.record(state.packets, state.byte_count, reason)
    if reason is not None:
        return None
    if representation == "v3":
        return _finalize_v3(
            state, npackets, payload_bytes, max_bursts, iat_clip
        )
    return _finalize(state, npackets, nlengths, byte_width, temporal_features)


def iter_capture_flows(
    capture: str | Path,
    npackets: int = 64,
    nlengths: int = 32,
    idle_timeout: float = 60.0,
    min_packets: int = 1,
    background_filter: BackgroundFlowFilter | None = None,
    audit: FlowAudit | None = None,
    payload_bytes: int = 3,
    byte_width: int = 32,
    rich_temporal_features: bool = False,
    iat_clip: float = 60.0,
    representation: str = "legacy",
    max_bursts: int = 32,
) -> Iterator[FlowFeatures]:
    if npackets < 1 or nlengths < 1:
        raise ValueError("npackets and nlengths must be positive")
    if representation not in {"legacy", "v3"}:
        raise ValueError("representation must be 'legacy' or 'v3'")
    if representation == "legacy" and (payload_bytes < 0 or byte_width < 28 + payload_bytes):
        raise ValueError("byte_width must fit 28 header bytes and the configured payload")
    if representation == "v3" and (payload_bytes < 1 or max_bursts < 1):
        raise ValueError("V3 payload_bytes and max_bursts must be positive")
    if iat_clip <= 0:
        raise ValueError("iat_clip must be positive")
    temporal_features = 13 if rich_temporal_features else 1
    active: dict[tuple, _FlowState] = {}
    with Path(capture).open("rb") as handle:
        reader = _reader(handle)
        datalink = reader.datalink()
        for timestamp, buffer in reader:
            try:
                ip = _network_packet(buffer, datalink)
                if not isinstance(ip, dpkt.ip.IP) or ip.mf or ip.offset:
                    continue
                parsed = (
                    payload_feature(ip)
                    if representation == "v3"
                    else packet_feature(ip, payload_bytes=payload_bytes, byte_width=byte_width)
                )
            except (dpkt.dpkt.UnpackError, ValueError, IndexError):
                continue
            if parsed is None:
                continue
            (
                row,
                sport,
                dport,
                src,
                total_length,
                payload_length,
                tcp_flags,
                is_tcp,
            ) = parsed
            dst = ip.dst
            protocol = int(ip.p)
            key = _canonical_key(src, sport, dst, dport, protocol)
            state = active.get(key)
            if state is not None and timestamp - state.last > idle_timeout:
                finished = _finish_state(
                    state,
                    npackets,
                    nlengths,
                    min_packets,
                    audit,
                    byte_width,
                    temporal_features,
                    representation,
                    payload_bytes,
                    max_bursts,
                    iat_clip,
                )
                if finished is not None:
                    yield finished
                state = None
            if state is None:
                state = _FlowState(
                    key=key,
                    origin=(src, sport, dst, dport),
                    start=float(timestamp),
                    last=float(timestamp),
                    background_reason=(
                        background_filter.reason(src, dst, sport, dport, protocol)
                        if background_filter is not None
                        else None
                    ),
                )
                active[key] = state
            direction = 1.0 if (src, sport, dst, dport) == state.origin else -1.0
            if representation == "v3":
                if len(state.payload_rows) < npackets:
                    iat = (
                        0.0
                        if state.packets == 0
                        else max(0.0, float(timestamp) - state.last)
                    )
                    state.payload_rows.append(row)
                    state.packet_metadata.append(
                        (direction, total_length, payload_length, iat, is_tcp, tcp_flags)
                    )
            elif len(state.byte_rows) < npackets:
                state.byte_rows.append(row)
            if len(state.lengths) < nlengths:
                if rich_temporal_features:
                    iat = (
                        0.0
                        if state.packets == 0
                        else max(0.0, float(timestamp) - state.last)
                    )
                    flags = [float(bool(tcp_flags & (1 << bit))) for bit in range(8)]
                    state.lengths.append(
                        np.asarray(
                            [
                                np.log1p(min(total_length, 1500)) / np.log1p(1500),
                                direction,
                                np.log1p(min(iat, iat_clip)) / np.log1p(iat_clip),
                                min(payload_length / max(total_length, 1), 1.0),
                                float(is_tcp),
                                *flags,
                            ],
                            dtype=np.float32,
                        )
                    )
                else:
                    state.lengths.append(direction * min(total_length, 1500) / 1500.0)
            state.last = float(timestamp)
            state.packets += 1
            state.byte_count += total_length
    for state in active.values():
        finished = _finish_state(
            state,
            npackets,
            nlengths,
            min_packets,
            audit,
            byte_width,
            temporal_features,
            representation,
            payload_bytes,
            max_bursts,
            iat_clip,
        )
        if finished is not None:
            yield finished
