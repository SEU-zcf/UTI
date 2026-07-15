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
) -> FlowFeatures | None:
    reason = state.background_reason
    if state.packets < min_packets:
        reason = "min_packets"
    if audit is not None:
        audit.record(state.packets, state.byte_count, reason)
    if reason is not None:
        return None
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
) -> Iterator[FlowFeatures]:
    if npackets < 1 or nlengths < 1:
        raise ValueError("npackets and nlengths must be positive")
    if payload_bytes < 0 or byte_width < 28 + payload_bytes:
        raise ValueError("byte_width must fit 28 header bytes and the configured payload")
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
                parsed = packet_feature(
                    ip, payload_bytes=payload_bytes, byte_width=byte_width
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
            if len(state.byte_rows) < npackets:
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
        )
        if finished is not None:
            yield finished
