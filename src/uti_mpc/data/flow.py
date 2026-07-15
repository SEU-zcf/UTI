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
    lengths: list[float] = field(default_factory=list)
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


def packet_feature(ip: dpkt.ip.IP) -> tuple[bytes, int, int, bytes, int] | None:
    """Return 32-byte semantic row plus endpoints and IP total length."""
    raw_ip = bytes(ip)
    if len(raw_ip) < 20 or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        return None
    transport = bytes(ip.data)
    if isinstance(ip.data, dpkt.tcp.TCP):
        if len(transport) < 20:
            return None
        offset = max(20, ((transport[12] >> 4) & 0x0F) * 4)
        control = transport[4:20]
        payload = transport[offset : offset + 3]
    else:
        if len(transport) < 8:
            return None
        control = transport[4:8] + bytes(12)
        payload = transport[8:11]
    row = raw_ip[:12] + control.ljust(16, b"\x00")[:16] + payload.ljust(3, b"\x00") + b"\x00"
    total_length = int(ip.len) if int(ip.len) > 0 else len(raw_ip)
    return row, int(ip.data.sport), int(ip.data.dport), ip.src, total_length


def _finalize(state: _FlowState, npackets: int, nlengths: int) -> FlowFeatures:
    byte_tokens = np.zeros((npackets, 32), dtype=np.uint8)
    byte_mask = np.zeros(npackets, dtype=np.bool_)
    for index, row in enumerate(state.byte_rows[:npackets]):
        byte_tokens[index] = np.frombuffer(row, dtype=np.uint8)
        byte_mask[index] = True
    length_direction = np.zeros(nlengths, dtype=np.float32)
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
) -> FlowFeatures | None:
    reason = state.background_reason
    if state.packets < min_packets:
        reason = "min_packets"
    if audit is not None:
        audit.record(state.packets, state.byte_count, reason)
    if reason is not None:
        return None
    return _finalize(state, npackets, nlengths)


def iter_capture_flows(
    capture: str | Path,
    npackets: int = 64,
    nlengths: int = 32,
    idle_timeout: float = 60.0,
    min_packets: int = 1,
    background_filter: BackgroundFlowFilter | None = None,
    audit: FlowAudit | None = None,
) -> Iterator[FlowFeatures]:
    active: dict[tuple, _FlowState] = {}
    with Path(capture).open("rb") as handle:
        reader = _reader(handle)
        datalink = reader.datalink()
        for timestamp, buffer in reader:
            try:
                ip = _network_packet(buffer, datalink)
                if not isinstance(ip, dpkt.ip.IP) or ip.mf or ip.offset:
                    continue
                parsed = packet_feature(ip)
            except (dpkt.dpkt.UnpackError, ValueError, IndexError):
                continue
            if parsed is None:
                continue
            row, sport, dport, src, total_length = parsed
            dst = ip.dst
            protocol = int(ip.p)
            key = _canonical_key(src, sport, dst, dport, protocol)
            state = active.get(key)
            if state is not None and timestamp - state.last > idle_timeout:
                finished = _finish_state(state, npackets, nlengths, min_packets, audit)
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
                state.lengths.append(direction * min(total_length, 1500) / 1500.0)
            state.last = float(timestamp)
            state.packets += 1
            state.byte_count += total_length
    for state in active.values():
        finished = _finish_state(state, npackets, nlengths, min_packets, audit)
        if finished is not None:
            yield finished
