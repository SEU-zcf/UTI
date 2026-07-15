from __future__ import annotations

import ipaddress
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BackgroundFlowFilter:
    """Conservative protocol rules for removing mislabeled infrastructure flows."""

    udp_ports: frozenset[int] = frozenset()
    tcp_ports: frozenset[int] = frozenset()
    exclude_ipv4_multicast: bool = True
    exclude_limited_broadcast: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BackgroundFlowFilter | None":
        if not bool(config.get("enabled", False)):
            return None
        return cls(
            udp_ports=_validated_ports(config.get("udp_ports", [])),
            tcp_ports=_validated_ports(config.get("tcp_ports", [])),
            exclude_ipv4_multicast=bool(config.get("exclude_ipv4_multicast", True)),
            exclude_limited_broadcast=bool(config.get("exclude_limited_broadcast", True)),
        )

    def reason(
        self, src: bytes, dst: bytes, sport: int, dport: int, protocol: int
    ) -> str | None:
        ports = self.udp_ports if protocol == 17 else self.tcp_ports if protocol == 6 else frozenset()
        matched = sorted({int(sport), int(dport)} & ports)
        if matched:
            protocol_name = "udp" if protocol == 17 else "tcp"
            return f"{protocol_name}_port_{matched[0]}"
        destination = ipaddress.IPv4Address(dst)
        if self.exclude_ipv4_multicast and destination.is_multicast:
            return "ipv4_multicast"
        if self.exclude_limited_broadcast and dst == b"\xff\xff\xff\xff":
            return "ipv4_limited_broadcast"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "udp_ports": sorted(self.udp_ports),
            "tcp_ports": sorted(self.tcp_ports),
            "exclude_ipv4_multicast": self.exclude_ipv4_multicast,
            "exclude_limited_broadcast": self.exclude_limited_broadcast,
        }


def _validated_ports(values: Sequence[int]) -> frozenset[int]:
    ports = frozenset(int(value) for value in values)
    if any(not 0 <= port <= 65535 for port in ports):
        raise ValueError("Background-filter ports must be in [0, 65535]")
    return ports


@dataclass
class FlowAudit:
    candidate_flows: int = 0
    candidate_packets: int = 0
    candidate_bytes: int = 0
    kept_flows: int = 0
    kept_packets: int = 0
    kept_bytes: int = 0
    dropped_flows: int = 0
    dropped_packets: int = 0
    dropped_bytes: int = 0
    reason_flows: Counter[str] = field(default_factory=Counter)
    reason_packets: Counter[str] = field(default_factory=Counter)
    reason_bytes: Counter[str] = field(default_factory=Counter)

    def record(self, packets: int, byte_count: int, reason: str | None) -> None:
        self.candidate_flows += 1
        self.candidate_packets += int(packets)
        self.candidate_bytes += int(byte_count)
        if reason is None:
            self.kept_flows += 1
            self.kept_packets += int(packets)
            self.kept_bytes += int(byte_count)
            return
        self.dropped_flows += 1
        self.dropped_packets += int(packets)
        self.dropped_bytes += int(byte_count)
        self.reason_flows[reason] += 1
        self.reason_packets[reason] += int(packets)
        self.reason_bytes[reason] += int(byte_count)

    def merge(self, other: "FlowAudit") -> None:
        for name in (
            "candidate_flows",
            "candidate_packets",
            "candidate_bytes",
            "kept_flows",
            "kept_packets",
            "kept_bytes",
            "dropped_flows",
            "dropped_packets",
            "dropped_bytes",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.reason_flows.update(other.reason_flows)
        self.reason_packets.update(other.reason_packets)
        self.reason_bytes.update(other.reason_bytes)

    def to_dict(self) -> dict[str, Any]:
        reasons = sorted(self.reason_flows)
        return {
            "candidate": {
                "flows": self.candidate_flows,
                "packets": self.candidate_packets,
                "bytes": self.candidate_bytes,
            },
            "kept": {
                "flows": self.kept_flows,
                "packets": self.kept_packets,
                "bytes": self.kept_bytes,
            },
            "dropped": {
                "flows": self.dropped_flows,
                "packets": self.dropped_packets,
                "bytes": self.dropped_bytes,
            },
            "by_reason": {
                reason: {
                    "flows": self.reason_flows[reason],
                    "packets": self.reason_packets[reason],
                    "bytes": self.reason_bytes[reason],
                }
                for reason in reasons
            },
        }
