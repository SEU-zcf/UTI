from pathlib import Path

import pytest

dpkt = pytest.importorskip("dpkt")

from uti_mpc.data.flow import iter_capture_flows
from uti_mpc.data.sanitization import BackgroundFlowFilter, FlowAudit


def _ethernet_packet(src, dst, sport, dport, payload, udp=False):
    if udp:
        transport = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
        transport.ulen = 8 + len(payload)
        protocol = dpkt.ip.IP_PROTO_UDP
    else:
        transport = dpkt.tcp.TCP(sport=sport, dport=dport, seq=1, flags=dpkt.tcp.TH_ACK, data=payload)
        transport.off = 5
        protocol = dpkt.ip.IP_PROTO_TCP
    ip = dpkt.ip.IP(src=src, dst=dst, p=protocol, ttl=64, data=transport)
    ip.len = 20 + len(bytes(transport))
    ethernet = dpkt.ethernet.Ethernet(
        src=b"\x00\x01\x02\x03\x04\x05",
        dst=b"\x06\x07\x08\x09\x0a\x0b",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(ethernet)


def test_bidirectional_flow_and_packet_layout(tmp_path: Path):
    capture = tmp_path / "Chat.pcap"
    left = b"\x0a\x00\x00\x01"
    right = b"\x0a\x00\x00\x02"
    with capture.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle)
        writer.writepkt(_ethernet_packet(left, right, 1111, 443, b"abc"), ts=1.0)
        writer.writepkt(_ethernet_packet(right, left, 443, 1111, b"xyz"), ts=2.0)
        writer.writepkt(_ethernet_packet(left, right, 2222, 53, b"dns", udp=True), ts=3.0)
        writer.close()
    flows = list(iter_capture_flows(capture, npackets=4, nlengths=4))
    assert len(flows) == 2
    tcp = next(flow for flow in flows if flow.byte_mask.sum() == 2)
    assert tcp.byte_tokens.shape == (4, 32)
    assert tcp.byte_mask.tolist() == [True, True, False, False]
    assert tcp.length_direction[0] > 0
    assert tcp.length_direction[1] < 0
    assert bytes(tcp.byte_tokens[0, 28:31]) == b"abc"
    assert tcp.byte_tokens[0, 31] == 0
    udp = next(flow for flow in flows if flow.byte_mask.sum() == 1)
    assert bytes(udp.byte_tokens[0, 16:28]) == bytes(12)
    assert bytes(udp.byte_tokens[0, 28:31]) == b"dns"


def test_raw_ip_linktype_101_is_supported(tmp_path: Path):
    capture = tmp_path / "vpn_raw.pcap"
    left = b"\x0a\x00\x00\x01"
    right = b"\x0a\x00\x00\x02"
    ethernet_packet = _ethernet_packet(left, right, 1111, 443, b"abc")
    with capture.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=101)
        writer.writepkt(ethernet_packet[14:], ts=1.0)
        writer.close()
    flows = list(iter_capture_flows(capture, npackets=4, nlengths=4))
    assert len(flows) == 1
    assert flows[0].byte_mask.tolist() == [True, False, False, False]


def test_background_protocol_filter_keeps_application_flow_and_audits_dns(tmp_path: Path):
    capture = tmp_path / "mixed.pcap"
    left = b"\x0a\x00\x00\x01"
    right = b"\x0a\x00\x00\x02"
    with capture.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle)
        writer.writepkt(_ethernet_packet(left, right, 1111, 443, b"app"), ts=1.0)
        writer.writepkt(_ethernet_packet(left, right, 2222, 53, b"dns", udp=True), ts=2.0)
        writer.close()
    background_filter = BackgroundFlowFilter(udp_ports=frozenset({53}))
    audit = FlowAudit()
    flows = list(
        iter_capture_flows(
            capture,
            npackets=4,
            nlengths=4,
            background_filter=background_filter,
            audit=audit,
        )
    )
    assert len(flows) == 1
    assert audit.candidate_flows == 2
    assert audit.kept_flows == 1
    assert audit.dropped_flows == 1
    assert audit.reason_flows["udp_port_53"] == 1


def test_multicast_background_reason_is_detected():
    background_filter = BackgroundFlowFilter()
    reason = background_filter.reason(
        b"\x0a\x00\x00\x01", b"\xe0\x00\x00\xfc", 40000, 40001, 17
    )
    assert reason == "ipv4_multicast"
