from pathlib import Path

from uti_mpc.data.labels import LabelResolver


def test_iscx_vpn_and_nonvpn_path_labels_are_distinguished(tmp_path: Path):
    root = tmp_path / "ISCX-VPN-NonVPN-2016"
    resolver = LabelResolver(root)
    assert resolver.resolve(root / "NonVPN-PCAPs-01" / "facebook_video1a.pcap") == 4
    assert resolver.resolve(root / "VPN-PCAPs-02" / "vpn_sftp_A.pcap") == 8
    assert resolver.resolve(root / "VPN-PCAPS-01" / "vpn_bittorrent.pcap") == 8
    assert resolver.resolve(root / "NonVPN-PCAPs-03" / "skype_file1.pcap") == 3
    assert resolver.resolve(root / "NonVPN-PCAPs-02" / "gmailchat1.pcapng") == 1
