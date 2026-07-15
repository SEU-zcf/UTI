from __future__ import annotations

import csv
import re
from pathlib import Path


ISCXVPN2016_CLASSES: dict[int, str] = {
    1: "Chat",
    2: "Email",
    3: "FileTransfer",
    4: "Streaming",
    5: "VoIP",
    6: "VPN-Chat",
    7: "VPN-Email",
    8: "VPN-FileTransfer",
    9: "VPN-Streaming",
    10: "VPN-VoIP",
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class LabelResolver:
    """Resolve one class per capture, with explicit CSV mappings taking priority."""

    def __init__(self, root: Path, mapping_csv: str | Path | None = None) -> None:
        self.root = root.resolve()
        self.explicit: dict[str, int] = {}
        if mapping_csv:
            with Path(mapping_csv).open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
                    raise ValueError("label-map CSV must contain 'path' and 'label' columns")
                for row in reader:
                    label = self._parse_label(row["label"])
                    self.explicit[row["path"].replace("\\", "/")] = label

    @staticmethod
    def _parse_label(value: str) -> int:
        stripped = value.strip()
        if stripped.isdigit():
            label = int(stripped)
            if label in ISCXVPN2016_CLASSES:
                return label
        normalized = _normalize(stripped)
        for label, name in ISCXVPN2016_CLASSES.items():
            if _normalize(name) == normalized:
                return label
        raise ValueError(f"Unknown ISCXVPN2016 label: {value}")

    def resolve(self, capture: Path) -> int:
        relative = capture.resolve().relative_to(self.root).as_posix()
        for key in (relative, capture.name):
            if key in self.explicit:
                return self.explicit[key]

        text = _normalize(relative)
        # The official dataset contains both VPN-PCAPS-* and NonVPN-PCAPs-*.
        # A substring test would incorrectly label every NonVPN directory as VPN.
        path_parts = [_normalize(part) for part in capture.resolve().relative_to(self.root).parts]
        is_vpn = (
            _normalize(capture.stem).startswith("vpn")
            or any(part.startswith("vpn") and not part.startswith("nonvpn") for part in path_parts)
        )
        if any(
            token in text
            for token in (
                "filetransfer",
                "ftps",
                "sftp",
                "scp",
                "files",
                "skypefile",
                "bittorrent",
                "torrent",
            )
        ):
            base = 3
        elif any(
            token in text
            for token in ("streaming", "youtube", "vimeo", "netflix", "spotify", "video")
        ):
            base = 4
        elif any(token in text for token in ("voip", "audio", "voice")):
            base = 5
        elif any(token in text for token in ("chat", "messenger", "hangout")):
            base = 1
        elif any(token in text for token in ("email", "mail", "imap", "smtp")):
            base = 2
        else:
            raise ValueError(
                f"Cannot infer label for {relative!r}; provide --label-map with path,label columns"
            )
        return base + 5 if is_vpn else base
