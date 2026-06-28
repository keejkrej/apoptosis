from __future__ import annotations

from pathlib import Path

from apoptosis.services.labeling import LabelingSession

_session: LabelingSession | None = None

DEFAULT_DATA_DIR = Path("/home/jack/data/lisca_review/fig6/20260327")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "labels.json"


def get_session_optional() -> LabelingSession | None:
    return _session


def get_session() -> LabelingSession:
    if _session is None:
        msg = "Labeling session is not configured"
        raise RuntimeError(msg)
    return _session


def configure_session(
    data_dir: Path,
    labels_path: Path | None = None,
) -> LabelingSession:
    global _session
    resolved_labels = labels_path or DEFAULT_LABELS_PATH
    _session = LabelingSession(data_dir=data_dir, labels_path=resolved_labels)
    return _session
