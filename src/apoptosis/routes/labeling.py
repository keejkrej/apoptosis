from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from apoptosis.api import api
from apoptosis.core.roi import CHANNEL_BRIGHTFIELD, CHANNEL_TOTO
from apoptosis.core.session import configure_session, get_session


class LabelRequest(BaseModel):
    position: str
    roi_id: int
    death_frame: int = Field(ge=0)


@api.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent.parent / "static" / "label.html").read_text()


def _session_or_503():
    try:
        return get_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@api.get("/api/session")
def session_info() -> dict[str, str | int]:
    session = _session_or_503()
    return {
        "data_dir": str(session.data_dir),
        "labels_path": str(session.labels_path),
        "total_rois": len(session.rois()),
        "labeled_rois": session.store.labeled_count(),
    }


@api.get("/api/rois")
def list_rois() -> list[dict[str, object]]:
    session = _session_or_503()
    return [
        {
            "position": roi.position,
            "roi_id": roi.roi_id,
            "key": roi.key,
            "timepoints": roi.timepoints,
            "labeled": roi.labeled,
            "death_frame": roi.death_frame,
            "is_healthy": roi.is_healthy,
        }
        for roi in session.list_rois()
    ]


@api.get("/api/rois/{position}/{roi_id}")
def roi_detail(position: str, roi_id: int) -> dict[str, object]:
    session = _session_or_503()
    try:
        roi = session.get_roi(position, roi_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return session.roi_detail(roi)


@api.get("/api/rois/{position}/{roi_id}/frame/{time_index}")
def roi_frame(
    position: str,
    roi_id: int,
    time_index: int,
    channel: str = "brightfield",
) -> Response:
    session = _session_or_503()
    try:
        roi = session.get_roi(position, roi_id)
        channel_id = CHANNEL_TOTO if channel == "toto" else CHANNEL_BRIGHTFIELD
        png = session.render_frame(roi, time_index, channel_id)
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")


@api.post("/api/labels")
def save_label(body: LabelRequest) -> dict[str, object]:
    session = _session_or_503()
    try:
        roi = session.get_roi(body.position, body.roi_id)
        label = session.save_label(roi, body.death_frame)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    detail = session.roi_detail(roi)
    label_info = detail["label"]
    assert isinstance(label_info, dict)
    return {
        "position": label.position,
        "roi_id": label.roi_id,
        "death_frame": label.death_frame,
        "is_healthy": label_info["is_healthy"],
        "labeled_at": label.labeled_at,
    }


@api.get("/api/labels")
def all_labels() -> list[dict[str, object]]:
    session = _session_or_503()
    return [
        {
            "position": label.position,
            "roi_id": label.roi_id,
            "key": label.key,
            "death_frame": label.death_frame,
            "labeled_at": label.labeled_at,
        }
        for label in session.store.all_labels()
    ]


__all__ = ["configure_session"]
