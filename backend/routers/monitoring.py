"""Operator-only aggregate telemetry, hidden from OpenAPI and disabled without a token."""

from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Response
from monitoring import get_monitor

router = APIRouter()


@router.get("/monitoring", include_in_schema=False)
def monitoring_snapshot(response: Response, authorization: Optional[str] = Header(default=None)):
    monitor = get_monitor()
    if not monitor.authorized(authorization):
        raise HTTPException(status_code=404, detail="Not found")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return monitor.snapshot()
