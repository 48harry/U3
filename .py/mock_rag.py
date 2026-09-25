"""
Stand-in for the ontology / RAG service, so the ML side can test the push path before the real one exists.

It accepts the handoff document (POST /diagnose, schema rag_handoff v1.0 - see main.RagHandoff) and answers with
the report shape the real service is expected to return (7 fixed sections, RSW용접건_매뉴얼_RAG_활용정리 §9).
The ML handoff carries steps (1)-(2) only (what is unusual, symptom translation, situation ids); sections that need
causes, checks and manual pages are left as placeholders here - the real ontology fills them from situation_ids.

    uvicorn mock_rag:app --app-dir .py --port 8001
    RSW_RAG_URL=http://127.0.0.1:8001/diagnose uvicorn main:app          # main pushes every handoff here
    python .py/replay.py test/test_0.csv --start-hours 156 --chunk-s 600  # then: GET :8001/received

The real service only has to keep POST /diagnose and the top-level response keys.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="mock RAG (RSW root cause)", version="0.2.0")
RECEIVED: deque[dict[str, Any]] = deque(maxlen=200)
TODO = "[mock] 온톨로지가 situation_ids로 채울 항목"


def report(h: dict[str, Any]) -> dict[str, Any]:
    fc = h.get("fault_class")
    return {
        "detected_anomaly": (f"{h['gun_id']} {h['detected_at']} - {h.get('summary_ko', '')} "
                             f"(트리거: {h['trigger']['source']})"),
        "suspected_device": TODO,
        "cause_candidates": [{"situation_id": s, "name": TODO} for s in h.get("situation_ids", [])[:3]],
        "manual_references": [],
        "additional_checks": [s["name_ko"] for s in h.get("symptoms", [])],
        "recommended_actions": [TODO],
        "confidence_and_limits": ([f"고장 유형 근거: {fc['basis']}"] if fc else []) + h.get("caveats", []),
    }


@app.post("/diagnose")
async def diagnose(handoff: dict[str, Any]) -> dict[str, Any]:
    RECEIVED.append(handoff)
    return {"event_id": handoff.get("event_id"), "status": "ok", "generator": "mock", "report": report(handoff)}


@app.get("/received")
async def received(limit: int = 20) -> list[dict[str, Any]]:
    return [{"event_id": h.get("event_id"), "gun_id": h.get("gun_id"), "detected_at": h.get("detected_at"),
             "summary_ko": h.get("summary_ko"), "situation_ids": h.get("situation_ids")}
            for h in list(RECEIVED)[-limit:]]
