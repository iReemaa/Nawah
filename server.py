"""
Nawah API server
=================
Thin FastAPI wrapper around `nawah_pipeline.py`. Run locally with:

    uvicorn server:app --reload --port 8000

Endpoints:
  GET  /api/health          -> {"ok": true/false, "error": "..." }
  POST /api/plan            -> runs the full 5-stage pipeline once
  POST /api/chat            -> session-aware conversational turn (nawah_turn)

CORS is wide open (`*`) here for local development. Before you put this
behind a public domain, replace `allow_origins=["*"]` below with the exact
origin(s) your frontend is served from.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Nawah Advisory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Import the pipeline lazily-but-once, and remember if it failed so that
# /api/health can explain why (missing OPENAI_API_KEY, missing data files,
# missing sklearn, etc.) instead of the whole process crashing at boot.
# --------------------------------------------------------------------------
_pipeline = None
_import_error: Optional[str] = None

try:
    import nawah_pipeline as _pipeline
except Exception as e:  # noqa: BLE001 - we want to surface ANY startup failure
    _import_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline failed to load. Check GET /api/health for details "
                "(usually a missing OPENAI_API_KEY or a missing knowledge-base "
                "file under NAWAH_DATA_DIR)."
            ),
        )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class PlanRequest(BaseModel):
    business_idea: str = Field(..., min_length=2, description="Free-form business idea, Arabic or English")
    nationality: str = Field("unspecified", description="saudi | non_saudi | unspecified")
    answers: Dict[str, Any] = Field(default_factory=dict, description="User answers for dynamic setup questions")

class PlanResponse(BaseModel):
    final_text: str
    language: str
    plan: Dict[str, Any]
    verifier_1: Dict[str, Any]
    verifier_2: Dict[str, Any]
    cost_total_sar: float


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    session: Dict[str, Any]
    cost_total_sar: float


class LocationRequest(BaseModel):
    business_idea: str = Field(..., min_length=2)
    nationality: str = Field("unspecified", description="unused here, kept for parity with /api/plan calls")
    city_hint: Optional[str] = Field(None, description="City name if known, e.g. 'الرياض' or 'Riyadh'")


class LocationArea(BaseModel):
    name: str
    why: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None


class LocationResponse(BaseModel):
    areas: list[LocationArea] = Field(default_factory=list)
    note: Optional[str] = None
    disclaimer: Optional[str] = None
    suggestions: str  # kept for backwards compatibility with older frontends
    cost_total_sar: float


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    if _pipeline is None:
        return {"ok": False, "error": _import_error}
    return {"ok": True}


@app.post("/api/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest):
    _require_pipeline()
    nationality = req.nationality if req.nationality in ("saudi", "non_saudi", "unspecified") else "unspecified"
    try:
        result = _pipeline.run_nawah_pipeline_v3(req.business_idea, nationality, answers=req.answers)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return PlanResponse(
        final_text=result["final_text"],
        language=result["language"],
        plan=result["plan"],
        verifier_1=result["verifier_1"],
        verifier_2=result["verifier_2"],
        cost_total_sar=result["cost_total_sar"],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _require_pipeline()
    try:
        reply, session, cost_log = _pipeline.nawah_turn(req.message, req.session)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return ChatResponse(
        reply=reply,
        session=session,
        cost_total_sar=round(sum(e["cost_sar"] for e in cost_log), 4),
    )


@app.post("/api/location", response_model=LocationResponse)
def suggest_locations(req: LocationRequest):
    """For users who answered 'no location yet' in the pre-plan questions.
    Not grounded in the audited knowledge base (there's no rent/competition
    data in it) -- backed by live web_search instead, and the agent is
    instructed to stay directional and source-attributed rather than assert
    numbers it didn't actually find. See location_advisor_agent's docstring."""
    _require_pipeline()
    try:
        result = _pipeline.location_advisor_agent(req.business_idea, city_hint=req.city_hint)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    return LocationResponse(
        areas=result.get("areas", []),
        note=result.get("note"),
        disclaimer=result.get("disclaimer"),
        suggestions=result["suggestions"],
        cost_total_sar=result["cost_total_sar"],
    )