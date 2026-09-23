"""
RSW welding-gun anomaly detection — real-time inference API.

Loads models/baseline_<model>.joblib (written by .py/train.py), keeps a per-gun rolling
buffer of raw 1 Hz sensor readings, applies the same preprocessing as .py/preprocess.py
online, scores the trailing window and returns a JSON document for the ontology / RAG stage.

    uvicorn main:app --reload
    RSW_MODEL_PATH=models/baseline_pca.joblib uvicorn main:app

Endpoints
    GET  /health                  liveness + model status
    GET  /model                   model card (window, threshold, features, preprocessing, val/test metrics)
    POST /predict                 append readings for one gun, score the latest window
    GET  /guns                    buffer / alarm state of every gun seen so far
    DELETE /guns/{gun_id}         forget a gun's buffer (e.g. after maintenance)

A gun needs `window_s` seconds of readings before the first score (HTTP 202 "warming_up"
until then); the 10-minute rolling features are exact once 600 s of history exist.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, ".py"))  # train.py / preprocess.py live there
import train as trainlib  # noqa: E402  (also makes PCADetector importable for pca bundles)
from preprocess import BINARY_COL, ROLL, SENSOR_COLS, TERMINAL_TO_CLASS, VALUE_COLS, fill_gaps  # noqa: E402

MODEL_PATH = os.environ.get("RSW_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "baseline_iforest.joblib"))
ROLL_S = int(pd.Timedelta(ROLL).total_seconds())  # 10-minute rolling features, as in preprocess.py
# defaults for bundles written before preprocess_config was stored; a current bundle overrides them
DEFAULT_PREPROCESS = {"gap_fill_limit": 60, "outlier_hi": 5.0, "outlier_lo": 0.0, "resample": None}
BUFFER_S = 1800  # per-gun history kept in memory (>= ROLL_S + window + gap limit)
TOP_K_FEATURES = 8
CODE_TO_CLASS = TERMINAL_TO_CLASS  # E012 -> E01 ...
FaultClass = Literal["E01", "E02", "E03", "E04"]
SENSOR_NAME = {
    "c1": "Electrode cap offset", "c2": "Electrode force", "c3": "Electrode position",
    "c4": "Force build-up", "c5": "Balance pressure", "c6": "Friction", "c7": "Maximum aperture",
    "c8": "Maximum electrode force", "c9": "Start friction", "c10": "US2",
    "c11": "Welding point count", "c12": "Position count", "c13": "Setpoint counterbalance pressure",
    "c14": "Setpoint electrode force", "c15": "Setpoint electrode position",
    "c16": "Setpoint sheet thickness", "c17": "Setpoint velocity", "c18": "Setpoint force build-up",
    "c19": "Offset value in robot",
    "welds_delta": "Welds per second", "pos_delta": "Position moves per second",
    "welds_10min": "Welds in last 10 min", "weld_duty_10min": "Welding duty cycle (10 min)",
    "error_share_10min": "Share of time in error state (10 min)",
    "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)",
}
log = logging.getLogger("rsw-api")


# ============================================================ input schema
class SensorReading(BaseModel):
    """One 1 Hz sample from the welding controller (same columns as the dataset CSVs)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "time": "2021-09-05T02:23:20Z", "c1": 0.0, "c2": 0, "c3": 104.1, "c4": 0, "c5": 0, "c6": -93,
        "c7": 160.58, "c8": 6026, "c9": 106, "c10": "off", "c11": 4374310, "c12": 18192620,
        "c13": 14.99, "c14": 3000, "c15": 130.0, "c16": 2.6, "c17": 1, "c18": 0, "c19": 940521253,
        "error": "0"}})

    time: datetime = Field(description="sample timestamp (ISO 8601, UTC if no offset)")
    c1: float = Field(description="Electrode cap offset")
    c2: float = Field(description="Electrode force")
    c3: float = Field(description="Electrode position")
    c4: float = Field(description="Force build-up")
    c5: float = Field(description="Balance pressure")
    c6: float = Field(description="Friction")
    c7: float = Field(description="Maximum aperture (gun constant)")
    c8: float = Field(description="Maximum electrode force (gun constant)")
    c9: float = Field(description="Start friction (gun constant)")
    c10: bool = Field(description="US2 status; accepts true/false, 'on'/'off', 1/0")
    c11: float = Field(ge=0, description="Welding point count (cumulative)")
    c12: float = Field(ge=0, description="Position count (cumulative)")
    c13: float = Field(description="Setpoint counterbalance pressure")
    c14: float = Field(description="Setpoint electrode force")
    c15: float = Field(description="Setpoint electrode position")
    c16: float = Field(description="Setpoint sheet thickness; > 5 or <= 0 = non-welding operation")
    c17: float = Field(description="Setpoint velocity")
    c18: float = Field(description="Setpoint force build-up")
    c19: float = Field(description="Offset value in robot")
    error: str = Field("0", pattern=r"^(0|E\d{3})$", description="controller state code, '0' = none")

    @field_validator("c10", mode="before")
    @classmethod
    def _parse_c10(cls, v: Any) -> bool:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("on", "true", "1"):
                return True
            if s in ("off", "false", "0"):
                return False
            raise ValueError("c10 must be on/off")
        return bool(v)

    @field_validator("time", mode="after")
    @classmethod
    def _naive_utc(cls, v: datetime) -> datetime:
        # preprocess.py works in naive UTC (pd.to_datetime(utc=True).tz_localize(None)); match it
        return v.astimezone(timezone.utc).replace(tzinfo=None) if v.tzinfo else v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gun_id: str = Field(min_length=1, max_length=64, description="welding gun identifier")
    readings: list[SensorReading] = Field(min_length=1, max_length=3600,
                                          description="new samples, any order; appended to the gun's buffer")


# ============================================================ output schema
class FeatureContribution(BaseModel):
    feature: str = Field(description="window feature name, e.g. 'c5_mean'")
    sensor: str = Field(description="underlying signal, e.g. 'c5'")
    sensor_name: str
    statistic: Literal["mean", "std"] = Field(description="window statistic of the signal")
    value: float = Field(description="feature value in the model's (z-scored) input space")
    reference: float = Field(description="typical value on normal windows")
    contribution: float = Field(description="score decrease when this feature is set to its reference; > 0 pushes toward anomaly")
    share: float = Field(ge=0, le=1, description="contribution / sum of positive contributions")


class WindowContext(BaseModel):
    latest_error_code: str
    error_active_share: float = Field(ge=0, le=1, description="share of window samples with a non-zero code")
    non_welding_share: float = Field(ge=0, le=1, description="share of window samples flagged as cap dressing / changing")
    welds_in_window: float
    welds_10min: float
    weld_duty_10min: float = Field(ge=0, le=1)
    known_code_class_hint: FaultClass | None = Field(
        description="fault class whose terminal code equals the latest error code (E012→E01, E016→E02, E028→E03, E029→E04)")


class ModelInfo(BaseModel):
    name: str
    type: str
    created: str
    window_s: int
    threshold: float
    threshold_q: float
    sustain: int
    n_features: int


class AnomalyResult(BaseModel):
    """Document handed to the ontology / root-cause stage."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "gun_id": "G17", "status": "ok", "window_start": "2021-09-05T02:22:21", "window_end": "2021-09-05T02:23:20",
        "n_samples": 60, "history_s": 1800, "is_anomaly": True, "anomaly_score": 0.61, "threshold": 0.54,
        "score_z": 4.6, "severity": "critical", "sustained_alarm": True, "consecutive_alarms": 3,
        "contributing_features": [{"feature": "c5_mean", "sensor": "c5", "sensor_name": "Balance pressure",
                                   "statistic": "mean", "value": -3.1, "reference": 0.02, "contribution": 0.08, "share": 0.41}],
        "context": {"latest_error_code": "E029", "error_active_share": 1.0, "non_welding_share": 0.0,
                    "welds_in_window": 0, "welds_10min": 12, "weld_duty_10min": 0.02, "known_code_class_hint": "E04"},
        "model": {"name": "baseline_iforest", "type": "iforest", "created": "2026-09-22T16:33:02", "window_s": 60,
                  "threshold": 0.54, "threshold_q": 0.99, "sustain": 3, "n_features": 48}}})

    gun_id: str
    status: Literal["ok"] = "ok"
    window_start: datetime
    window_end: datetime
    n_samples: int = Field(description="1 Hz samples in the scored window")
    history_s: int = Field(description="seconds of contiguous history behind the window; < 600 means rolling features are partial")
    is_anomaly: bool = Field(description="anomaly_score > threshold")
    anomaly_score: float = Field(description="higher = more anomalous")
    threshold: float
    score_z: float = Field(description="(score - normal mean) / normal std on training windows")
    severity: Literal["normal", "warning", "critical"] = Field(
        description="normal: below threshold; warning: above; critical: `sustain` consecutive windows above")
    sustained_alarm: bool
    consecutive_alarms: int
    contributing_features: list[FeatureContribution] = Field(description="top features by contribution, descending")
    context: WindowContext
    model: ModelInfo


class WarmingUp(BaseModel):
    gun_id: str
    status: Literal["warming_up"] = "warming_up"
    n_samples: int
    samples_needed: int
    message: str


class GunStatus(BaseModel):
    gun_id: str
    n_samples: int
    first_time: datetime | None
    last_time: datetime | None
    consecutive_alarms: int
    last_score: float | None


# ============================================================ model + state
class Detector:
    """Wraps the joblib bundle: online preprocessing -> window vector -> score -> attributions."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.bundle: dict[str, Any] = joblib.load(path)
        b = self.bundle
        self.model = b["model"]
        self.window: int = int(b["window"])
        self.threshold: float = float(b["threshold"])
        self.sustain: int = int(b.get("sustain", 3))
        self.feat_cols: list[str] = list(b["feature_cols"])
        self.model_cols: list[str] = list(b["model_cols"])
        scaler = b.get("scaler") or {}
        if scaler.get("scale", "global") != "global":
            raise RuntimeError(f"bundle was trained on '{scaler['scale']}'-scaled data; the API can only reproduce "
                               "the global scaler online - retrain from preprocess.py --scale global")
        self.scaler: dict[str, dict[str, float]] = scaler.get("params") or {}
        cfg = {**DEFAULT_PREPROCESS, **{k: v for k, v in (b.get("preprocess_config") or {}).items()
                                        if k in DEFAULT_PREPROCESS}}
        if cfg["resample"]:
            raise RuntimeError(f"bundle was trained on {cfg['resample']}-resampled data; the API scores 1 Hz windows "
                               "- retrain without --resample")
        if not b.get("preprocess_config"):
            log.warning("bundle has no preprocess_config (retrain with the current train.py); using defaults %s", cfg)
        self.gap_fill_limit: int = int(cfg["gap_fill_limit"])
        self.outlier_hi: float = float(cfg["outlier_hi"])
        self.outlier_lo: float = float(cfg["outlier_lo"])
        tr = b.get("metrics", {}).get("train", {})
        self.score_mean: float = float(tr.get("score_mean", 0.0))
        self.score_std: float = float(tr.get("score_std", 1.0)) or 1.0
        ref = b.get("feature_reference")
        self.reference = np.asarray(ref, dtype=np.float32) if ref else np.zeros(len(self.model_cols), np.float32)
        if not ref:
            log.warning("bundle has no feature_reference (retrain with the current train.py); using zeros")
        self.info = ModelInfo(name=os.path.splitext(os.path.basename(path))[0], type=str(b.get("model_type", "?")),
                              created=str(b.get("created", "?")), window_s=self.window, threshold=self.threshold,
                              threshold_q=float(b.get("threshold_q", float("nan"))), sustain=self.sustain,
                              n_features=len(self.model_cols))

    # ---- preprocessing identical in spirit to preprocess.py, on a rolling buffer
    def features(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
        df = raw.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        raw_index = df.index
        df = df.resample("s").asfreq()
        present = df.index.isin(raw_index)
        code = df["error"].where(present).ffill().fillna("0").astype(str)
        x = fill_gaps(df[SENSOR_COLS].astype("float32"), present, self.gap_fill_limit)
        code = code.loc[x.index]
        if len(x) == 0:
            return None
        # only the last contiguous segment is scored (windows never straddle a long gap)
        step = x.index.to_series().diff().dt.total_seconds().fillna(1)
        seg = (step > 1).cumsum()
        x, code = x[seg == seg.iloc[-1]], code[seg == seg.iloc[-1]]
        non_welding = (x["c16"] > self.outlier_hi) | (x["c16"] <= self.outlier_lo)
        x.loc[non_welding, VALUE_COLS + [BINARY_COL]] = np.nan
        x = x.ffill().bfill()

        f = pd.DataFrame(index=x.index)
        f["welds_delta"] = x["c11"].diff().fillna(0).clip(lower=0)
        f["pos_delta"] = x["c12"].diff().fillna(0).clip(lower=0)
        f["welds_10min"] = f["welds_delta"].rolling(f"{ROLL_S}s", min_periods=1).sum()
        f["weld_duty_10min"] = (x["c2"] > 0).astype("float32").rolling(f"{ROLL_S}s", min_periods=1).mean()
        error_active = (code != "0").astype("float32")
        f["error_share_10min"] = error_active.rolling(f"{ROLL_S}s", min_periods=1).mean()
        for c in VALUE_COLS + [BINARY_COL]:
            f[c] = x[c]
        hour = f.index.hour + f.index.minute / 60
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        for c, p in self.scaler.items():
            if c in f.columns:
                f[c] = (f[c] - p["mean"]) / p["std"]
        f["error_active"], f["non_welding"], f["error_code"] = error_active, non_welding.astype("float32"), code
        return f, error_active, non_welding

    def window_vector(self, f: pd.DataFrame) -> np.ndarray:
        w = f.iloc[-self.window:]
        stats = pd.concat([w[self.feat_cols].mean().add_suffix("_mean"), w[self.feat_cols].std().fillna(0).add_suffix("_std")])
        return stats.reindex(self.model_cols).to_numpy(dtype=np.float32)

    def score(self, vec: np.ndarray) -> tuple[float, list[FeatureContribution]]:
        n = len(vec)
        X = np.tile(vec, (n + 1, 1))
        X[np.arange(1, n + 1), np.arange(n)] = self.reference  # row j+1: feature j set to its reference
        s = trainlib.anomaly_score(self.model, X)
        contrib = s[0] - s[1:]
        pos = float(np.clip(contrib, 0, None).sum()) or 1.0
        order = np.argsort(-contrib)[:TOP_K_FEATURES]
        feats = []
        for j in order:
            name = self.model_cols[j]
            sensor, stat = name.rsplit("_", 1)
            feats.append(FeatureContribution(feature=name, sensor=sensor, sensor_name=SENSOR_NAME.get(sensor, sensor),
                                             statistic=stat, value=float(vec[j]), reference=float(self.reference[j]),
                                             contribution=float(contrib[j]), share=float(max(contrib[j], 0) / pos)))
        return float(s[0]), feats


class GunState:
    def __init__(self) -> None:
        self.rows: deque[dict[str, Any]] = deque(maxlen=BUFFER_S)
        self.lock = asyncio.Lock()
        self.consecutive_alarms = 0
        self.last_score: float | None = None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model not found: {MODEL_PATH} — run .py/train.py or set RSW_MODEL_PATH")
    app.state.detector = Detector(MODEL_PATH)
    app.state.guns: dict[str, GunState] = {}
    log.info("loaded %s (window %ss, threshold %.4f)", MODEL_PATH, app.state.detector.window, app.state.detector.threshold)
    yield


app = FastAPI(title="RSW gun anomaly detection", version="0.1.0", lifespan=lifespan,
              description="Real-time anomaly scoring of resistance-spot-welding gun sensor streams.")


# ============================================================ endpoints
@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    d: Detector = request.app.state.detector
    return {"status": "ok", "model_path": d.path, "model": d.info.model_dump(), "guns_tracked": len(request.app.state.guns)}


@app.get("/model")
async def model_card(request: Request) -> dict[str, Any]:
    d: Detector = request.app.state.detector
    b = d.bundle
    metrics = b.get("metrics", {})
    return {"model": d.info.model_dump(), "features": d.model_cols,
            "preprocess": {"gap_fill_limit_s": d.gap_fill_limit, "outlier_hi": d.outlier_hi, "outlier_lo": d.outlier_lo,
                           "rolling_s": ROLL_S, "scaled_columns": sorted(d.scaler)},
            "validation": {k: v for k, v in metrics.get("val", {}).items() if k != "per_file"},
            "test": {k: v for k, v in metrics.get("test", {}).items() if k != "per_file"} or None,
            "train_files": b.get("train_files"), "val_files": b.get("val_files"),
            "test_files": metrics.get("test", {}).get("files")}


@app.post("/predict", response_model=AnomalyResult | WarmingUp,
          responses={202: {"model": WarmingUp, "description": "not enough history yet"}})
async def predict(req: PredictRequest, request: Request) -> AnomalyResult | JSONResponse:
    d: Detector = request.app.state.detector
    guns: dict[str, GunState] = request.app.state.guns
    g = guns.setdefault(req.gun_id, GunState())
    async with g.lock:
        for r in req.readings:
            row = r.model_dump()
            row[BINARY_COL] = 1.0 if row[BINARY_COL] else 0.0
            g.rows.append(row)
        out = d.features(g.frame())
        if out is None or len(out[0]) < d.window:
            n = 0 if out is None else len(out[0])
            body = WarmingUp(gun_id=req.gun_id, n_samples=n, samples_needed=d.window,
                             message=f"need {d.window} contiguous 1 Hz samples, have {n}")
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=body.model_dump())
        f, error_active, non_welding = out
        vec = d.window_vector(f)
        score, feats = d.score(vec)
        is_anomaly = score > d.threshold
        g.consecutive_alarms = g.consecutive_alarms + 1 if is_anomaly else 0
        g.last_score = score
        sustained = g.consecutive_alarms >= d.sustain
        w = f.iloc[-d.window:]
        latest_code = str(w["error_code"].iloc[-1])
        ctx = WindowContext(latest_error_code=latest_code,
                            error_active_share=float(w["error_active"].mean()),
                            non_welding_share=float(w["non_welding"].mean()),
                            welds_in_window=float(w["welds_delta"].sum() * d.scaler.get("welds_delta", {}).get("std", 1.0)
                                                  + d.window * d.scaler.get("welds_delta", {}).get("mean", 0.0)),
                            welds_10min=float(w["welds_10min"].iloc[-1] * d.scaler.get("welds_10min", {}).get("std", 1.0)
                                              + d.scaler.get("welds_10min", {}).get("mean", 0.0)),
                            weld_duty_10min=float(w["weld_duty_10min"].iloc[-1]),
                            known_code_class_hint=CODE_TO_CLASS.get(latest_code))
        return AnomalyResult(
            gun_id=req.gun_id, window_start=w.index[0].to_pydatetime(), window_end=w.index[-1].to_pydatetime(),
            n_samples=int(len(w)), history_s=int(len(f)), is_anomaly=bool(is_anomaly), anomaly_score=score,
            threshold=d.threshold, score_z=(score - d.score_mean) / d.score_std,
            severity="critical" if sustained else "warning" if is_anomaly else "normal",
            sustained_alarm=sustained, consecutive_alarms=g.consecutive_alarms,
            contributing_features=feats, context=ctx, model=d.info)


@app.get("/guns", response_model=list[GunStatus])
async def guns(request: Request) -> list[GunStatus]:
    out = []
    for gid, g in request.app.state.guns.items():
        times = [r["time"] for r in g.rows]
        out.append(GunStatus(gun_id=gid, n_samples=len(g.rows), first_time=min(times) if times else None,
                             last_time=max(times) if times else None, consecutive_alarms=g.consecutive_alarms,
                             last_score=g.last_score))
    return out


@app.delete("/guns/{gun_id}", status_code=status.HTTP_204_NO_CONTENT)
async def forget_gun(gun_id: str, request: Request) -> None:
    if request.app.state.guns.pop(gun_id, None) is None:
        raise HTTPException(status_code=404, detail=f"unknown gun {gun_id}")
