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

Per-gun normalisation (bundle `gun_norm`, see train.py): during the first `warmup_s` of a gun's
stream the rows are scored with the global scaling and collected; at the end of the warm-up the
gun's mean (/std) over its normal welding rows and, optionally, its own threshold are fixed and
every later window is re-normalised with them. `AnomalyResult.gun_norm` says which regime a
result comes from. DELETE /guns/{id} restarts the warm-up (do it after maintenance).
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
# a request may not carry more rows than fit in the buffer next to 10 min of history: the rolling
# features of its first row need that history, and rows beyond the buffer would be dropped silently
MAX_CHUNK = BUFFER_S - ROLL_S
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
    c16: float = Field(description="Setpoint sheet thickness; <= 0 = non-welding operation (cap dressing); the bundle may add an upper bound")
    c17: float = Field(description="Setpoint velocity")
    c18: float = Field(description="Setpoint force build-up")
    c19: float = Field(description="Offset value in robot (a time counter - not a model input)")
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
    readings: list[SensorReading] = Field(
        min_length=1, max_length=MAX_CHUNK,
        description=f"new samples, any order; appended to the gun's buffer. At most {MAX_CHUNK} (the 30-min buffer minus "
                    "10 min of rolling history); every completed window in the chunk is scored")


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
    threshold: float = Field(description="global threshold (quantile of the training-normal scores)")
    threshold_q: float
    sustain: int
    n_features: int
    alarm_max_non_welding: float | None = Field(
        None, description="windows whose non_welding_share exceeds this never alarm (None = no gate)")
    gun_norm: Literal["none", "center", "scale"] = Field(
        "none", description="per-gun re-normalisation after the warm-up: center = subtract the gun mean, scale = also divide by its std")
    gun_warmup_s: int | None = Field(None, description="warm-up length per gun stream")
    gun_threshold_q: float | None = Field(None, description="quantile of the warm-up scores used as the gun's threshold (None = global)")
    rule_cooldown_s: int = Field(description="after a terminal-code rule trigger the rule stays quiet this long per gun")


class AnomalyResult(BaseModel):
    """Document handed to the ontology / root-cause stage."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "gun_id": "G17", "status": "ok", "window_start": "2021-09-05T02:22:21", "window_end": "2021-09-05T02:23:20",
        "n_samples": 60, "history_s": 1800, "is_anomaly": True, "anomaly_score": 0.61, "threshold": 0.54,
        "score_z": 4.6, "severity": "critical", "rule_triggered": True, "severity_source": "model+rule",
        "sustained_alarm": True, "consecutive_alarms": 3,
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
    is_anomaly: bool = Field(description="anomaly_score > threshold (raw model verdict, before the non-welding gate)")
    alarm_held: bool = Field(False, description="the window is mostly non-welding (share > model.alarm_max_non_welding): "
                                                "its sensors are carried-forward constants, so no alarm is raised or counted")
    hold_reason: str | None = Field(None, description="why alarm_held is true")
    anomaly_score: float = Field(description="higher = more anomalous")
    threshold: float = Field(description="threshold this window was judged against: the gun's own once it has one, else the global")
    gun_norm: Literal["warming_up", "gun", "global"] = Field(
        "global", description="warming_up: global scaling while the gun statistics are being collected; "
                              "gun: window re-normalised with this gun's warm-up statistics; global: no per-gun normalisation")
    gun_threshold: float | None = Field(None, description="this gun's threshold once its warm-up is over (None otherwise)")
    score_z: float = Field(description="(score - normal mean) / normal std on training windows")
    severity: Literal["normal", "warning", "critical"] = Field(
        description="normal: below threshold or held; warning: above; critical: sustained alarm (above threshold for "
                    ">= sustain x window seconds), OR the terminal-code rule fired in this request (rule_triggered)")
    rule_triggered: bool = Field(False, description="a terminal-code episode (E012/E016/E028/E029) STARTED in this request's "
                                                    "rows and the gun's rule cooldown had expired: severity is forced to "
                                                    "critical regardless of the model. Fires once per episode, not while "
                                                    "the code persists")
    rule_trigger_time: datetime | None = Field(None, description="timestamp of the row that fired the rule")
    rule_code: str | None = Field(None, description="the terminal code that fired the rule (it may have cleared by window_end)")
    rule_class_hint: FaultClass | None = Field(None, description="fault class of rule_code - use this, not "
                                                                 "context.known_code_class_hint, when rule_triggered")
    rule_code_active: bool = Field(False, description="the latest error code is a terminal code (a state, not a trigger)")
    severity_source: Literal["model", "rule", "model+rule", "none"] = Field(
        "none", description="what made severity critical: the model's sustained alarm, the terminal-code rule, or both")
    sustained_in_request: bool = Field(False, description="some window of this request had a sustained model alarm")
    sustained_alarm: bool = Field(description="the model has alarmed continuously for >= sustain x window seconds "
                                              "(time-based: independent of how often the client calls)")
    consecutive_alarms: int = Field(description="consecutive alarming windows scored (they overlap when calls are < window apart)")
    alarm_duration_s: int = Field(0, description="seconds of continuous alarm up to window_end (0 = no alarm)")
    windows_scored: int = Field(1, description="windows scored for this request: every completed window since the previous "
                                               "request (spaced window_s apart, ending at the latest row)")
    critical_in_request: bool = Field(False, description="some window of this request was critical (a sustained alarm "
                                                         "or a rule trigger) - even if the latest window is not")
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
    gun_norm: Literal["warming_up", "gun", "global"] = "global"
    gun_threshold: float | None = None
    warmup_rows: int = Field(0, description="rows collected for the gun statistics so far")


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
        self.outlier_hi: float | None = None if cfg["outlier_hi"] is None else float(cfg["outlier_hi"])
        self.outlier_lo: float = float(cfg["outlier_lo"])
        g = b.get("alarm_max_non_welding")
        self.alarm_max_non_welding: float | None = None if g is None else float(g)
        gn = b.get("gun_norm")
        self.gun_norm: dict[str, Any] | None = dict(gn) if gn else None
        self.norm_cols: list[str] = [c for c in (gn or {}).get("columns", []) if c in self.feat_cols]
        if self.gun_norm and not self.norm_cols:
            raise RuntimeError("bundle gun_norm has no usable columns")
        tr = b.get("metrics", {}).get("train", {})
        self.score_mean: float = float(tr.get("score_mean", 0.0))
        self.score_std: float = float(tr.get("score_std", 1.0)) or 1.0
        ref = b.get("feature_reference")
        self.reference = np.asarray(ref, dtype=np.float32) if ref else np.zeros(len(self.model_cols), np.float32)
        if not ref:
            log.warning("bundle has no feature_reference (retrain with the current train.py); using zeros")
        rule = b.get("rule") or {"codes": list(trainlib.TERMINAL_CODES), "cooldown_s": trainlib.RULE_COOLDOWN_S}
        self.rule_codes: list[str] = list(rule["codes"])
        self.rule_cooldown_s: int = int(rule["cooldown_s"])
        self.info = ModelInfo(name=os.path.splitext(os.path.basename(path))[0], type=str(b.get("model_type", "?")),
                              created=str(b.get("created", "?")), window_s=self.window, threshold=self.threshold,
                              threshold_q=float(b.get("threshold_q", float("nan"))), sustain=self.sustain,
                              n_features=len(self.model_cols), alarm_max_non_welding=self.alarm_max_non_welding,
                              gun_norm=(gn or {}).get("mode", "none"), gun_warmup_s=(gn or {}).get("warmup_s"),
                              gun_threshold_q=(gn or {}).get("threshold_q"), rule_cooldown_s=self.rule_cooldown_s)

    # ---- per-gun normalisation, the online counterpart of train.window_file()
    def gun_normalise(self, g: "GunState", f: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Collect warm-up rows / fix the gun statistics when the warm-up ends / re-normalise the rows after it.
        Returns (features to score, threshold to judge against)."""
        gn = self.gun_norm
        if gn is None:
            return f, self.threshold
        if g.t0 is None:
            g.t0 = f.index[0]
        t_end = g.t0 + pd.Timedelta(seconds=int(gn["warmup_s"]))
        if g.norm_status == "warming_up":
            new = f[(f.index < t_end) & ((f.index > g.last_warm_time) if g.last_warm_time is not None else True)]
            if len(new):
                g.warm_parts.append(new[self.feat_cols + ["non_welding", "error_active"]])
                g.last_warm_time, g.warmup_rows = new.index[-1], g.warmup_rows + len(new)
            if f.index[-1] >= t_end:
                self._finish_warmup(g, t_end)
        if g.norm_status != "gun":
            return f, self.threshold
        f = f.copy()
        sel = f.index >= g.norm["t_end"]
        if sel.any():
            f.loc[sel, self.norm_cols] = ((f.loc[sel, self.norm_cols].to_numpy(dtype="float64") - g.norm["mean"])
                                          / g.norm["std"]).astype("float32")
        return f, g.gun_threshold if g.gun_threshold is not None else self.threshold

    def _finish_warmup(self, g: "GunState", t_end: pd.Timestamp) -> None:
        gn = self.gun_norm
        warm = pd.concat(g.warm_parts) if g.warm_parts else pd.DataFrame(columns=self.feat_cols + ["non_welding", "error_active"])
        g.warm_parts.clear()
        ok = (warm["non_welding"].to_numpy() == 0) & (warm["error_active"].to_numpy() == 0)
        if int(ok.sum()) < int(gn["min_rows"]):
            g.norm_status = "global"
            log.warning("gun warm-up ended with %d normal welding rows (< %d): keeping the global scaling", int(ok.sum()), gn["min_rows"])
            return
        x = warm.loc[ok, self.norm_cols].astype("float64")
        mean = x.mean().to_numpy()
        std = np.maximum(x.std(ddof=0).fillna(0.0).to_numpy(), float(gn["std_floor"])) if gn["mode"] == "scale" \
            else np.ones(len(self.norm_cols))
        g.norm = {"mean": mean, "std": std, "t_end": t_end}
        q = gn.get("threshold_q")
        if q is not None:
            # the warm-up windows, re-normalised with the gun statistics, calibrate the gun's threshold
            # (minute-aligned windows per contiguous segment, as train.py does offline)
            wf = warm.copy()
            wf[self.norm_cols] = ((wf[self.norm_cols].to_numpy(dtype="float64") - mean) / std).astype("float32")
            step = wf.index.to_series().diff().dt.total_seconds().fillna(1)
            wf["segment_id"] = (step > 1).cumsum()
            wf["ttf_s"], wf["label"] = 0.0, 0.0
            wf["file"] = wf["class"] = wf["gun"] = "online"
            w = trainlib.window_features(wf, self.window, self.feat_cols)
            if len(w) >= 10:
                s = trainlib.anomaly_score(self.model, w[self.model_cols].to_numpy(dtype=np.float32))
                g.gun_threshold = float(max(self.threshold, np.quantile(s, float(q))))
        g.norm_status = "gun"

    def window_threshold(self, g: "GunState", window_start: pd.Timestamp) -> float:
        """The gun's own threshold for windows entirely after its warm-up, the global one otherwise
        (train.window_thresholds: warm-up windows are judged with the global threshold)."""
        if g.norm_status == "gun" and g.gun_threshold is not None and window_start >= g.norm["t_end"]:
            return g.gun_threshold
        return self.threshold

    # ---- terminal-code rule: fires on an episode start, then keeps quiet for rule_cooldown_s
    def rule_check(self, g: "GunState", code: pd.Series) -> tuple[pd.Timestamp, str] | None:
        """Scan the rows not seen yet; return (time, code) of the first trigger, None if the rule did not fire."""
        new = code[code.index > g.rule_seen] if g.rule_seen is not None else code
        if new.empty:
            return None
        prev = new.shift(fill_value=g.rule_prev_code)
        onset = new.isin(self.rule_codes) & (new != prev)
        fired = None
        for t in new.index[onset.to_numpy()]:
            if g.rule_last_trigger is None or (t - g.rule_last_trigger).total_seconds() >= self.rule_cooldown_s:
                g.rule_last_trigger = t
                fired = fired or (t, str(new.loc[t]))
        g.rule_seen, g.rule_prev_code = new.index[-1], str(new.iloc[-1])
        return fired

    def window_ends(self, g: "GunState", f: pd.DataFrame) -> list[int]:
        """Row positions of the window ends to score, chronological: the latest row, then every `window`
        rows back, down to the previous request's last window (same segment) or the segment start."""
        last_end = g.last_end if g.last_end is not None and g.last_end >= f.index[0] else None
        ends, pos = [], len(f) - 1
        while pos >= self.window - 1 and (last_end is None or f.index[pos] > last_end):
            ends.append(pos)
            pos -= self.window
        return ends[::-1] or [len(f) - 1]  # nothing new (a resend): re-score the latest window

    # ---- preprocessing identical in spirit to preprocess.py, on a rolling buffer
    def features(self, raw: pd.DataFrame, carry: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series | None] | None:
        """Buffer rows -> (feature frame of the last contiguous segment, sensor values of the latest welding row).
        `carry` = the latest welding row seen by an earlier request: non-welding rows take the last welding value
        however long ago it was (preprocess.py), and a cap-dressing block can outlast the 30-min buffer.
        None when nothing can be filled yet (the stream has not shown a single welding row)."""
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
        non_welding = x["c16"] <= self.outlier_lo
        if self.outlier_hi is not None:
            non_welding |= x["c16"] > self.outlier_hi
        held_cols = VALUE_COLS + [BINARY_COL]
        welding = ~non_welding
        carry = x.loc[welding, held_cols].iloc[-1] if welding.any() else carry
        x.loc[non_welding, held_cols] = np.nan
        if carry is not None and non_welding.iloc[0]:
            x.iloc[0, x.columns.get_indexer(held_cols)] = carry[held_cols].to_numpy()
        x = x.ffill().bfill()
        if x[held_cols].isna().any().any():
            return None

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
        return f, carry

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
    def __init__(self, gun_norm: bool = False) -> None:
        self.rows: deque[dict[str, Any]] = deque(maxlen=BUFFER_S)
        self.lock = asyncio.Lock()
        self.consecutive_alarms = 0
        self.last_score: float | None = None
        # time-based sustained alarm: start of the current alarm run, end of the last scored window
        self.alarm_since: pd.Timestamp | None = None
        self.last_end: pd.Timestamp | None = None
        # terminal-code rule: last row seen, its code, time of the last trigger (cooldown)
        self.rule_seen: pd.Timestamp | None = None
        self.rule_prev_code = "0"
        self.rule_last_trigger: pd.Timestamp | None = None
        # sensor values of the latest welding row: what non-welding rows carry once it has left the buffer
        self.carry: pd.Series | None = None
        # per-gun normalisation: rows collected during the warm-up, then the fixed statistics
        self.norm_status: str = "warming_up" if gun_norm else "global"
        self.t0: pd.Timestamp | None = None
        self.last_warm_time: pd.Timestamp | None = None
        self.warm_parts: list[pd.DataFrame] = []
        self.warmup_rows = 0
        self.norm: dict[str, Any] | None = None
        self.gun_threshold: float | None = None

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
                           "alarm_max_non_welding": d.alarm_max_non_welding,
                           "rolling_s": ROLL_S, "scaled_columns": sorted(d.scaler)},
            "gun_norm": d.gun_norm, "dropped_features": b.get("dropped_features"),
            "validation": {k: v for k, v in metrics.get("val", {}).items() if k != "per_file"},
            "test": {k: v for k, v in metrics.get("test", {}).items() if k != "per_file"} or None,
            "train_files": b.get("train_files"), "val_files": b.get("val_files"),
            "test_files": metrics.get("test", {}).get("files")}


@app.post("/predict", response_model=AnomalyResult | WarmingUp,
          responses={202: {"model": WarmingUp, "description": "not enough history yet"}})
async def predict(req: PredictRequest, request: Request) -> AnomalyResult | JSONResponse:
    d: Detector = request.app.state.detector
    guns: dict[str, GunState] = request.app.state.guns
    g = guns.setdefault(req.gun_id, GunState(gun_norm=d.gun_norm is not None))
    async with g.lock:
        for r in req.readings:
            row = r.model_dump()
            row[BINARY_COL] = 1.0 if row[BINARY_COL] else 0.0
            g.rows.append(row)
        out = d.features(g.frame(), g.carry)
        if out is not None:
            g.carry = out[1]
        rule = d.rule_check(g, out[0]["error_code"]) if out is not None else None
        if out is None or len(out[0]) < d.window:
            n = 0 if out is None else len(out[0])
            body = WarmingUp(gun_id=req.gun_id, n_samples=n, samples_needed=d.window,
                             message=f"need {d.window} contiguous 1 Hz samples, have {n}" if out is not None else
                             "no welding row seen yet: non-welding rows (c16 <= 0) carry the last welding values")
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=body.model_dump())
        f, _ = d.gun_normalise(g, out[0])
        ends = d.window_ends(g, f)
        if g.last_end is None or g.last_end < f.index[0]:  # first request or after a long gap: a new alarm run
            g.alarm_since, g.consecutive_alarms = None, 0
        fresh = g.last_end is None or f.index[ends[-1]] > g.last_end
        vecs = np.stack([d.window_vector(f.iloc[: e + 1]) for e in ends])
        scores = trainlib.anomaly_score(d.model, vecs)
        sustained_in_request = False
        for e, sc in zip(ends, scores):  # chronological: advance the alarm state window by window
            w = f.iloc[e - d.window + 1: e + 1]
            threshold = d.window_threshold(g, w.index[0])
            nw_share = float(w["non_welding"].mean())
            is_anomaly = bool(sc > threshold)
            held = d.alarm_max_non_welding is not None and nw_share > d.alarm_max_non_welding
            alarm = is_anomaly and not held
            if fresh:
                g.consecutive_alarms = g.consecutive_alarms + 1 if alarm else 0
                g.alarm_since = (g.alarm_since or w.index[0]) if alarm else None
                g.last_end = w.index[-1]
            duration = int((w.index[-1] - g.alarm_since).total_seconds()) + 1 if alarm and g.alarm_since is not None else 0
            sustained = duration >= d.sustain * d.window
            sustained_in_request |= sustained
        score, feats = d.score(vecs[-1])  # the latest window, with feature attributions
        g.last_score = score
        latest_code = str(w["error_code"].iloc[-1])
        # terminal-code rule (history.md P2-1 / MVP review): the model scores the terminal state below threshold, so
        # the rule forces critical - once per episode start, then quiet for rule_cooldown_s
        rule_hit = rule is not None
        severity = "critical" if (sustained or rule_hit) else "warning" if alarm else "normal"
        severity_source = ("model+rule" if sustained and rule_hit else "model" if sustained else "rule" if rule_hit else "none")
        ctx = WindowContext(latest_error_code=latest_code,
                            error_active_share=float(w["error_active"].mean()),
                            non_welding_share=nw_share,
                            welds_in_window=float(w["welds_delta"].sum() * d.scaler.get("welds_delta", {}).get("std", 1.0)
                                                  + d.window * d.scaler.get("welds_delta", {}).get("mean", 0.0)),
                            welds_10min=float(w["welds_10min"].iloc[-1] * d.scaler.get("welds_10min", {}).get("std", 1.0)
                                              + d.scaler.get("welds_10min", {}).get("mean", 0.0)),
                            weld_duty_10min=float(w["weld_duty_10min"].iloc[-1]),
                            known_code_class_hint=CODE_TO_CLASS.get(latest_code))
        return AnomalyResult(
            gun_id=req.gun_id, window_start=w.index[0].to_pydatetime(), window_end=w.index[-1].to_pydatetime(),
            n_samples=int(len(w)), history_s=int(len(f)), is_anomaly=is_anomaly, anomaly_score=score,
            alarm_held=bool(held),
            hold_reason=f"non_welding_share {nw_share:.2f} > gate {d.alarm_max_non_welding}" if held else None,
            threshold=threshold, gun_norm=g.norm_status, gun_threshold=g.gun_threshold,
            score_z=(score - d.score_mean) / d.score_std,
            severity=severity, rule_triggered=rule_hit,
            rule_trigger_time=rule[0].to_pydatetime() if rule_hit else None,
            rule_code=rule[1] if rule_hit else None, rule_class_hint=CODE_TO_CLASS.get(rule[1]) if rule_hit else None,
            rule_code_active=latest_code in d.rule_codes, severity_source=severity_source,
            sustained_alarm=sustained, consecutive_alarms=g.consecutive_alarms, alarm_duration_s=duration,
            windows_scored=len(ends), sustained_in_request=sustained_in_request,
            critical_in_request=sustained_in_request or rule_hit,
            contributing_features=feats, context=ctx, model=d.info)


@app.get("/guns", response_model=list[GunStatus])
async def guns(request: Request) -> list[GunStatus]:
    out = []
    for gid, g in request.app.state.guns.items():
        times = [r["time"] for r in g.rows]
        out.append(GunStatus(gun_id=gid, n_samples=len(g.rows), first_time=min(times) if times else None,
                             last_time=max(times) if times else None, consecutive_alarms=g.consecutive_alarms,
                             last_score=g.last_score, gun_norm=g.norm_status, gun_threshold=g.gun_threshold,
                             warmup_rows=g.warmup_rows))
    return out


@app.delete("/guns/{gun_id}", status_code=status.HTTP_204_NO_CONTENT)
async def forget_gun(gun_id: str, request: Request) -> None:
    if request.app.state.guns.pop(gun_id, None) is None:
        raise HTTPException(status_code=404, detail=f"unknown gun {gun_id}")
