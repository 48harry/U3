"""
End-to-end smoke test of the pipeline on synthetic data:

    CSV (train + test) -> preprocess.py -> preprocess.py --split test -> train.py -> train.py --evaluate
        -> main.py (FastAPI) /predict on the raw test CSV rows

It checks that every stage accepts the previous stage's output and that the API reproduces the
offline preprocessing (same score for the same window). It runs in ~30 s and touches only a
temporary directory - never train/, preprocessed/ or models/.
"""
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".py")
sys.path.insert(0, PY)

CLASSES = ["E01", "E02", "E03", "E04"]
TERMINAL = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
HOURS = 2  # per synthetic file
PRE_FAILURE_S = 600  # short pre-failure window so 2 h files still hold plenty of normal windows


def synth_file(path, cls, seed, start="2021-09-01T00:00:00Z"):
    """1 Hz file that ends at its failure: gap of 90 s (splits a segment), a cap-dressing block
    (c16 = 0), drifting balance pressure in the last 10 min and the class terminal code at the end."""
    rng = np.random.default_rng(seed)
    n = HOURS * 3600
    t = pd.date_range(start, periods=n, freq="s")
    ttf = np.arange(n)[::-1]
    df = pd.DataFrame({"time": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
    df["c1"] = rng.normal(2, 0.5, n).round(2)
    df["c2"] = np.where(rng.random(n) < 0.3, rng.normal(2400, 50, n), 0).round(0)
    df["c3"] = rng.normal(140, 2, n).round(1)
    df["c4"] = (rng.random(n) < 0.3).astype(int)
    df["c5"] = rng.normal(480, 3, n) + np.where(ttf < PRE_FAILURE_S, 40 * (1 - ttf / PRE_FAILURE_S), 0)
    df["c6"] = rng.normal(-65, 3, n).round(0)
    df["c7"], df["c8"], df["c9"] = 222.3, 4309, 132  # gun constants
    df["c10"] = np.where(rng.random(n) < 0.9, "on", "off")
    df["c11"] = np.cumsum(rng.random(n) < 0.3)
    df["c12"] = np.cumsum(rng.random(n) < 0.5)
    df["c13"] = 14.99
    df["c14"] = 2200
    df["c15"] = 150
    df["c16"] = 1.6
    df["c17"] = 1
    df["c18"] = (rng.random(n) < 0.3).astype(int)
    df["c19"] = rng.normal(1.02e9, 5e4, n).round(0)
    df["error"] = "0"
    df.loc[(ttf < 300), "error"] = TERMINAL[cls]  # terminal code in the last 5 min
    df.loc[3000:3299, "error"] = "E003"  # a long-lived minor state in the middle
    # cap dressing block -> non_welding. Deliberately starts mid-minute (t+1530 s) so that one
    # 60 s window is half covered: that is what distinguishes the "mean" aggregation from "max".
    df.loc[1530:1799, "c16"] = 0.0
    df = df.drop(index=range(2400, 2490))  # 90 s gap -> new segment
    df = df.drop(index=[100, 101, 500])  # short gaps -> ffill
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("rsw")
    train_dir, test_dir = ws / "train", ws / "test"
    train_dir.mkdir(), test_dir.mkdir()
    seed = 0
    for cls in CLASSES:
        for gun in range(2):
            synth_file(train_dir / f"{cls}_{gun}.csv", cls, seed, start=f"2021-09-{1 + seed:02d}T00:00:00Z")
            seed += 1
    synth_file(test_dir / "test_0.csv", "E04", 99, start="2021-10-01T00:00:00Z")
    synth_file(test_dir / "test_1.csv", "E02", 98, start="2021-10-02T00:00:00Z")
    return {"ws": ws, "train": train_dir, "test": test_dir, "pre": ws / "preprocessed", "models": ws / "models"}


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"{' '.join(map(str, args))}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}"
    return r.stdout


# ------------------------------------------------------------------ stages
def test_1_preprocess_train(workspace):
    out = run(os.path.join(PY, "preprocess.py"), "--input-dir", workspace["train"], "--out-dir", workspace["pre"],
              "--pre-failure-window", str(PRE_FAILURE_S))
    assert "files kept 8/8" in out
    pre = workspace["pre"]
    assert (pre / "scaler.json").exists() and (pre / "preprocess_config.json").exists()
    df = pd.read_parquet(pre / "E01_0.parquet")
    assert df["segment_id"].nunique() == 2, "the 90 s gap must split the file into two segments"
    assert df["non_welding"].sum() == 270
    assert df["label"].sum() == PRE_FAILURE_S + 1 and df["terminal_code"].sum() == 300
    assert "c11" not in df.columns and "c12" not in df.columns, "counters are replaced by deltas"
    assert {"welds_delta", "pos_delta", "welds_10min", "error_share_10min"} <= set(df.columns)
    normal = df[(df["label"] == 0) & (df["error_active"] == 0)]
    assert abs(normal["c5"].mean()) < 0.2, "z-score is fitted on normal rows of all files"


def test_2_preprocess_test_split(workspace):
    out = run(os.path.join(PY, "preprocess.py"), "--split", "test", "--input-dir", workspace["test"],
              "--train-out-dir", workspace["pre"])
    assert "files kept 2/2" in out
    m = pd.read_csv(workspace["pre"] / "test" / "manifest.csv")
    assert dict(zip(m["file"], m["class"])) == {"test_0": "E04", "test_1": "E02"}
    assert (m["class_source"] == "terminal code E029").iloc[0]
    cfg = json.load(open(workspace["pre"] / "test" / "preprocess_config.json", encoding="utf-8"))
    assert cfg["pre_failure_window"] == PRE_FAILURE_S, "unset options are inherited from the train run"
    assert not (workspace["pre"] / "test" / "scaler.json").exists(), "the test split never writes a scaler"
    tr, te = pd.read_parquet(workspace["pre"] / "E04_0.parquet"), pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
    assert list(tr.columns) == list(te.columns)
    # same scaler as train: a constant sensor maps to the same z-value in both splits
    assert np.isclose(tr["c13"].iloc[0], te["c13"].iloc[0])


def test_2b_window_columns(workspace):
    """Windowing must give the 24 features a _mean/_std pair and leave metadata names alone.
    Regression: `non_welding` is aggregated with "mean" and used to become `non_welding_mean`,
    which broke --exclude-non-welding and dropped the column from the score CSVs."""
    from train import feature_columns, model_input_columns, window_features

    df = pd.read_parquet(workspace["pre"] / "E01_0.parquet")
    feat = feature_columns(df)
    w = window_features(df, 60, feat)
    assert len(feat) == 24 and len(model_input_columns(feat)) == 48
    assert set(model_input_columns(feat)) <= set(w.columns)
    for c in ("non_welding", "error_active", "terminal_code", "label", "ttf_s", "n"):
        assert c in w.columns, f"{c} must keep its own name"
        assert f"{c}_mean" not in w.columns
    nw = w["non_welding"]
    assert nw.min() == 0.0 and nw.max() == 1.0, "non_welding is a share over the window"
    assert (nw == 0.5).sum() == 1, "the half-covered window must be 0.5, i.e. a mean and not a max"
    assert (nw == 1.0).sum() == 4
    assert w["n"].max() == 60 and w["n"].min() >= 30


def test_3_train_and_evaluate(workspace):
    # 30 min warm-up so that the 2 h synthetic files exercise the per-gun normalisation + threshold
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--model-dir", workspace["models"],
              "--val-frac", "0.5", "--window", "60", "--warmup-hours", "0.5")
    assert "val: AUROC" in out and "test: AUROC" in out and "gun-normalised" in out
    metrics = json.load(open(workspace["models"] / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert metrics["train_files"] and metrics["val_files"]
    assert not set(metrics["train_files"]) & set(metrics["val_files"]), "split is by file"
    assert metrics["metrics"]["test"]["files"] == ["test_0.parquet", "test_1.parquet"]
    assert metrics["metrics"]["test"]["file_class"] == {"test_0": "E04", "test_1": "E02"}
    assert len(metrics["feature_reference"]) == len(metrics["model_cols"])
    assert metrics["preprocess_config"]["gap_fill_limit"] == 60
    assert metrics["preprocess_config"]["outlier_hi"] is None, "c16 > x rule is off by default"
    assert metrics["alarm_max_non_welding"] == 0.5 and "held_windows" in metrics["metrics"]["val"]
    assert metrics["dropped_features"] == ["c19"] and len(metrics["model_cols"]) == 46, "c19 is a time counter (leak)"
    gn = metrics["gun_norm"]
    assert gn["mode"] == "center" and gn["warmup_s"] == 1800 and gn["threshold_q"] == 0.99
    assert "c7" in gn["columns"] and "hour_sin" not in gn["columns"] and "weld_duty_10min" not in gn["columns"]
    assert metrics["metrics"]["test"]["files_gun_normalised"] == 2 and metrics["metrics"]["test"]["files_gun_threshold"] == 2
    for v in metrics["metrics"]["test"]["per_file"].values():
        assert v["gun_norm"] == "gun" and v["threshold"] >= metrics["threshold"]
    assert metrics["metrics"]["val"]["auroc"] > 0.5, "the drifting c5 in the pre-failure window must be detectable"
    # terminal-code rule, evaluated as main.py fires it: one trigger per file (the code starts 300 s before failure)
    rule = metrics["metrics"]["test"]["rule_terminal_code"]
    assert rule["flag"] == "terminal_any" and rule["n_triggers"] == 2 and rule["files_hit"] == 2
    assert rule["files_with_false_trigger"] == 0 and 4 <= rule["rule_lead_min_median"] <= 5
    assert metrics["rule"] == {"codes": ["E012", "E016", "E028", "E029"], "cooldown_s": 1800}

    out = run(os.path.join(PY, "train.py"), "--evaluate", "--data-dir", workspace["pre"], "--model-dir", workspace["models"])
    assert "saved" in out
    tm = json.load(open(workspace["models"] / "baseline_iforest_test_metrics.json", encoding="utf-8"))
    assert tm["test"]["auroc"] == pytest.approx(metrics["metrics"]["test"]["auroc"])
    scores = pd.read_csv(workspace["models"] / "scores" / "test_0_iforest.csv", index_col=0, parse_dates=True)
    assert {"score", "alarm", "threshold", "ttf_s", "label", "error_active", "non_welding", "warmup"} <= set(scores.columns)
    assert scores["warmup"].iloc[0] == 1 and scores["warmup"].iloc[-1] == 0
    assert np.allclose(scores.loc[scores["warmup"] == 1, "threshold"], metrics["threshold"])
    assert (scores.loc[scores["warmup"] == 0, "threshold"] >= metrics["threshold"] - 1e-9).all()

    # without gun normalisation the bundle carries gun_norm = None and plain global thresholds
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5", "--gun-norm", "none",
              "--no-test", "--model-dir", workspace["ws"] / "models_global")
    m2 = json.load(open(workspace["ws"] / "models_global" / "baseline_iforest_metrics.json", encoding="utf-8"))
    assert m2["gun_norm"] is None and m2["metrics"]["val"]["files_gun_normalised"] == 0

    # --exclude-non-welding must run (it used to raise KeyError: 'non_welding')
    out = run(os.path.join(PY, "train.py"), "--data-dir", workspace["pre"], "--val-frac", "0.5",
              "--exclude-non-welding", "--no-test", "--model-dir", workspace["ws"] / "models_xnw")
    assert "fitting iforest" in out and "val: AUROC" in out

    out = run(os.path.join(PY, "train.py"), "--score", str(workspace["pre"] / "test" / "test_1.parquet"),
              "--model-dir", workspace["models"])
    assert "windows, alarm rate" in out


def test_3b_rule_triggers():
    """The rule fires on an episode start (0 -> 1) and then keeps quiet for the cooldown."""
    from train import rule_triggers

    flags = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    ttf = np.arange(len(flags))[::-1] * 60.0
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=1800)).tolist() == [1]
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=60)).tolist() == [1, 4, 7]
    assert np.flatnonzero(rule_triggers(flags, ttf, cooldown_s=240)).tolist() == [1, 7]


def test_4_api_matches_offline(workspace):
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / "baseline_iforest.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})
    with TestClient(main.app) as client:
        card = client.get("/model").json()
        assert card["model"]["n_features"] == 46 and card["test"]["auroc"] is not None
        assert card["dropped_features"] == ["c19"] and "c19_mean" not in card["features"]
        assert card["preprocess"]["gap_fill_limit_s"] == 60
        assert card["model"]["gun_norm"] == "center" and card["model"]["gun_warmup_s"] == 1800
        assert card["gun_norm"]["threshold_q"] == 0.99

        # first 30 s -> warming up
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[:30].to_dict("records")})
        assert r.status_code == 202 and r.json()["status"] == "warming_up"
        # 20 minutes of history -> scored; window = the trailing 60 s
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[30:1200].to_dict("records")})
        assert r.status_code == 200, r.text
        res = r.json()
        for k in ("is_anomaly", "anomaly_score", "threshold", "score_z", "severity", "sustained_alarm",
                  "consecutive_alarms", "contributing_features", "context", "model"):
            assert k in res
        assert res["n_samples"] == 60 and res["severity"] in ("normal", "warning", "critical")
        assert res["context"]["known_code_class_hint"] is None
        assert len(res["contributing_features"]) == 8 and res["contributing_features"][0]["reference"] != 0
        assert res["alarm_held"] is False and res["hold_reason"] is None
        assert card["model"]["alarm_max_non_welding"] == 0.5
        assert res["gun_norm"] == "warming_up" and res["gun_threshold"] is None, "20 min < 30 min warm-up"
        assert res["threshold"] == pytest.approx(card["model"]["threshold"])

        # the API's online preprocessing must reproduce the offline pipeline on the same window
        d = main.app.state.detector
        off = pd.read_parquet(workspace["pre"] / "test" / "test_0.parquet")
        w_end = pd.Timestamp(res["window_end"])
        assert pd.Timestamp(res["window_start"]) == w_end - pd.Timedelta(seconds=59)
        vec_off = d.window_vector(off.loc[:w_end])
        score_off = float(main.trainlib.anomaly_score(d.model, vec_off[None])[0])
        assert res["anomaly_score"] == pytest.approx(score_off, abs=1e-4)

        # past the warm-up: the gun statistics are fixed, later windows are re-normalised and judged
        # against the gun's own threshold - exactly as train.window_file() / gun_thresholds() do offline
        # (up to the 90 s gap: raw row 2390 ~ t = 2393 s > the 1800 s warm-up)
        r = client.post("/predict", json={"gun_id": "G1", "readings": raw.iloc[1200:2390].to_dict("records")})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["gun_norm"] == "gun" and res["gun_threshold"] is not None
        assert res["threshold"] == pytest.approx(res["gun_threshold"]) and res["threshold"] >= card["model"]["threshold"]
        gs = {g["gun_id"]: g for g in client.get("/guns").json()}["G1"]
        assert gs["gun_norm"] == "gun" and gs["warmup_rows"] >= 1700
        gn, cols = d.gun_norm, d.norm_cols
        stats = main.trainlib.gun_norm_stats(off, cols, gn)
        assert stats is not None and stats["n"] >= 1000
        off_n = main.trainlib.apply_gun_norm(off, cols, stats)
        w_end = pd.Timestamp(res["window_end"])
        vec_off = d.window_vector(off_n.loc[:w_end])
        score_off = float(main.trainlib.anomaly_score(d.model, vec_off[None])[0])
        assert res["anomaly_score"] == pytest.approx(score_off, abs=1e-3)
        w_off, calib = main.trainlib.window_file(off, 60, d.feat_cols, gn, cols)
        thr_off = main.trainlib.gun_thresholds(d.model, d.model_cols, calib, d.threshold, gn["threshold_q"])["test_0"]
        assert res["gun_threshold"] == pytest.approx(thr_off, abs=2e-3)
        # gun constants (c7-c9) are centred away: their window means are 0 after the warm-up
        assert all(c["value"] == 0 for c in res["contributing_features"] if c["sensor"] in ("c7", "c8", "c9"))

        # the failure end: terminal code -> class hint for the ontology stage
        r = client.post("/predict", json={"gun_id": "G2", "readings": raw.iloc[-1500:-750].to_dict("records")})
        assert r.status_code == 200 and r.json()["rule_triggered"] is False
        r = client.post("/predict", json={"gun_id": "G2", "readings": raw.iloc[-750:].to_dict("records")})
        assert r.status_code == 200 and r.json()["context"]["known_code_class_hint"] == "E04"
        assert r.json()["context"]["latest_error_code"] == "E029"
        # terminal-code rule: severity is critical whatever the model says
        assert r.json()["rule_triggered"] is True and r.json()["severity"] == "critical"
        assert r.json()["severity_source"] in ("rule", "model+rule") and r.json()["critical_in_request"] is True
        assert pd.Timestamp(r.json()["rule_trigger_time"]) == pd.Timestamp(raw["time"].iloc[-300]).tz_localize(None)
        assert r.json()["rule_code"] == "E029" and r.json()["rule_class_hint"] == "E04"
        # ML -> RAG handoff rides on the rule trigger and lands in the outbox (pull; no RSW_RAG_URL in tests)
        ho = r.json()["handoff"]
        assert ho is not None and ho["schema_version"] == "1.0" and ho["gun_id"] == "G2"
        assert ho["fault_class"]["code"] == "E04" and ho["fault_class"]["basis"] == "rule_trigger"
        assert ho["trigger"]["source"] in ("rule", "model+rule") and ho["trigger"]["rule_code"] == "E029"
        assert ho["situation_ids"][0] == "S04" and "E04" in ho["summary_ko"]
        assert "hypotheses" not in ho and "search" not in ho, "causes / manuals are the ontology's job"
        recs = client.get("/handoffs", params={"gun_id": "G2"}).json()
        assert recs[-1]["handoff"]["event_id"] == ho["event_id"] and recs[-1]["delivery"] == "pull_only"
        assert client.get(f"/handoffs/{ho['event_id']}").status_code == 200
        assert client.get("/handoffs/nope").status_code == 404
        prev = client.post("/handoffs/preview", json={k: v for k, v in r.json().items() if k != "handoff"})
        assert prev.status_code == 200 and prev.json()["fault_class"]["code"] == "E04"
        assert prev.json()["situation_ids"] == ho["situation_ids"]
        assert r.json()["windows_scored"] == 13, "a window every 60 s back from the latest row: ceil(750 / 60)"
        assert res["rule_triggered"] is False and res["severity_source"] in ("none", "model")
        assert res["handoff"] is None or res["critical_in_request"], "handoffs only on critical events"

        # a window inside the cap-dressing block: alarms are held, never counted, severity stays normal
        r = client.post("/predict", json={"gun_id": "G3", "readings": raw.iloc[1400:1700].to_dict("records")})
        assert r.status_code == 200, r.text
        held = r.json()
        assert held["context"]["non_welding_share"] == 1.0 and held["alarm_held"] is True
        assert held["hold_reason"].startswith("non_welding_share 1.00 > gate 0.5")
        assert held["severity"] == "normal" and held["consecutive_alarms"] == 0

        assert {g["gun_id"] for g in client.get("/guns").json()} == {"G1", "G2", "G3"}
        assert client.delete("/guns/G1").status_code == 204
        assert client.delete("/guns/G1").status_code == 404


def test_5_api_chunks_sustain_rule(workspace):
    """Chunk limit, time-based sustained alarm, and the edge-triggered terminal-code rule with cooldown."""
    os.environ["RSW_MODEL_PATH"] = str(workspace["models"] / "baseline_iforest.joblib")
    sys.path.insert(0, ROOT)
    main = importlib.reload(importlib.import_module("main"))
    from fastapi.testclient import TestClient

    raw = pd.read_csv(workspace["test"] / "test_0.csv", dtype={"c10": str, "error": str})

    def post(client, gun, part):
        return client.post("/predict", json={"gun_id": gun, "readings": part.to_dict("records")})

    with TestClient(main.app) as client:
        # a chunk larger than the buffer minus 10 min of history is refused, not silently truncated
        assert main.MAX_CHUNK == 1200
        assert post(client, "big", raw.iloc[:1201]).status_code == 422

        d = main.app.state.detector
        thr = d.threshold
        d.threshold = -1e9  # every window alarms (all guns below are still in their warm-up -> global threshold)
        try:
            # one 1170-row chunk: every window is scored, the run is long enough -> sustained
            res = post(client, "S1", raw.iloc[:1170]).json()
            assert res["windows_scored"] == 19 and res["consecutive_alarms"] == 19
            assert res["sustained_alarm"] is True and res["severity"] == "critical" and res["alarm_duration_s"] >= 1100
            # 10 s chunks: windows overlap, so 3 consecutive alarming requests are only 80 s of alarm -
            # sustained needs sustain x window = 180 s of continuous alarm, whatever the call rate
            first, sustained_at = None, None
            for i in range(0, 400, 10):
                r = post(client, "S2", raw.iloc[i: i + 10])
                if r.status_code != 200:
                    continue
                res = r.json()
                first = first or pd.Timestamp(res["window_start"])
                if res["sustained_alarm"]:
                    sustained_at = pd.Timestamp(res["window_end"])
                    assert res["consecutive_alarms"] > 3
                    break
            assert sustained_at is not None and 179 <= (sustained_at - first).total_seconds() < 190
        finally:
            d.threshold = thr

        # the rule fires once per episode start and is quiet during the cooldown, even if the code comes back
        part = raw.iloc[:600].copy()
        part.loc[150:199, "error"] = "E029"  # a cross-class E029 flap long before the failure
        part.loc[250:299, "error"] = "E029"
        triggers, active = [], 0
        for i in range(0, 600, 60):
            r = post(client, "R1", part.iloc[i: i + 60])
            if r.status_code == 200:
                res = r.json()
                triggers.append(res["rule_triggered"])
                active += res["rule_code_active"]
                if res["rule_triggered"]:
                    assert res["critical_in_request"] and res["severity"] == "critical" and res["context"]["known_code_class_hint"] == "E04"
                elif not res["sustained_alarm"]:
                    assert res["severity"] != "critical", "a persisting terminal code alone is not critical"
        assert sum(triggers) == 1 and active >= 1

        # a cap-dressing block longer than the 30-min buffer: the non-welding rows keep carrying the last welding
        # values (regression: the buffer held no welding row any more -> NaN features -> HTTP 500 on real test_0)
        assert post(client, "CD", raw.iloc[:1200]).status_code == 200
        for a, b in ((1200, 2390), (2390, 3590)):
            r = post(client, "CD", raw.iloc[a:b].assign(c16=0.0))
            assert r.status_code == 200, r.text
        res = r.json()
        assert res["context"]["non_welding_share"] == 1.0 and res["alarm_held"] is True
        assert np.isfinite(res["anomaly_score"])
        # a stream that starts in non-welding has nothing to carry yet -> 202, not 500
        r = post(client, "CD0", raw.iloc[:120].assign(c16=0.0))
        assert r.status_code == 202 and "no welding row" in r.json()["message"]

        # replay client: the whole 2 h test file in 5-min chunks -> one rule event (the terminal code at the end)
        from replay import replay_file

        s = replay_file(client, str(workspace["test"] / "test_0.csv"), "RP", chunk_s=300)
        assert s["requests"] == 24 and s["rule_triggers"] == 1
        rule_events = [e for e in s["critical_events"] if e["rule_trigger_time"]]
        assert len(rule_events) == 1 and rule_events[0]["class_hint"] == "E04" and "rule" in rule_events[0]["severity_source"]
        assert all(e["handoff_event_id"] for e in s["critical_events"]), "every critical event carries a handoff"
        assert s["windows_scored"] >= 100 and s["last"]["gun_norm"] == "gun"
        with pytest.raises(ValueError):
            replay_file(client, str(workspace["test"] / "test_0.csv"), "RP", chunk_s=1201)
