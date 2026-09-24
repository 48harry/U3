"""
Anomaly-detection baseline on the preprocessed RSW train set.

Input : preprocessed/<file>.parquet from preprocess.py (1 Hz rows, z-scored, with
        segment_id / error_active / non_welding / ttf_s / label columns).
Model : unsupervised detector fitted on NORMAL windows only
          iforest  IsolationForest (default)
          pca      PCA reconstruction error
        Rows are first aggregated into fixed windows (--window, default 60 s) inside each
        segment: mean + std of every continuous feature, max of the flags. That turns the
        37 M second-rows into ~600 k windows and gives the detector some temporal context.
Split : by FILE (= gun), stratified by class, so validation guns are never seen in training.
Normal: label == 0 (outside the pre-failure window) and error_active == 0;
        add --exclude-non-welding to also drop cap-dressing windows.
Score : higher = more anomalous. The alarm threshold is the --threshold-q quantile of the
        training-normal scores (i.e. 1 % false-alarm rate at q = 0.99).
Guns  : --gun-norm center|scale (default center) re-normalises every file / stream with the gun's
        own statistics estimated on a warm-up period (--warmup-hours, default 6) - see the
        "per-gun normalisation" section - and, unless --no-gun-threshold, judges each gun
        against its own threshold calibrated on its warm-up windows. main.py reproduces both
        online from the bundle's `gun_norm`. --gun-norm scale exists but is NOT recommended:
        a gun that idles through its warm-up gets a tiny std and alarms constantly afterwards
        (56 % normal alarm rate on one validation gun).
Gate  : windows whose non-welding share exceeds --alarm-max-non-welding (default 0.5) never
        alarm: their sensor values are carried-forward constants (cap dressing), not
        measurements. Scores / AUROC are unaffected; alarm rates, recall and alarm runs are.
        The gate is stored in the bundle and applied identically by main.py (`alarm_held`).
Eval  : on validation files - AUROC / AUPRC of pre-failure vs normal windows, per class,
        alarm rate on normal windows, recall on pre-failure windows, and per file: how many
        hours before failure the final alarm run (--sustain consecutive windows, reaching the
        failure) starts, and the number of sustained false-alarm runs per day > 24 h earlier.
        Every evaluation also reports (a) the OPERATING POINT: how many files get a final alarm
        run >= OP_LEAD_MIN minutes before the failure and <= OP_FALSE_RUNS false runs per day,
        (b) the TERMINAL-CODE RULE as a baseline: a window whose error code is the class's
        terminal code (E012/E016/E028/E029) is an alarm regardless of the model - what main.py
        does with `known_code_class_hint`. --label-window relabels the pre-failure window
        (default: the 3600 s preprocess.py used), --cv k adds a gun-level k-fold estimate.
Test  : the 8 held-out files (preprocess.py --split test -> preprocessed/test/) get the same
        evaluation, with the SAME threshold, after training (metrics["test"]) or on their own
        with --evaluate (loads the saved bundle, writes models/baseline_<model>_test_metrics.json
        and per-file scores to models/scores/). Their class is the one preprocess.py inferred
        from the terminal code, so per-class numbers on test are "by inferred class".
Save  : models/baseline_<model>.joblib - a dict with the fitted model, feature list, window,
        threshold, the scaler + preprocess config from preprocess.py and the metrics. Inference:

            from train import load_bundle, score_frame
            b = load_bundle("models/baseline_iforest.joblib")
            out = score_frame(b, pd.read_parquet("preprocessed/E04_3.parquet"))  # -> time, score, alarm

        or  python train.py --score preprocessed/E04_3.parquet

Usage:
    python train.py                              # iforest, 60 s windows, then val + test evaluation
    python train.py --evaluate                   # test evaluation only, with the saved bundle
    python train.py --model pca --window 120
    python train.py --max-files 8                # quick check
"""
import argparse
import datetime as dt
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["E01", "E02", "E03", "E04"]
# target operating point (test.md §1, history.md P2): a sustained alarm at least this early, at most this many false runs
OP_LEAD_MIN = 30.0
OP_FALSE_RUNS = 1.0
META_COLS = {"file", "class", "gun", "error_code", "segment_id", "dow", "ttf_s", "label",
             "error_active", "terminal_code", "non_welding", "warmup", "gun_norm"}
FLAG_COLS = ["error_active", "terminal_code", "non_welding", "label", "warmup"]
# c19 ("offset value in robot") is a counter that grows ~55/s through every file. Every file is
# exactly 7 days long and ends at its failure, so within a file c19 == time since start ==
# 168 h - time to failure: a label leak, not a measurement (a supervised model reaches AUROC 0.98
# with it and 0.62 without - history.md, old test.md §8). Dropped from the model input by default.
DROP_FEATURES_DEFAULT = ["c19"]


# ------------------------------------------------------------- windowing
def feature_columns(df, drop=()):
    """Continuous model inputs = every column that is not metadata / flag (minus `drop`)."""
    return [c for c in df.columns if c not in META_COLS and c not in set(drop)]


def window_features(df, window, feat_cols=None):
    """1 Hz rows -> one row per (segment, window): mean & std of features, max of flags, min ttf."""
    feat_cols = feat_cols or feature_columns(df)
    agg = {c: ["mean", "std"] for c in feat_cols}
    agg.update({c: "max" for c in FLAG_COLS if c in df.columns})
    agg["non_welding"] = "mean"  # share of the window, not a 0/1 flag like the others
    agg["ttf_s"] = "min"
    g = df.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).agg(agg)
    # only the model features get the _mean/_std suffix; metadata keeps its own name, so that
    # `non_welding` (aggregated with "mean") does not silently become `non_welding_mean`.
    g.columns = [f"{a}_{b}" if a in feat_cols and b in ("mean", "std") else a for a, b in g.columns]
    g = g.droplevel(0)
    g = g[g["ttf_s"].notna()]
    g["n"] = df.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).size().droplevel(0).reindex(g.index)
    g = g[g["n"] >= max(2, window // 2)]  # drop windows with less than half the samples
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)
    for c in ("file", "class", "gun"):
        g[c] = df[c].iloc[0]
    return g


def model_input_columns(feat_cols):
    return [f"{c}_mean" for c in feat_cols] + [f"{c}_std" for c in feat_cols]


# ------------------------------------------------------- per-gun normalisation
# Guns differ by an offset of up to ~1.5 sigma on several sensors (c1, c14, ...), so one global
# z-score + one global threshold gives per-gun normal alarm rates anywhere between 0 and 13 %.
# preprocess.py's --scale per-file cannot be reproduced online, so the per-gun statistics are
# estimated from a WARM-UP period instead: the normal welding rows of the first `warmup_s` of a
# file (online: of a gun's stream). Rows after the warm-up are re-normalised with them; the
# warm-up rows themselves keep the global scaling (that is all the serving layer has at the
# time) and are flagged `warmup`. mode "center" subtracts the gun mean only (the gun constants
# c7-c9 become 0, i.e. --drop-static); "scale" also divides by the gun std, floored at
# std_floor (in global-z units) because a gun that idles through its warm-up has std ~ 0.
# With threshold_q, the warm-up windows are scored with the gun statistics and the gun's
# threshold becomes max(global threshold, that quantile) - a per-gun 99 % operating point.
GUN_NORM_DEFAULT = {"mode": "center", "warmup_s": 6 * 3600, "min_rows": 600, "std_floor": 0.25, "threshold_q": None}


def gun_norm_columns(feat_cols, scaler_cols=None):
    """Columns the per-gun normalisation applies to: the z-scored sensors / counter features
    (never the ratios weld_duty / error_share, nor the time-of-day features)."""
    base = scaler_cols or [c for c in feat_cols if c.startswith("c") or c in ("welds_delta", "pos_delta", "welds_10min")]
    return [c for c in feat_cols if c in base]


def gun_norm_stats(df, cols, cfg):
    """Mean/std of the normal welding rows in the first cfg['warmup_s'] of the frame, or None when
    there are fewer than cfg['min_rows'] of them (the file then keeps the global scaling)."""
    t_end = df.index[0] + pd.Timedelta(seconds=cfg["warmup_s"])
    ok = (df.index < t_end) & (df["non_welding"].to_numpy() == 0) & (df["error_active"].to_numpy() == 0)
    n = int(ok.sum())
    if n < cfg["min_rows"]:
        return None
    x = df.loc[ok, cols].astype("float64")
    mean = x.mean().to_numpy()
    if cfg["mode"] == "scale":
        std = np.maximum(x.std(ddof=0).fillna(0.0).to_numpy(), cfg["std_floor"])
    else:
        std = np.ones(len(cols))
    return {"mean": mean, "std": std, "t_end": t_end, "n": n}


def apply_gun_norm(df, cols, stats, rows=None):
    """(x - mean) / std on `cols`, for the rows at/after the warm-up end (default) or a boolean mask."""
    out = df.copy()
    sel = (out.index >= stats["t_end"]) if rows is None else rows
    out.loc[sel, cols] = ((out.loc[sel, cols].to_numpy(dtype="float64") - stats["mean"]) / stats["std"]).astype("float32")
    return out


def window_file(df, window, feat_cols, gn=None, cols=None):
    """Window one preprocessed file. With a gun-norm config the rows after the warm-up are normalised
    with the gun's own warm-up statistics (exactly what main.py does online) and warm-up windows are
    flagged. Returns (windows, calibration windows): the latter are the warm-up windows normalised
    with the gun statistics (they calibrate the gun threshold), None without gun normalisation."""
    stats = gun_norm_stats(df, cols, gn) if gn else None
    if stats is None:
        w = window_features(df, window, feat_cols)
        w["warmup"], w["gun_norm"] = 0.0, "global"
        return w, None
    df = df.assign(warmup=(df.index < stats["t_end"]).astype("float32"))
    w = window_features(apply_gun_norm(df, cols, stats), window, feat_cols)
    w["gun_norm"] = "gun"
    wu = df.index < stats["t_end"]
    calib = window_features(apply_gun_norm(df[wu], cols, stats, rows=np.ones(int(wu.sum()), dtype=bool)), window, feat_cols)
    return w, calib


def gun_thresholds(model, model_cols, calib, global_thr, q):
    """file -> alarm threshold = max(global, q-quantile of the gun-normalised warm-up window scores)."""
    if calib is None or q is None or len(calib) == 0:
        return {}
    out = {}
    for f, part in calib.groupby("file"):
        s = anomaly_score(model, part[model_cols].to_numpy(dtype=np.float32))
        out[f] = float(max(global_thr, np.quantile(s, q)))
    return out


def window_thresholds(w, global_thr, gun_thr):
    """Per-window threshold: the gun's own after its warm-up, the global one otherwise."""
    if not gun_thr:
        return global_thr
    thr = w["file"].map(gun_thr).fillna(global_thr).to_numpy(dtype="float64")
    if "warmup" in w.columns:
        thr = np.where(w["warmup"].to_numpy() > 0, global_thr, thr)
    return thr


# ----------------------------------------------------------------- models
class PCADetector:
    """Reconstruction error of a PCA fitted on normal windows."""

    def __init__(self, n_components=0.95, random_state=42):
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def score_samples(self, X):  # sklearn convention: higher = more normal
        rec = self.pca.inverse_transform(self.pca.transform(X))
        return -((X - rec) ** 2).mean(axis=1)


def build_model(name, seed):
    if name == "iforest":
        return IsolationForest(n_estimators=300, max_samples=1024, contamination="auto",
                               random_state=seed, n_jobs=-1)
    if name == "pca":
        return PCADetector(random_state=seed)
    raise ValueError(name)


def anomaly_score(model, X):
    return -model.score_samples(X)  # flip: higher = more anomalous


# ------------------------------------------------------------- evaluation
def alarm_mask(scores, threshold, non_welding=None, gate=None):
    """Window-level alarm decision: score above threshold, unless the window is mostly non-welding
    (share > gate) - those windows hold carried-forward constants, not measurements."""
    alarm = np.asarray(scores) > threshold
    if gate is not None and non_welding is not None:
        alarm &= np.asarray(non_welding) <= gate
    return alarm


def alarm_runs(alarm, sustain):
    """(start, end) index pairs of runs of >= `sustain` consecutive alarm windows."""
    above = np.concatenate([[False], np.asarray(alarm, dtype=bool), [False]])
    edges = np.flatnonzero(above[1:] != above[:-1])
    return [(a, b) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= sustain]


def alarm_timing(alarm, ttf_s, sustain, normal_before_h=24):
    """Chronological alarm flags -> when the final alarm run (the one that reaches the failure)
    starts, in hours before failure, and how many sustained false-alarm runs per day occur
    > normal_before_h before the failure."""
    runs = alarm_runs(alarm, sustain)
    n = len(alarm)
    final_start_h = np.nan
    if runs and runs[-1][1] >= n - sustain:  # last run touches the end of the series
        final_start_h = ttf_s[runs[-1][0]] / 3600
    normal_days = max((ttf_s[0] - normal_before_h * 3600) / 86400, 1e-9)
    n_false = sum(1 for a, b in runs if ttf_s[a] > normal_before_h * 3600)
    return final_start_h, n_false / normal_days


def evaluate(val, scores, threshold, sustain, gate=None, gun_thr=None):
    """threshold: the global one; gun_thr: file -> per-gun threshold (applied after the warm-up)."""
    normal = (val["label"] == 0) & (val["error_active"] == 0)
    pre = val["label"] == 1
    y, s = pre.values.astype(int), scores
    nw = val["non_welding"].values if "non_welding" in val.columns else None
    thr = window_thresholds(val, threshold, gun_thr)
    a = alarm_mask(s, thr, nw, gate)
    welding = np.ones(len(val), dtype=bool) if nw is None or gate is None else nw <= gate

    def auroc(mask):
        return float(roc_auc_score(y[mask], s[mask])) if y[mask].any() and not y[mask].all() else np.nan

    m = {"n_windows": int(len(val)), "n_pre_failure": int(pre.sum()), "n_normal": int(normal.sum()),
         "auroc": auroc(np.ones(len(val), dtype=bool)),
         "auprc": float(average_precision_score(y, s)) if y.any() else np.nan,
         "alarm_rate_normal": float(a[normal.values].mean()),
         "recall_pre_failure": float(a[pre.values].mean()),
         "alarm_rate_error_state": float(a[(val["error_active"] == 1).values].mean())
         if (val["error_active"] == 1).any() else np.nan,
         # gate diagnostics: how much was held, and how the score separates on welding windows only
         "alarm_gate_non_welding": gate,
         "held_windows": int(((s > thr) & ~welding).sum()),
         "held_share_of_windows": float((~welding).mean()),
         "auroc_welding_windows": auroc(welding),
         # per-gun normalisation diagnostics
         "warmup_windows": int(val["warmup"].sum()) if "warmup" in val.columns else 0,
         "files_gun_normalised": int((val.groupby("file")["gun_norm"].first() == "gun").sum()) if "gun_norm" in val.columns else 0,
         "files_gun_threshold": len(gun_thr or {}),
         "alarm_rate_normal_after_warmup": float(a[normal.values & (val["warmup"].values == 0)].mean())
         if "warmup" in val.columns and (normal.values & (val["warmup"].values == 0)).any() else np.nan}
    per_class, per_file = {}, {}
    for k in CLASSES:
        sel = (val["class"] == k).values
        if sel.any() and y[sel].any() and not y[sel].all():
            per_class[k] = {"auroc": auroc(sel), "auroc_welding_windows": auroc(sel & welding),
                            "recall_pre_failure": float(a[sel & pre.values].mean()),
                            "alarm_rate_normal": float(a[sel & normal.values].mean())}
    for f, idx in val.groupby("file").indices.items():
        order = idx[np.argsort(-val["ttf_s"].values[idx])]  # chronological
        final_h, false_per_day = alarm_timing(a[order], val["ttf_s"].values[order], sustain)
        per_file[f] = {"final_alarm_run_starts_h_before_failure": float(final_h),
                       "false_alarm_runs_per_day": float(false_per_day),
                       "alarm_rate_normal": float(a[idx][normal.values[idx]].mean()),
                       "held_share_of_windows": float((~welding[idx]).mean()),
                       "gun_norm": str(val["gun_norm"].values[idx[0]]) if "gun_norm" in val.columns else "global",
                       "threshold": float((gun_thr or {}).get(f, threshold))}
    rates = np.array([v["alarm_rate_normal"] for v in per_file.values()])
    m["alarm_rate_normal_per_file_min_max_std"] = [float(rates.min()), float(rates.max()), float(rates.std())]
    # terminal-code rule (main.py's known_code_class_hint) as a model-free baseline / safety net
    rule = val["terminal_code"].to_numpy() > 0 if "terminal_code" in val.columns else np.zeros(len(val), dtype=bool)
    err = (val["error_active"] == 1).to_numpy()
    a_rule = a | rule
    for f, idx in val.groupby("file").indices.items():
        order = idx[np.argsort(-val["ttf_s"].values[idx])]
        final_rule_h, _ = alarm_timing(a_rule[order], val["ttf_s"].values[order], sustain)
        per_file[f]["final_alarm_run_with_rule_h_before_failure"] = float(final_rule_h)
        per_file[f]["rule_lead_min"] = float(val["ttf_s"].values[idx][rule[idx]].max() / 60) if rule[idx].any() else np.nan
    m["rule_terminal_code"] = {
        "alarm_rate_error_state": float(a_rule[err].mean()) if err.any() else np.nan,
        "alarm_rate_terminal_state": float(a_rule[rule].mean()) if rule.any() else np.nan,
        "n_terminal_windows": int(rule.sum()),
        "files_with_terminal_code": int(sum(1 for v in per_file.values() if v["rule_lead_min"] == v["rule_lead_min"])),
        "rule_lead_min_median": float(np.nanmedian([v["rule_lead_min"] for v in per_file.values()]))}
    # operating point
    lead = np.nan_to_num(np.array([v["final_alarm_run_starts_h_before_failure"] for v in per_file.values()]) * 60, nan=-1)
    lead_rule = np.nan_to_num(np.array([v["final_alarm_run_with_rule_h_before_failure"] for v in per_file.values()]) * 60, nan=-1)
    false_runs = np.array([v["false_alarm_runs_per_day"] for v in per_file.values()])
    m["operating_point"] = {
        "lead_min_target": OP_LEAD_MIN, "false_runs_per_day_target": OP_FALSE_RUNS, "files": len(per_file),
        "files_lead_ok": int((lead >= OP_LEAD_MIN).sum()), "files_false_runs_ok": int((false_runs <= OP_FALSE_RUNS).sum()),
        "files_both_ok": int(((lead >= OP_LEAD_MIN) & (false_runs <= OP_FALSE_RUNS)).sum()),
        "files_with_final_run": int((lead >= 0).sum()), "files_with_final_run_with_rule": int((lead_rule >= 0).sum()),
        # the rule alone: first terminal-code window before the failure (the code often clears before the last row,
        # so a rule "run" need not touch the end of the file - judge the rule by its lead, not by the run)
        "files_rule_lead_ok": int(sum(1 for v in per_file.values() if v["rule_lead_min"] >= OP_LEAD_MIN)),
        "final_lead_min_median": float(np.median(lead[lead >= 0])) if (lead >= 0).any() else np.nan,
        "final_lead_min_median_with_rule": float(np.median(lead_rule[lead_rule >= 0])) if (lead_rule >= 0).any() else np.nan}
    m["per_class"], m["per_file"] = per_class, per_file
    return m


def relabel(w, label_window):
    """Recompute `label` (pre-failure window) from ttf_s - lets the evaluation vary the label window
    without re-running preprocess.py."""
    w = w.copy()
    w["label"] = (w["ttf_s"] <= label_window).astype("float32")
    return w


def cv_folds(files, k, seed):
    """Gun-level k-fold, stratified by class: list of k validation file lists."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in CLASSES:
        fs = sorted(f for f in files if os.path.basename(f).startswith(c))
        rng.shuffle(fs)
        for i, f in enumerate(fs):
            folds[i % k].append(f)
    return [sorted(f) for f in folds]


def print_metrics(tag, v):
    lo, hi, sd = v.get("alarm_rate_normal_per_file_min_max_std", (np.nan, np.nan, np.nan))
    print(f"{tag}: AUROC {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}  alarm@normal {v['alarm_rate_normal']:.3f}  "
          f"recall@pre-failure {v['recall_pre_failure']:.3f}  ({v['n_windows']:,} windows"
          + (f", {v['held_windows']:,} alarms held by the non-welding gate" if v.get("alarm_gate_non_welding") is not None else "")
          + f"; per-file alarm@normal {lo:.3f}..{hi:.3f} sd {sd:.3f}"
          + (f"; {v['files_gun_normalised']} files gun-normalised" if v.get("files_gun_normalised") else "")
          + (f", {v['files_gun_threshold']} with a gun threshold" if v.get("files_gun_threshold") else "") + ")")
    for k, r in v["per_class"].items():
        print(f"  {k}: AUROC {r['auroc']:.3f}  recall {r['recall_pre_failure']:.3f}  alarm@normal {r['alarm_rate_normal']:.3f}")
    op, rule = v.get("operating_point"), v.get("rule_terminal_code")
    if op and rule:
        print(f"  operating point (lead >= {op['lead_min_target']:.0f} min, false runs <= {op['false_runs_per_day_target']:.0f}/day): "
              f"{op['files_both_ok']}/{op['files']} files (lead ok {op['files_lead_ok']}, false runs ok {op['files_false_runs_ok']}); "
              f"final run in {op['files_with_final_run']} files (median lead {op['final_lead_min_median']:.1f} min); "
              f"with terminal-code rule: {op['files_with_final_run_with_rule']} files, median lead "
              f"{op['final_lead_min_median_with_rule']:.1f} min, error-state alarm rate {rule['alarm_rate_error_state']:.3f}")


def test_files_in(test_dir):
    return sorted(glob.glob(os.path.join(test_dir, "test_*.parquet")))


def evaluate_test(model, feat_cols, model_cols, window, threshold, sustain, files, score_dir=None, tag="", gate=None,
                  gn=None, label_window=None):
    """Window + score the preprocessed test files with the frozen model/threshold (+ the bundle's gun
    normalisation) and run the same evaluation as for validation. Optionally writes per-file score
    CSVs (time, score, alarm, threshold, ...)."""
    ws, cs = [], []
    cols = gun_norm_columns(feat_cols, gn["columns"]) if gn else None
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        w, calib = window_file(df, window, feat_cols, gn, cols)
        if label_window:
            w = relabel(w, label_window)
        ws.append(w)
        if calib is not None:
            cs.append(calib)
        print(f"[test {i}/{len(files)}] {os.path.basename(f)} (class {df['class'].iloc[0]}): "
              f"{len(df):,} rows -> {len(w):,} windows{'' if calib is None else ' (gun-normalised)'}", flush=True)
    test_w = pd.concat(ws)
    scores = anomaly_score(model, test_w[model_cols].to_numpy(dtype=np.float32))
    gun_thr = gun_thresholds(model, model_cols, pd.concat(cs) if cs else None, threshold, (gn or {}).get("threshold_q"))
    m = evaluate(test_w, scores, threshold, sustain, gate, gun_thr)
    m["files"] = [os.path.basename(f) for f in files]
    m["file_class"] = {f: str(c) for f, c in test_w.groupby("file")["class"].first().items()}
    m["class_source"] = "inferred by preprocess.py from the terminal code in the last 10 min"
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)
        thr = window_thresholds(test_w, threshold, gun_thr)
        test_w = test_w.assign(score=scores, alarm=alarm_mask(scores, thr, test_w["non_welding"].values, gate),
                               threshold=thr)
        for f, part in test_w.groupby("file"):
            cols = ["score", "alarm", "threshold"] + [c for c in ("ttf_s", "label", "error_active", "non_welding", "warmup")
                                                     if c in part.columns]
            part[cols].to_csv(os.path.join(score_dir, f"{f}_{tag}.csv"))
    return m


# ------------------------------------------------------------- inference
def load_bundle(path):
    return joblib.load(path)


def score_frame(bundle, df):
    """Preprocessed 1 Hz frame (as written by preprocess.py) -> DataFrame[time, score, alarm, threshold, ...]."""
    gn = bundle.get("gun_norm")
    cols = gun_norm_columns(bundle["feature_cols"], gn["columns"]) if gn else None
    w, calib = window_file(df, bundle["window"], bundle["feature_cols"], gn, cols)
    X = w[bundle["model_cols"]].to_numpy(dtype=np.float32)
    s = anomaly_score(bundle["model"], X)
    gun_thr = gun_thresholds(bundle["model"], bundle["model_cols"], calib, bundle["threshold"], (gn or {}).get("threshold_q"))
    thr = window_thresholds(w, bundle["threshold"], gun_thr)
    alarm = alarm_mask(s, thr, w["non_welding"].values if "non_welding" in w.columns else None,
                       bundle.get("alarm_max_non_welding"))
    out = pd.DataFrame({"score": s, "alarm": alarm, "threshold": thr}, index=w.index)
    for c in ("ttf_s", "label", "error_active", "non_welding", "warmup"):
        if c in w.columns:
            out[c] = w[c].values
    return out


# ------------------------------------------------------------------- main
def split_files(files, val_frac, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for k in CLASSES:
        fs = sorted(f for f in files if os.path.basename(f).startswith(k))
        rng.shuffle(fs)
        n_val = max(1, int(round(len(fs) * val_frac))) if len(fs) > 1 else 0
        val += fs[:n_val]
        train += fs[n_val:]
    return sorted(train), sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "preprocessed"))
    ap.add_argument("--test-dir", default=None, help="preprocessed test files; default <data-dir>/test. "
                    "Evaluated after training when present; pass --no-test to skip")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--evaluate", action="store_true", help="skip training; evaluate the saved bundle on --test-dir")
    ap.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--model", choices=["iforest", "pca"], default="iforest")
    ap.add_argument("--window", type=int, default=60, help="window length in seconds")
    ap.add_argument("--val-frac", type=float, default=0.25, help="share of files per class held out")
    ap.add_argument("--threshold-q", type=float, default=0.99, help="quantile of train-normal scores")
    ap.add_argument("--sustain", type=int, default=3, help="consecutive alarm windows for a sustained alarm")
    ap.add_argument("--alarm-max-non-welding", type=lambda v: None if str(v).lower() in ("none", "off") else float(v),
                    default=0.5, help="windows with a larger non-welding share never alarm (none = no gate)")
    ap.add_argument("--exclude-non-welding", action="store_true", help="drop cap-dressing windows from training")
    ap.add_argument("--drop-features", nargs="*", default=DROP_FEATURES_DEFAULT,
                    help="preprocessed columns kept out of the model input (default: c19, a time counter = label leak)")
    ap.add_argument("--gun-norm", choices=["none", "center", "scale"], default=GUN_NORM_DEFAULT["mode"],
                    help="per-gun re-normalisation from a warm-up period: center = subtract the gun mean, "
                         "scale = also divide by the gun std (floored), none = global z-score only")
    ap.add_argument("--warmup-hours", type=float, default=GUN_NORM_DEFAULT["warmup_s"] / 3600,
                    help="warm-up period per file / stream for the gun statistics")
    ap.add_argument("--warmup-min-rows", type=int, default=GUN_NORM_DEFAULT["min_rows"],
                    help="fewer normal welding rows in the warm-up -> the file keeps the global scaling")
    ap.add_argument("--std-floor", type=float, default=GUN_NORM_DEFAULT["std_floor"], help="--gun-norm scale: min gun std (z units)")
    ap.add_argument("--no-gun-threshold", action="store_true",
                    help="judge every gun against the global threshold; default: per-gun threshold = max(global, "
                         "--threshold-q quantile of the gun-normalised warm-up window scores)")
    ap.add_argument("--label-window", type=int, default=None,
                    help="pre-failure window in seconds for training / evaluation (default: the one preprocess.py used, 3600)")
    ap.add_argument("--cv", type=int, default=None, help="also run a gun-level k-fold (metrics['cv']) before the final fit")
    ap.add_argument("--max-train-windows", type=int, default=None, help="subsample normal windows")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", default=None, help="skip training; score this preprocessed parquet with the saved model")
    args = ap.parse_args()

    if args.score:
        b = load_bundle(os.path.join(args.model_dir, f"baseline_{args.model}.joblib"))
        out = score_frame(b, pd.read_parquet(args.score))
        os.makedirs(os.path.join(args.model_dir, "scores"), exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.score))[0]
        dst = os.path.join(args.model_dir, "scores", f"{stem}_{args.model}.csv")
        out.to_csv(dst)
        print(f"{len(out)} windows, alarm rate {out['alarm'].mean():.3f}, threshold {b['threshold']:.4f} -> {dst}")
        return

    test_dir = args.test_dir or os.path.join(args.data_dir, "test")
    test_files = [] if args.no_test else test_files_in(test_dir)
    if args.evaluate:
        if not test_files:
            raise SystemExit(f"no test_*.parquet in {test_dir} - run preprocess.py --split test first")
        bundle_path = os.path.join(args.model_dir, f"baseline_{args.model}.joblib")
        if not os.path.exists(bundle_path):
            raise SystemExit(f"{bundle_path} not found - train first")
        b = load_bundle(bundle_path)
        print(f"evaluating {bundle_path} (window {b['window']}s, threshold {b['threshold']:.4f}, "
              f"sustain {b['sustain']}) on {len(test_files)} test files")
        m = evaluate_test(b["model"], b["feature_cols"], b["model_cols"], b["window"], b["threshold"],
                          b["sustain"], test_files, os.path.join(args.model_dir, "scores"), args.model,
                          gate=b.get("alarm_max_non_welding"), gn=b.get("gun_norm"),
                          label_window=(b.get("args") or {}).get("label_window"))
        print_metrics("test", m)
        dst = os.path.join(args.model_dir, f"baseline_{args.model}_test_metrics.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump({"bundle": bundle_path, "created": b.get("created"), "threshold": b["threshold"],
                       "window": b["window"], "sustain": b["sustain"],
                       "alarm_max_non_welding": b.get("alarm_max_non_welding"), "gun_norm": b.get("gun_norm"), "test": m,
                       "evaluated": dt.datetime.now().isoformat(timespec="seconds")}, f, indent=2, default=str)
        print(f"saved {dst}")
        return

    files = sorted(glob.glob(os.path.join(args.data_dir, "E0*.parquet")))
    if not files:
        raise SystemExit(f"no E0*.parquet in {args.data_dir} — run preprocess.py first")
    if args.max_files:  # keep the classes balanced in a quick run
        per_class = [[f for f in files if os.path.basename(f).startswith(k)] for k in CLASSES]
        files = sorted(f for fs in per_class for f in fs[: -(-args.max_files // len(CLASSES))])
    train_files, val_files = split_files(files, args.val_frac, args.seed)
    print(f"train files {len(train_files)}, val files {len(val_files)}")

    scaler_path = os.path.join(args.data_dir, "scaler.json")
    config_path = os.path.join(args.data_dir, "preprocess_config.json")
    scaler = json.load(open(scaler_path, encoding="utf-8")) if os.path.exists(scaler_path) else None
    gn = None
    if args.gun_norm != "none":
        gn = {"mode": args.gun_norm, "warmup_s": int(round(args.warmup_hours * 3600)), "min_rows": args.warmup_min_rows,
              "std_floor": args.std_floor, "threshold_q": None if args.no_gun_threshold else args.threshold_q}

    t0 = time.time()
    feat_cols, gn_cols, parts, calibs = None, None, {}, {}
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        feat_cols = feat_cols or feature_columns(df, args.drop_features)
        if gn and gn_cols is None:
            gn_cols = gun_norm_columns(feat_cols, (scaler or {}).get("columns"))
            gn["columns"] = gn_cols
        w, calib = window_file(df, args.window, feat_cols, gn, gn_cols)
        if args.label_window:
            w = relabel(w, args.label_window)
        parts[f] = w
        if calib is not None:
            calibs[f] = calib
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {len(df):,} rows -> {len(w):,} windows"
              f"{'' if calib is None else ' (gun-normalised)'}", flush=True)
    print(f"windowed in {time.time() - t0:.0f}s" + (f", gun-norm {gn['mode']} on {gn_cols}" if gn else ""))
    model_cols = model_input_columns(feat_cols)
    q_gun = (gn or {}).get("threshold_q")

    def fit_and_eval(tr_files, va_files):
        tr = pd.concat([parts[f] for f in tr_files])
        normal = (tr["label"] == 0) & (tr["error_active"] == 0)
        if args.exclude_non_welding:
            normal &= tr["non_welding"] == 0
        fit = tr[normal]
        if args.max_train_windows and len(fit) > args.max_train_windows:
            fit = fit.sample(args.max_train_windows, random_state=args.seed)
        X = fit[model_cols].to_numpy(dtype=np.float32)
        model = build_model(args.model, args.seed).fit(X)
        s_tr = anomaly_score(model, X)
        thr = float(np.quantile(s_tr, args.threshold_q))
        m_val = None
        if va_files:
            va = pd.concat([parts[f] for f in va_files])
            cal = [calibs[f] for f in va_files if f in calibs]
            gun_thr = gun_thresholds(model, model_cols, pd.concat(cal) if cal else None, thr, q_gun)
            m_val = evaluate(va, anomaly_score(model, va[model_cols].to_numpy(dtype=np.float32)), thr, args.sustain,
                             args.alarm_max_non_welding, gun_thr)
        return model, X, s_tr, thr, m_val

    cv = None
    if args.cv:
        folds = cv_folds(files, args.cv, args.seed)
        fold_metrics = []
        for i, va_files in enumerate(folds, 1):
            tr_files = [f for f in files if f not in va_files]
            _, _, _, thr_i, m_i = fit_and_eval(tr_files, va_files)
            fold_metrics.append({k: m_i[k] for k in ("auroc", "auprc", "recall_pre_failure", "alarm_rate_normal",
                                                     "alarm_rate_normal_per_file_min_max_std")}
                                | {"threshold": thr_i, "val_files": [os.path.basename(f) for f in va_files],
                                   "operating_point": m_i["operating_point"], "per_class": m_i["per_class"]})
            print(f"cv fold {i}/{args.cv}: AUROC {m_i['auroc']:.3f}  recall {m_i['recall_pre_failure']:.3f}  "
                  f"alarm@normal {m_i['alarm_rate_normal']:.3f}  ({len(va_files)} guns)", flush=True)
        keys = ("auroc", "auprc", "recall_pre_failure", "alarm_rate_normal")
        cv = {"k": args.cv, "folds": fold_metrics,
              "mean": {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys},
              "sd": {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys},
              "files_both_ok": int(sum(m["operating_point"]["files_both_ok"] for m in fold_metrics)),
              "files_with_final_run": int(sum(m["operating_point"]["files_with_final_run"] for m in fold_metrics))}
        print(f"cv {args.cv}-fold: AUROC {cv['mean']['auroc']:.3f} +- {cv['sd']['auroc']:.3f}  "
              f"recall {cv['mean']['recall_pre_failure']:.3f} +- {cv['sd']['recall_pre_failure']:.3f}  "
              f"alarm@normal {cv['mean']['alarm_rate_normal']:.3f} +- {cv['sd']['alarm_rate_normal']:.3f}  "
              f"operating point {cv['files_both_ok']}/{len(files)} guns", flush=True)

    print(f"fitting {args.model} on the normal windows of {len(train_files)} files x {len(model_cols)} features")
    model, X, train_scores, threshold, m_val = fit_and_eval(train_files, val_files)
    print(f"fitted on {len(X):,} normal windows, threshold {threshold:.4f}")

    metrics = {"train": {"n_fit_windows": int(len(X)), "threshold": threshold,
                         "score_mean": float(train_scores.mean()), "score_std": float(train_scores.std())}}
    if cv:
        metrics["cv"] = cv
    if m_val is not None:
        metrics["val"] = m_val
        print_metrics("val", metrics["val"])
    if test_files:
        metrics["test"] = evaluate_test(model, feat_cols, model_cols, args.window, threshold, args.sustain, test_files,
                                        gate=args.alarm_max_non_welding, gn=gn, label_window=args.label_window)
        print_metrics("test", metrics["test"])
    else:
        print(f"no test files in {test_dir} - skipped test evaluation")

    os.makedirs(args.model_dir, exist_ok=True)
    bundle = {
        "model": model, "model_type": args.model, "feature_cols": feat_cols, "model_cols": model_cols,
        "window": args.window, "threshold": threshold, "threshold_q": args.threshold_q, "sustain": args.sustain,
        "alarm_max_non_welding": args.alarm_max_non_welding,
        # per-gun normalisation recipe (None = global z-score only); main.py reproduces it online
        "gun_norm": gn,
        "dropped_features": list(args.drop_features),
        # per-feature reference (median of the normal training windows): the serving layer replaces one
        # feature at a time with it to attribute a score to individual features
        "feature_reference": np.median(X, axis=0).astype(float).tolist(),
        "feature_scale": X.std(axis=0).astype(float).tolist(),
        "scaler": scaler,
        # the serving layer reproduces the preprocessing online from these (gap limit, c16 rule, resample)
        "preprocess_config": json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else None,
        "train_files": [os.path.basename(f) for f in train_files],
        "val_files": [os.path.basename(f) for f in val_files],
        "metrics": metrics, "args": vars(args), "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(args.model_dir, f"baseline_{args.model}.joblib")
    joblib.dump(bundle, path, compress=3)
    with open(os.path.join(args.model_dir, f"baseline_{args.model}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2, default=str)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
