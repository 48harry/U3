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
Eval  : on validation files - AUROC / AUPRC of pre-failure vs normal windows, per class,
        alarm rate on normal windows, recall on pre-failure windows, and per file: how many
        hours before failure the final alarm run (--sustain consecutive windows, reaching the
        failure) starts, and the number of sustained false-alarm runs per day > 24 h earlier.
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
META_COLS = {"file", "class", "gun", "error_code", "segment_id", "dow", "ttf_s", "label",
             "error_active", "terminal_code", "non_welding"}
FLAG_COLS = ["error_active", "terminal_code", "non_welding", "label"]


# ------------------------------------------------------------- windowing
def feature_columns(df):
    """Continuous model inputs = every column that is not metadata / flag."""
    return [c for c in df.columns if c not in META_COLS]


def window_features(df, window, feat_cols=None):
    """1 Hz rows -> one row per (segment, window): mean & std of features, max of flags, min ttf."""
    feat_cols = feat_cols or feature_columns(df)
    agg = {c: ["mean", "std"] for c in feat_cols}
    agg.update({c: "max" for c in FLAG_COLS if c in df.columns})
    agg["non_welding"] = "mean"
    agg["ttf_s"] = "min"
    g = df.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).agg(agg)
    g.columns = [f"{a}_{b}" if b in ("mean", "std") else a for a, b in g.columns]
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
def alarm_runs(scores, threshold, sustain):
    """(start, end) index pairs of runs of >= `sustain` consecutive windows above the threshold."""
    above = np.concatenate([[False], scores > threshold, [False]])
    edges = np.flatnonzero(above[1:] != above[:-1])
    return [(a, b) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= sustain]


def alarm_timing(scores, ttf_s, threshold, sustain, normal_before_h=24):
    """Chronological windows -> when the final alarm run (the one that reaches the failure) starts,
    in hours before failure, and how many sustained false-alarm runs per day occur > normal_before_h
    before the failure."""
    runs = alarm_runs(scores, threshold, sustain)
    n = len(scores)
    final_start_h = np.nan
    if runs and runs[-1][1] >= n - sustain:  # last run touches the end of the series
        final_start_h = ttf_s[runs[-1][0]] / 3600
    normal_days = max((ttf_s[0] - normal_before_h * 3600) / 86400, 1e-9)
    n_false = sum(1 for a, b in runs if ttf_s[a] > normal_before_h * 3600)
    return final_start_h, n_false / normal_days


def evaluate(val, scores, threshold, sustain):
    normal = (val["label"] == 0) & (val["error_active"] == 0)
    pre = val["label"] == 1
    y, s = pre.values.astype(int), scores
    m = {"n_windows": int(len(val)), "n_pre_failure": int(pre.sum()), "n_normal": int(normal.sum()),
         "auroc": float(roc_auc_score(y, s)) if y.any() and not y.all() else np.nan,
         "auprc": float(average_precision_score(y, s)) if y.any() else np.nan,
         "alarm_rate_normal": float((s[normal.values] > threshold).mean()),
         "recall_pre_failure": float((s[pre.values] > threshold).mean()),
         "alarm_rate_error_state": float((s[(val["error_active"] == 1).values] > threshold).mean())
         if (val["error_active"] == 1).any() else np.nan}
    per_class, per_file = {}, {}
    for k in CLASSES:
        sel = (val["class"] == k).values
        if sel.any() and y[sel].any() and not y[sel].all():
            per_class[k] = {"auroc": float(roc_auc_score(y[sel], s[sel])),
                            "recall_pre_failure": float((s[sel & pre.values] > threshold).mean()),
                            "alarm_rate_normal": float((s[sel & normal.values] > threshold).mean())}
    for f, idx in val.groupby("file").indices.items():
        order = idx[np.argsort(-val["ttf_s"].values[idx])]  # chronological
        final_h, false_per_day = alarm_timing(s[order], val["ttf_s"].values[order], threshold, sustain)
        per_file[f] = {"final_alarm_run_starts_h_before_failure": float(final_h),
                       "false_alarm_runs_per_day": float(false_per_day),
                       "alarm_rate_normal": float((s[idx][normal.values[idx]] > threshold).mean())}
    m["per_class"], m["per_file"] = per_class, per_file
    return m


def print_metrics(tag, v):
    print(f"{tag}: AUROC {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}  alarm@normal {v['alarm_rate_normal']:.3f}  "
          f"recall@pre-failure {v['recall_pre_failure']:.3f}  ({v['n_windows']:,} windows)")
    for k, r in v["per_class"].items():
        print(f"  {k}: AUROC {r['auroc']:.3f}  recall {r['recall_pre_failure']:.3f}  alarm@normal {r['alarm_rate_normal']:.3f}")


def test_files_in(test_dir):
    return sorted(glob.glob(os.path.join(test_dir, "test_*.parquet")))


def evaluate_test(model, feat_cols, model_cols, window, threshold, sustain, files, score_dir=None, tag=""):
    """Window + score the preprocessed test files with the frozen model/threshold and run the same
    evaluation as for validation. Optionally writes per-file score CSVs (time, score, alarm, ...)."""
    ws = []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        w = window_features(df, window, feat_cols)
        ws.append(w)
        print(f"[test {i}/{len(files)}] {os.path.basename(f)} (class {df['class'].iloc[0]}): "
              f"{len(df):,} rows -> {len(w):,} windows", flush=True)
    test_w = pd.concat(ws)
    scores = anomaly_score(model, test_w[model_cols].to_numpy(dtype=np.float32))
    m = evaluate(test_w, scores, threshold, sustain)
    m["files"] = [os.path.basename(f) for f in files]
    m["file_class"] = {f: str(c) for f, c in test_w.groupby("file")["class"].first().items()}
    m["class_source"] = "inferred by preprocess.py from the terminal code in the last 10 min"
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)
        test_w = test_w.assign(score=scores, alarm=scores > threshold)
        for f, part in test_w.groupby("file"):
            cols = ["score", "alarm"] + [c for c in ("ttf_s", "label", "error_active", "non_welding") if c in part.columns]
            part[cols].to_csv(os.path.join(score_dir, f"{f}_{tag}.csv"))
    return m


# ------------------------------------------------------------- inference
def load_bundle(path):
    return joblib.load(path)


def score_frame(bundle, df):
    """Preprocessed 1 Hz frame (as written by preprocess.py) -> DataFrame[time, score, alarm]."""
    w = window_features(df, bundle["window"], bundle["feature_cols"])
    X = w[bundle["model_cols"]].to_numpy(dtype=np.float32)
    s = anomaly_score(bundle["model"], X)
    out = pd.DataFrame({"score": s, "alarm": s > bundle["threshold"]}, index=w.index)
    for c in ("ttf_s", "label", "error_active", "non_welding"):
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
    ap.add_argument("--exclude-non-welding", action="store_true", help="drop cap-dressing windows from training")
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
                          b["sustain"], test_files, os.path.join(args.model_dir, "scores"), args.model)
        print_metrics("test", m)
        dst = os.path.join(args.model_dir, f"baseline_{args.model}_test_metrics.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump({"bundle": bundle_path, "created": b.get("created"), "threshold": b["threshold"],
                       "window": b["window"], "sustain": b["sustain"], "test": m,
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

    t0 = time.time()
    feat_cols, train_w, val_w = None, [], []
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        feat_cols = feat_cols or feature_columns(df)
        w = window_features(df, args.window, feat_cols)
        (val_w if f in val_files else train_w).append(w)
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {len(df):,} rows -> {len(w):,} windows", flush=True)
    train_w, val_w = pd.concat(train_w), pd.concat(val_w) if val_w else None
    print(f"windowed in {time.time() - t0:.0f}s")

    model_cols = model_input_columns(feat_cols)
    normal = (train_w["label"] == 0) & (train_w["error_active"] == 0)
    if args.exclude_non_welding:
        normal &= train_w["non_welding"] == 0
    fit = train_w[normal]
    if args.max_train_windows and len(fit) > args.max_train_windows:
        fit = fit.sample(args.max_train_windows, random_state=args.seed)
    X = fit[model_cols].to_numpy(dtype=np.float32)
    print(f"fitting {args.model} on {len(X):,} normal windows x {X.shape[1]} features")
    model = build_model(args.model, args.seed).fit(X)
    train_scores = anomaly_score(model, X)
    threshold = float(np.quantile(train_scores, args.threshold_q))

    metrics = {"train": {"n_fit_windows": int(len(X)), "threshold": threshold,
                         "score_mean": float(train_scores.mean()), "score_std": float(train_scores.std())}}
    if val_w is not None:
        val_scores = anomaly_score(model, val_w[model_cols].to_numpy(dtype=np.float32))
        metrics["val"] = evaluate(val_w, val_scores, threshold, args.sustain)
        print_metrics("val", metrics["val"])
    if test_files:
        metrics["test"] = evaluate_test(model, feat_cols, model_cols, args.window, threshold, args.sustain, test_files)
        print_metrics("test", metrics["test"])
    else:
        print(f"no test files in {test_dir} - skipped test evaluation")

    os.makedirs(args.model_dir, exist_ok=True)
    scaler_path = os.path.join(args.data_dir, "scaler.json")
    config_path = os.path.join(args.data_dir, "preprocess_config.json")
    bundle = {
        "model": model, "model_type": args.model, "feature_cols": feat_cols, "model_cols": model_cols,
        "window": args.window, "threshold": threshold, "threshold_q": args.threshold_q, "sustain": args.sustain,
        # per-feature reference (median of the normal training windows): the serving layer replaces one
        # feature at a time with it to attribute a score to individual features
        "feature_reference": np.median(X, axis=0).astype(float).tolist(),
        "feature_scale": X.std(axis=0).astype(float).tolist(),
        "scaler": json.load(open(scaler_path, encoding="utf-8")) if os.path.exists(scaler_path) else None,
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
