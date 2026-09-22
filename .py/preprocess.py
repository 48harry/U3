"""
RSW train-set preprocessing for anomaly-detection model training.

Every rule below comes from eda_out/ (72 files) and the paper's pre-processing section:

  gaps        1 Hz grid; whole rows drop out. 99 % of gaps are <= 35 s, so gaps up to
              --gap-fill-limit are forward-filled; longer ones break the series into
              segments (rows dropped) so windows never straddle a hole.
  discard     files with missing rate > --max-missing-rate (paper: 40 %). That is the 8
              E02 files at 64-84 %, including E02_2 (10 min "demo" file).
  outliers    c16 (sheet-thickness setpoint) > 5 or <= 0 = cap dressing / changing, not
              welding. Sensor values in those rows are blanked and forward-filled (paper),
              and the rows are flagged `non_welding` so the model can drop or keep them.
  counters    c11 / c12 are cumulative and gun-specific -> replaced by per-second deltas and
              a 10-min rolling weld count (welding-behaviour pattern, paper Fig. 8).
  c10         on/off -> 1/0.
  static      c7, c8, c9 are constant within a file (gun attributes). Kept as-is; they act as
              gun identity after scaling. Drop them with --drop-static if the detector should
              not be able to tell guns apart.
  error       a *state* column: codes persist for hours. Kept as `error_code`, plus
              `error_active` (any non-zero code), a 10-min rolling share (readme: minor-error
              frequency as precursor feature) and `terminal_code` = the code that marks the
              failure for this class (E01 E012, E02 E016, E03 E028, E04 E029 - 18/18, 17/18,
              18/18, 18/18 files).
  label       `ttf_s` seconds-to-failure and `label` = 1 inside the last --pre-failure-window
              seconds. For an unsupervised detector train on label == 0 rows (and optionally
              error_active == 0); the scaler is fitted on those rows only.
  scaling     z-score (paper) fitted on normal rows, global across files (--scale global) or
              per file (--scale per-file, removes gun-level offsets). Parameters are written
              to scaler.json so test files get the identical transform.

Outputs (in --out-dir):
  <file>.parquet (or .csv)   one row per second (or per --resample bin), float32
  manifest.csv               per-file rows / segments / drop reason
  scaler.json                mean/std per scaled column
  preprocess_config.json     the arguments used

Usage:
    python preprocess.py                         # ./train -> ./preprocessed
    python preprocess.py --resample 10s          # 10-second bins (mean of sensors, max of flags)
    python preprocess.py --scale per-file --drop-static
    python preprocess.py --pattern "E04_*" --max-files 3     # quick check
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root

SENSOR_COLS = [f"c{i}" for i in range(1, 20)]
COUNTER_COLS = ["c11", "c12"]
STATIC_COLS = ["c7", "c8", "c9"]
BINARY_COL = "c10"
# continuous sensors that get blanked in non-welding rows and z-scored
VALUE_COLS = [c for c in SENSOR_COLS if c not in COUNTER_COLS + [BINARY_COL]]
CLASSES = ["E01", "E02", "E03", "E04"]
TERMINAL_CODE = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
ROLL = "600s"  # 10-minute rolling window for activity / error-share features

FLAG_COLS = ["error_active", "terminal_code", "non_welding", "label"]
# derived continuous columns that get z-scored; the 10-min shares are already in [0, 1] and stay raw
DERIVED_CONT = ["welds_delta", "welds_10min", "pos_delta"]


# ------------------------------------------------------------------- loading
def load_file(path):
    """CSV -> 1 Hz grid DataFrame (NaN rows where the panel is missing), present mask."""
    df = pd.read_csv(path, low_memory=False, dtype={"c10": "string", "error": "string"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    num = [c for c in SENSOR_COLS if c != BINARY_COL]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype("float32")
    raw_index = df.index
    df = df.resample("s").asfreq()
    present = df.index.isin(raw_index)
    return df, present


# -------------------------------------------------------------- per file
def preprocess_file(path, args):
    name = os.path.splitext(os.path.basename(path))[0]
    cls, gun = name.split("_")
    df, present = load_file(path)
    missing_rate = 1 - present.mean()
    if missing_rate > args.max_missing_rate:
        return None, {"file": name, "class": cls, "gun": int(gun), "rows": 0, "segments": 0,
                      "missing_rate": missing_rate, "dropped": f"missing rate {missing_rate:.0%}"}
    t_end = df.index[-1]

    # --- error state: carry the code across gaps, then flags
    code = df["error"].where(present).ffill().fillna("0").astype(str)
    out = pd.DataFrame(index=df.index)
    out["error_code"] = code
    out["error_active"] = (code != "0").astype("float32")
    out["terminal_code"] = (code == TERMINAL_CODE[cls]).astype("float32")

    # --- sensors: c10 -> 0/1, non-welding rows blanked (paper outlier rule), then gap fill
    x = df[SENSOR_COLS].copy()
    x[BINARY_COL] = (x[BINARY_COL] == "on").astype("float32").where(present)
    # 1) gaps: fill up to the limit, drop what is left (long block-outs)
    x = x.ffill(limit=args.gap_fill_limit).bfill(limit=args.gap_fill_limit)
    keep = x.notna().all(axis=1)
    x, out = x[keep], out[keep]
    if len(x) == 0:
        return None, {"file": name, "class": cls, "gun": int(gun), "rows": 0, "segments": 0,
                      "missing_rate": missing_rate, "dropped": "no rows left after gap filtering"}
    # 2) non-welding rows (cap dressing / changing): blank the sensors and carry the last
    #    welding value across, however long the block is (paper). Rows stay, flagged.
    non_welding = (x["c16"] > args.outlier_hi) | (x["c16"] <= args.outlier_lo)
    out["non_welding"] = non_welding.astype("float32")
    x.loc[non_welding, VALUE_COLS + [BINARY_COL]] = np.nan  # counters keep counting during cap dressing
    x = x.ffill().bfill()

    # --- contiguous segments (a break wherever consecutive kept rows are > 1 s apart)
    step = x.index.to_series().diff().dt.total_seconds().fillna(1)
    out["segment_id"] = (step > 1).cumsum().astype("int32")

    # --- counters -> deltas (per segment, non-negative), rolling activity
    for c, new in [("c11", "welds_delta"), ("c12", "pos_delta")]:
        d = x[c].groupby(out["segment_id"]).diff().fillna(0).clip(lower=0)
        out[new] = d.astype("float32")
    out["welds_10min"] = out["welds_delta"].rolling(ROLL, min_periods=1).sum().astype("float32")
    out["weld_duty_10min"] = (x["c2"] > 0).astype("float32").rolling(ROLL, min_periods=1).mean().astype("float32")
    out["error_share_10min"] = out["error_active"].rolling(ROLL, min_periods=1).mean().astype("float32")

    # --- sensor columns
    sensor_keep = [c for c in VALUE_COLS if not (args.drop_static and c in STATIC_COLS)] + [BINARY_COL]
    for c in sensor_keep:
        out[c] = x[c].astype("float32")

    # --- time-of-day (shift pattern, paper Fig. 8) and time-to-failure label
    hour = out.index.hour + out.index.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    out["dow"] = out.index.dayofweek.astype("int8")
    out["ttf_s"] = (t_end - out.index).total_seconds().astype("int64")
    out["label"] = (out["ttf_s"] <= args.pre_failure_window).astype("float32")

    if args.resample:
        out = resample(out, args.resample)

    out.insert(0, "gun", np.int16(gun))
    out.insert(0, "class", cls)
    out.insert(0, "file", name)
    info = {"file": name, "class": cls, "gun": int(gun), "rows": len(out),
            "segments": int(out["segment_id"].nunique()), "missing_rate": missing_rate,
            "non_welding_share": float(out["non_welding"].mean()), "label_rows": int(out["label"].sum()),
            "start": out.index[0], "end": out.index[-1], "dropped": ""}
    return out, info


def resample(out, rule):
    """Aggregate 1 Hz rows into bins: mean for continuous, max for flags, sum for deltas, last for time fields."""
    agg = {c: "mean" for c in out.columns}
    agg.update({c: "max" for c in FLAG_COLS})
    agg.update({"welds_delta": "sum", "pos_delta": "sum", "segment_id": "min", "dow": "last",
                "ttf_s": "min", "error_code": "last"})
    grouped = out.groupby([out["segment_id"], pd.Grouper(freq=rule)]).agg(agg)
    grouped = grouped.droplevel(0)
    grouped = grouped[grouped["ttf_s"].notna()]
    for c in grouped.columns:
        if c not in ("error_code", "segment_id", "dow", "ttf_s"):
            grouped[c] = grouped[c].astype("float32")
    grouped["segment_id"] = grouped["segment_id"].astype("int32")
    grouped["dow"] = grouped["dow"].astype("int8")
    grouped["ttf_s"] = grouped["ttf_s"].astype("int64")
    return grouped


# -------------------------------------------------------------- scaling
class StreamingStats:
    """Running mean/variance of the columns to scale, fitted on normal rows only."""

    def __init__(self, cols):
        self.cols, self.n, self.s, self.ss = cols, 0, np.zeros(len(cols)), np.zeros(len(cols))

    def add(self, df):
        v = df[self.cols].to_numpy(dtype=np.float64)
        self.n += len(v)
        self.s += v.sum(axis=0)
        self.ss += (v ** 2).sum(axis=0)

    def params(self):
        mean = self.s / max(self.n, 1)
        std = np.sqrt(np.maximum(self.ss / max(self.n, 1) - mean ** 2, 0))
        std = np.where(std < 1e-6, 1.0, std)  # constant columns stay as (x - mean)
        return {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(self.cols, mean, std)}


def normal_rows(df):
    return df[(df["label"] == 0) & (df["error_active"] == 0)]


def apply_scaler(df, params):
    for c, p in params.items():
        if c in df.columns:
            df[c] = ((df[c] - p["mean"]) / p["std"]).astype("float32")
    return df


# -------------------------------------------------------------- I/O
def writer(fmt):
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
            return "parquet", lambda df, p: df.to_parquet(p + ".parquet")
        except ImportError:
            print("pyarrow not installed -> writing CSV")
    return "csv", lambda df, p: df.to_csv(p + ".csv")


def reader(fmt):
    if fmt == "parquet":
        return lambda p: pd.read_parquet(p + ".parquet")
    return lambda p: pd.read_csv(p + ".csv", index_col=0, parse_dates=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=os.path.join(PROJECT_ROOT, "train"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "preprocessed"))
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-missing-rate", type=float, default=0.40, help="paper: discard guns above this")
    ap.add_argument("--gap-fill-limit", type=int, default=60, help="seconds; longer gaps split segments")
    ap.add_argument("--outlier-hi", type=float, default=5.0, help="c16 above this = non-welding")
    ap.add_argument("--outlier-lo", type=float, default=0.0, help="c16 at/below this = non-welding")
    ap.add_argument("--pre-failure-window", type=int, default=3600, help="seconds before failure labelled 1")
    ap.add_argument("--resample", default=None, help="pandas offset, e.g. 10s or 1min; default keeps 1 Hz")
    ap.add_argument("--scale", choices=["global", "per-file", "none"], default="global")
    ap.add_argument("--drop-static", action="store_true", help="drop c7/c8/c9 (constant per gun)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.train_dir, args.pattern)))[: args.max_files]
    if not paths:
        raise SystemExit(f"no files matching {args.pattern!r} in {os.path.abspath(args.train_dir)} — pass --train-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    fmt, write = writer(args.format)
    read = reader(fmt)

    scale_cols = [c for c in VALUE_COLS if not (args.drop_static and c in STATIC_COLS)] + DERIVED_CONT
    stats = StreamingStats(scale_cols)
    manifest, t0 = [], time.time()

    # pass 1: clean, derive, write; accumulate global scaler stats on normal rows
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
        out, info = preprocess_file(p, args)
        manifest.append(info)
        if out is None:
            print("  dropped:", info["dropped"])
            continue
        if args.scale == "per-file":
            local = StreamingStats(scale_cols)
            local.add(normal_rows(out))
            out = apply_scaler(out, local.params())
        elif args.scale == "global":
            stats.add(normal_rows(out))
        write(out, os.path.join(args.out_dir, info["file"]))

    # pass 2: apply the global scaler
    params = {}
    if args.scale == "global" and stats.n > 0:
        params = stats.params()
        for info in manifest:
            if info["dropped"]:
                continue
            p = os.path.join(args.out_dir, info["file"])
            write(apply_scaler(read(p), params), p)
    elif args.scale == "per-file":
        params = {"note": "per-file z-score; fit the same way on each test file (normal rows)"}

    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    with open(os.path.join(args.out_dir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump({"scale": args.scale, "columns": scale_cols, "params": params, "n_fit_rows": stats.n}, f, indent=2)
    with open(os.path.join(args.out_dir, "preprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    m = pd.DataFrame(manifest)
    kept = m[m["dropped"] == ""]
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out_dir} ({fmt})")
    print(f"files kept {len(kept)}/{len(m)}, rows {int(kept['rows'].sum()):,}, "
          f"label=1 rows {int(kept['label_rows'].sum()):,}")
    if (m["dropped"] != "").any():
        print("dropped:", ", ".join(f"{r.file} ({r.dropped})" for r in m[m["dropped"] != ""].itertuples()))


if __name__ == "__main__":
    main()
