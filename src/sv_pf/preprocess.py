# src/sv_pf/preprocess.py
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sv_pf.config import DATA_RAW, DATA_PROCESSED, PERIODS


def load_raw(path):
    """Charge un CSV brut yfinance, indexé par date et trié."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def check_raw(df, name=""):
    """Vérifie l'intégrité des données brutes."""
    assert df.index.is_unique, f"{name}: dates dupliquées"
    assert df.index.is_monotonic_increasing, f"{name}: dates non triées"
    ohlc = df[["Open", "High", "Low", "Close"]]
    assert not ohlc.isna().any().any(), f"{name}: NaN présents"
    assert (df["Close"] > 0).all(), f"{name}: Close <= 0"

    bad_high = df["High"] < df[["Open", "Close"]].max(axis=1)
    bad_low = df["Low"] > df[["Open", "Close"]].min(axis=1)
    n_bad = int((bad_high | bad_low).sum())
    if n_bad:
        warnings.warn(f"{name}: {n_bad} séances avec OHLC incohérent")
    n_flat = int((df["High"] == df["Low"]).sum())
    if n_flat:
        warnings.warn(f"{name}: {n_flat} séances avec High == Low")


def compute_returns(df):
    out = df[["Open", "High", "Low", "Close"]].copy()
    out.insert(0, "y", 100.0 * np.log(df["Close"]).diff())
    out["range_ok"] = (df["High"] > df["Low"])  # False si High == Low
    return out.dropna(subset=["y"])


def preprocess_period(name, raw_dir=DATA_RAW, out_dir=DATA_PROCESSED):
    df = load_raw(Path(raw_dir) / f"sp500_{name}.csv")
    check_raw(df, name)
    out = compute_returns(df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"returns_{name}.csv")
    return out


def load_returns(name, out_dir=DATA_PROCESSED):
    """Retourne y comme array float64 1D, prêt pour particles."""
    df = pd.read_csv(Path(out_dir) / f"returns_{name}.csv", index_col="Date", parse_dates=True)
    return df["y"].to_numpy(dtype=np.float64)


if __name__ == "__main__":
    for name in PERIODS:
        out = preprocess_period(name)
        print(f"{name}: {len(out)} rendements, {out.index.min().date()} -> {out.index.max().date()}")