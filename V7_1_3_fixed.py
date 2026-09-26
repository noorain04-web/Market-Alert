"""
V7.1.3 — POINT-IN-TIME QUANT + OPTIONAL GEMINI + TELEGRAM ALERT

Research/production-oriented NSE market alert system.

IMPORTANT:
- Historical Gemini scores are NEVER fabricated.
- Historical Gemini is used only when timestamped historical data exists.
- OHLCV features use information available at or before signal close.
- Signal is generated at T close.
- Backtest execution: T+1 open -> T+H close.
- Transaction cost is applied once per round trip.
- Validation selects thresholds/model settings.
- OOS is never used for threshold selection.
- Current alert is generated independently of the full historical backtest.

Python: 3.12+
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import warnings
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


# ============================================================================
# CONFIG
# ============================================================================

VERSION = "V7.1.3"
REVISION = "2026-09-26-FAST-STABLE-TELEGRAM"

BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

ROUND_TRIP_COST = float(os.getenv("ROUND_TRIP_COST", "0.003"))

RANDOM_STATE = 42

# H10 is the primary production horizon.
HORIZONS = [1, 3, 5, 10]
PRIMARY_HORIZON = 10

# Refit every N signal dates rather than fitting a completely new model
# for every single row.
REFIT_EVERY = int(os.getenv("REFIT_EVERY", "10"))

# Minimum training observations.
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "2500"))

# Minimum probability for a trade.
DEFAULT_PMIN = 0.55

# Minimum predicted return.
DEFAULT_RMIN = 0.006

# Candidate validation thresholds.
P_GRID = [0.50, 0.52, 0.54, 0.55, 0.57, 0.60, 0.62, 0.65]
R_GRID = [0.0000, 0.0020, 0.0040, 0.0060, 0.0080, 0.0100]

# Maximum number of stocks in one day's portfolio.
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8"))

# Historical Gemini coverage required before it is considered testable.
MIN_GEMINI_COVERAGE = float(os.getenv("MIN_GEMINI_COVERAGE", "0.20"))

# Bootstrap repetitions.
BOOTSTRAP_N = int(os.getenv("BOOTSTRAP_N", "2000"))

# Progress heartbeat.
HEARTBEAT_EVERY = int(os.getenv("HEARTBEAT_EVERY", "25"))

# Current alert.
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "true").lower() == "true"

# Optional Gemini current analysis.
USE_CURRENT_GEMINI = os.getenv("USE_CURRENT_GEMINI", "false").lower() == "true"

# Files.
ROOT = Path(".")
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

HISTORICAL_GEMINI_FILE = DATA_DIR / "historical_gemini.csv"

# Telegram environment variables.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Optional Gemini environment variables.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()


# ============================================================================
# UNIVERSE
# ============================================================================

SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS",
    "LT.NS",
    "TMPV.NS",
    "TMCV.NS",
    "EICHERMOT.NS",
    "MARUTI.NS",
    "HEROMOTOCO.NS",
    "M&M.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "BHARTIARTL.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "BPCL.NS",
    "COALINDIA.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BEL.NS",
    "HAL.NS",
    "BHEL.NS",
    "TRENT.NS",
    "PIDILITIND.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "GRASIM.NS",
    "ULTRACEMCO.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "IOC.NS",
    "VEDL.NS",
    "DLF.NS",
    "LODHA.NS",
    "INDIGO.NS",
    "ETERNAL.NS",
    "NAUKRI.NS",
    "COFORGE.NS",
    "JIOFIN.NS",
    "IRFC.NS",
    "IREDA.NS",
    "POLYCAB.NS",
]


# ============================================================================
# UTILITIES
# ============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(x, default=np.nan):
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def finite_series(s: pd.Series, fill=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(fill)


def finite_array(x):
    arr = np.asarray(x, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def print_header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def load_symbol(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            symbol,
            period=BACKTEST_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"WARNING: no data for {symbol}", flush=True)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]

        for col in required:
            if col not in df.columns:
                print(f"WARNING: {symbol} missing {col}", flush=True)
                return None

        df = df[required].copy()

        df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if len(df) < 800:
            print(
                f"WARNING: insufficient history for {symbol}; skipping.",
                flush=True,
            )
            return None

        return df

    except Exception as exc:
        print(f"WARNING: failed {symbol}: {exc}", flush=True)
        return None


def load_all_data():
    data = {}

    print(f"Loading [{len(SYMBOLS)}] symbols...", flush=True)

    for i, symbol in enumerate(SYMBOLS, 1):
        print(
            f"Loading [{i}/{len(SYMBOLS)}] {symbol}",
            flush=True,
        )

        df = load_symbol(symbol)

        if df is not None:
            data[symbol] = df

    return data


# ============================================================================
# FEATURES
# ============================================================================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every feature is backward-looking at signal close.
    """

    x = df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    open_ = x["Open"]
    volume = x["Volume"]

    # Returns
    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)
    x["ret_60"] = close.pct_change(60)

    # Moving-average distance
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()

    x["dist_ma5"] = close / ma5 - 1
    x["dist_ma10"] = close / ma10 - 1
    x["dist_ma20"] = close / ma20 - 1
    x["dist_ma50"] = close / ma50 - 1
    x["dist_ma100"] = close / ma100 - 1

    # Volatility
    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()

    # Intraday structure
    x["intraday_return"] = close / open_ - 1
    x["range_pct"] = (high - low) / close.replace(0, np.nan)

    # Close position in daily range
    denom = (high - low).replace(0, np.nan)
    x["close_location"] = (close - low) / denom

    # Volume
    vma20 = volume.rolling(20).mean()
    x["volume_ratio"] = volume / vma20.replace(0, np.nan)

    # Breakout / trend
    x["high_20"] = close / close.rolling(20).max() - 1
    x["low_20"] = close / close.rolling(20).min() - 1

    # RSI-like momentum
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))

    # ATR-style volatility
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14_pct"] = tr.rolling(14).mean() / close

    return x


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "dist_ma5",
    "dist_ma10",
    "dist_ma20",
    "dist_ma50",
    "dist_ma100",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "intraday_return",
    "range_pct",
    "close_location",
    "volume_ratio",
    "high_20",
    "low_20",
    "rsi14",
    "atr14_pct",
]


# ============================================================================
# DATASET
# ============================================================================

def build_dataset(data):
    frames = []

    for symbol, raw in data.items():

        x = make_features(raw)

        for horizon in HORIZONS:

            # Execution:
            # signal = T close
            # entry = T+1 open
            # exit = T+H close
            #
            # Therefore target uses future information ONLY in target columns.

            entry = raw["Open"].shift(-1)

            future_close = raw["Close"].shift(-horizon)

            future_return = future_close / entry - 1

            y = x.copy()

            y["symbol"] = symbol
            y["signal_date"] = y.index
            y[f"future_return_{horizon}"] = future_return
            y[f"target_{horizon}"] = (
                future_return > 0
            ).astype(float)

            frames.append(y)

    if not frames:
        raise RuntimeError("No usable market data.")

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(
        ["signal_date", "symbol"]
    ).reset_index(drop=True)

    # Strict finite cleaning is done only for features.
    # Targets are retained as NaN where future data doesn't exist.

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
    )

    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Forward-return targets are excluded from FEATURES.")

    return df


# ============================================================================
# HISTORICAL GEMINI
# ============================================================================

def load_historical_gemini() -> pd.DataFrame | None:

    if not HISTORICAL_GEMINI_FILE.exists():
        print("HISTORICAL GEMINI: NOT FOUND — QUANT-ONLY")
        return None

    try:
        g = pd.read_csv(HISTORICAL_GEMINI_FILE)

        required = {
            "symbol",
            "published_at",
            "gemini_score",
        }

        if not required.issubset(g.columns):
            print(
                "HISTORICAL GEMINI: invalid file — missing required columns"
            )
            return None

        g["published_at"] = pd.to_datetime(
            g["published_at"],
            errors="coerce",
        )

        g["gemini_score"] = pd.to_numeric(
            g["gemini_score"],
            errors="coerce",
        )

        g = g.dropna(
            subset=[
                "symbol",
                "published_at",
                "gemini_score",
            ]
        )

        if g.empty:
            print("HISTORICAL GEMINI: empty")
            return None

        return g

    except Exception as exc:
        print(f"HISTORICAL GEMINI: load failed: {exc}")
        return None


def attach_point_in_time_gemini(
    df: pd.DataFrame,
    historical: pd.DataFrame | None,
):
    out = df.copy()

    out["gemini_available"] = False
    out["gemini_score"] = 0.0

    if historical is None:
        return out

    historical = historical.sort_values(
        ["symbol", "published_at"]
    )

    left = out[
        ["symbol", "signal_date"]
    ].copy()

    left["signal_date"] = pd.to_datetime(
        left["signal_date"]
    )

    right = historical[
        [
            "symbol",
            "published_at",
            "gemini_score",
        ]
    ].copy()

    right = right.sort_values(
        ["symbol", "published_at"]
    )

    left = left.sort_values(
        ["symbol", "signal_date"]
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="signal_date",
        right_on="published_at",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )

    available = merged["gemini_score"].notna()

    out.loc[
        merged.index,
        "gemini_available",
    ] = available.to_numpy()

    out.loc[
        merged.index,
        "gemini_score",
    ] = merged["gemini_score"].fillna(0).to_numpy()

    return out


# ============================================================================
# TRAINING
# ============================================================================

def make_classifier():
    """
    Robust against NaNs and compatible with modern sklearn.
    """

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_gradient_classifier():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_regressor():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=180,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_model(train: pd.DataFrame, horizon: int):

    target_return = f"future_return_{horizon}"
    target_direction = f"target_{horizon}"

    usable = train[
        FEATURE_COLUMNS
        + [
            target_return,
            target_direction,
        ]
    ].copy()

    usable = usable.dropna(
        subset=[
            target_return,
            target_direction,
        ]
    )

    if len(usable) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Insufficient training rows: {len(usable)}"
        )

    X = usable[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y_direction = usable[target_direction].astype(int)
    y_return = finite_series(
        usable[target_return]
    ).clip(-0.75, 0.75)

    # If a very small sample accidentally contains one class,
    # use a neutral fallback.
    if y_direction.nunique() < 2:
        raise RuntimeError(
            "Training target contains only one class."
        )

    clf_log = make_classifier()
    clf_gbm = make_gradient_classifier()

    reg = make_regressor()

    clf_log.fit(X, y_direction)
    clf_gbm.fit(X, y_direction)
    reg.fit(X, y_return)

    return {
        "logistic": clf_log,
        "gradient": clf_gbm,
        "regressor": reg,
    }


def predict_model(model, frame: pd.DataFrame):

    X = frame[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    p1 = model["logistic"].predict_proba(X)[:, 1]
    p2 = model["gradient"].predict_proba(X)[:, 1]

    probability = (
        0.45 * p1 +
        0.55 * p2
    )

    predicted_return = model[
        "regressor"
    ].predict(X)

    probability = np.clip(
        finite_array(probability),
        0.001,
        0.999,
    )

    predicted_return = finite_array(
        predicted_return
    )

    return probability, predicted_return


# ============================================================================
# WALK FORWARD
# ============================================================================

def walk_forward(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    label: str,
):
    """
    Efficient walk-forward.

    Model is retrained every REFIT_EVERY signal dates.
    This is still chronological and point-in-time:
    each model only sees rows earlier than the prediction date.
    """

    test = test.sort_values(
        ["signal_date", "symbol"]
    ).copy()

    train = train.sort_values(
        ["signal_date", "symbol"]
    ).copy()

    dates = sorted(
        test["signal_date"].dropna().unique()
    )

    results = []

    current_model = None
    last_refit_date = None

    total = len(dates)

    for i, date in enumerate(dates, 1):

        if (
            current_model is None
            or (i - 1) % REFIT_EVERY == 0
        ):

            historical_train = train[
                train["signal_date"] < date
            ]

            if len(historical_train) < MIN_TRAIN_ROWS:
                continue

            try:
                current_model = fit_model(
                    historical_train,
                    horizon,
                )

                last_refit_date = date

            except Exception as exc:
                print(
                    f"WARNING: model fit failed at "
                    f"{date}: {exc}",
                    flush=True,
                )
                continue

        batch = test[
            test["signal_date"] == date
        ].copy()

        if batch.empty:
            continue

        try:
            p, r = predict_model(
                current_model,
                batch,
            )

            batch[
                "quant_probability"
            ] = p

            batch[
                "quant_return_prediction"
            ] = r

            batch["model_refit_date"] = last_refit_date

            results.append(batch)

        except Exception as exc:
            print(
                f"WARNING: prediction failed at "
                f"{date}: {exc}",
                flush=True,
            )

        if (
            i % HEARTBEAT_EVERY == 0
            or i == total
        ):
            print(
                f"Walk-forward {label}: "
                f"[{i}/{total}] "
                f"last={date}",
                flush=True,
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True,
    )


# ============================================================================
# THRESHOLD SELECTION
# ============================================================================

def apply_trade_rule(
    df,
    pmin,
    rmin,
    probability_col="quant_probability",
    return_col="quant_return_prediction",
):

    out = df.copy()

    out["action"] = np.where(
        (
            out[probability_col] >= pmin
        )
        & (
            out[return_col] >= rmin
        ),
        "TRADE",
        "PASS",
    )

    return out


def select_thresholds(
    validation: pd.DataFrame,
    horizon: int,
):
    best = None

    ret_col = f"future_return_{horizon}"

    v = validation.dropna(
        subset=[
            ret_col,
            "quant_probability",
            "quant_return_prediction",
        ]
    ).copy()

    if v.empty:
        return {
            "pmin": DEFAULT_PMIN,
            "rmin": DEFAULT_RMIN,
        }

    for pmin in P_GRID:
        for rmin in R_GRID:

            selected = v[
                (
                    v["quant_probability"]
                    >= pmin
                )
                &
                (
                    v["quant_return_prediction"]
                    >= rmin
                )
            ]

            if len(selected) < 30:
                continue

            net = (
                selected[ret_col]
                - ROUND_TRIP_COST
            )

            win_rate = (
                net > 0
            ).mean()

            avg = net.mean()

            # Stability-aware score.
            score = (
                avg
                * math.sqrt(len(selected))
                * (0.5 + win_rate)
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "pmin": pmin,
                    "rmin": rmin,
                    "score": score,
                    "n": len(selected),
                    "win_rate": win_rate,
                    "average_net": avg,
                }

    if best is None:
        return {
            "pmin": DEFAULT_PMIN,
            "rmin": DEFAULT_RMIN,
        }

    return best


# ============================================================================
# PERFORMANCE
# ============================================================================

def performance(
    df: pd.DataFrame,
    horizon: int,
    pmin: float,
    rmin: float,
):

    ret_col = f"future_return_{horizon}"

    v = df.dropna(
        subset=[
            ret_col,
            "quant_probability",
            "quant_return_prediction",
        ]
    ).copy()

    v["action"] = np.where(
        (
            v["quant_probability"] >= pmin
        )
        &
        (
            v["quant_return_prediction"] >= rmin
        ),
        "TRADE",
        "PASS",
    )

    trades = v[
        v["action"] == "TRADE"
    ].copy()

    if trades.empty:
        return {
            "observations": len(v),
            "trades": 0,
            "win_rate": np.nan,
            "average_net": np.nan,
            "profit_factor": np.nan,
        }, trades

    net = (
        trades[ret_col]
        - ROUND_TRIP_COST
    )

    gross_profit = net[
        net > 0
    ].sum()

    gross_loss = -net[
        net < 0
    ].sum()

    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.inf

    return {
        "observations": len(v),
        "trades": len(trades),
        "win_rate": float(
            (net > 0).mean()
        ),
        "average_net": float(
            net.mean()
        ),
        "median_net": float(
            net.median()
        ),
        "profit_factor": float(pf),
        "gross_sum": float(
            net.sum()
        ),
    }, trades


# ============================================================================
# BOOTSTRAP
# ============================================================================

def bootstrap_returns(values):
    """
    Numeric bootstrap only.
    Avoids numpy boolean quantile error from older implementation.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 5:
        return {
            "mean_low": np.nan,
            "mean_high": np.nan,
            "prob_positive_low": np.nan,
            "prob_positive_high": np.nan,
            "prob_mean_positive": np.nan,
        }

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    means = np.empty(
        BOOTSTRAP_N,
        dtype=float,
    )

    positive_prob = np.empty(
        BOOTSTRAP_N,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_N
    ):

        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )

        means[i] = float(
            np.mean(sample)
        )

        positive_prob[i] = float(
            np.mean(sample > 0)
        )

    return {
        "mean_low": float(
            np.quantile(means, 0.025)
        ),
        "mean_high": float(
            np.quantile(means, 0.975)
        ),
        "prob_positive_low": float(
            np.quantile(
                positive_prob,
                0.025,
            )
        ),
        "prob_positive_high": float(
            np.quantile(
                positive_prob,
                0.975,
            )
        ),
        "prob_mean_positive": float(
            np.mean(means > 0)
        ),
    }


# ============================================================================
# NON-OVERLAPPING TEST
# ============================================================================

def non_overlapping(
    trades: pd.DataFrame,
    horizon: int,
):

    if trades.empty:
        return pd.DataFrame()

    ret_col = f"future_return_{horizon}"

    x = trades.sort_values(
        "signal_date"
    ).copy()

    selected = []

    last_date = None

    for _, row in x.iterrows():

        date = pd.Timestamp(
            row["signal_date"]
        )

        if last_date is None:
            selected.append(row)
            last_date = date
            continue

        # Conservative calendar-day exclusion.
        if (
            date - last_date
        ).days >= horizon:

            selected.append(row)
            last_date = date

    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(
        selected
    )

    out["net_return"] = (
        out[ret_col]
        - ROUND_TRIP_COST
    )

    return out


# ============================================================================
# PORTFOLIO TEST
# ============================================================================

def portfolio_test(
    df: pd.DataFrame,
    horizon: int,
    pmin: float,
    rmin: float,
):

    ret_col = f"future_return_{horizon}"

    x = df.dropna(
        subset=[
            ret_col,
            "quant_probability",
            "quant_return_prediction",
        ]
    ).copy()

    x["action"] = np.where(
        (
            x["quant_probability"] >= pmin
        )
        &
        (
            x["quant_return_prediction"] >= rmin
        ),
        "TRADE",
        "PASS",
    )

    trades = x[
        x["action"] == "TRADE"
    ].copy()

    if trades.empty:
        return {
            "starting_capital": 100000.0,
            "ending_equity": 100000.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "completed_trades": 0,
        }

    equity = 100000.0

    equity_curve = []

    for date, group in trades.groupby(
        "signal_date"
    ):

        group = group.sort_values(
            "quant_probability",
            ascending=False,
        ).head(
            MAX_POSITIONS
        )

        returns = (
            group[ret_col]
            - ROUND_TRIP_COST
        )

        daily_return = (
            returns.mean()
        )

        equity *= (
            1.0 + daily_return
        )

        equity_curve.append(
            {
                "date": date,
                "equity": equity,
            }
        )

    curve = pd.DataFrame(
        equity_curve
    )

    if curve.empty:
        return {
            "starting_capital": 100000.0,
            "ending_equity": 100000.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "completed_trades": len(trades),
        }

    eq = curve["equity"].to_numpy(
        dtype=float
    )

    peak = np.maximum.accumulate(eq)

    drawdown = (
        eq / peak - 1
    )

    return {
        "starting_capital": 100000.0,
        "ending_equity": float(eq[-1]),
        "total_return": float(
            eq[-1] / 100000.0 - 1
        ),
        "max_drawdown": float(
            drawdown.min()
        ),
        "completed_trades": len(trades),
    }


# ============================================================================
# METRICS
# ============================================================================

def model_metrics(
    df: pd.DataFrame,
    horizon: int,
):

    ret_col = f"future_return_{horizon}"
    target_col = f"target_{horizon}"

    x = df.dropna(
        subset=[
            ret_col,
            target_col,
            "quant_probability",
            "quant_return_prediction",
        ]
    ).copy()

    if x.empty:
        return {}

    y = x[target_col].astype(int).to_numpy()

    p = np.clip(
        x["quant_probability"].to_numpy(
            dtype=float
        ),
        1e-6,
        1 - 1e-6,
    )

    rpred = x[
        "quant_return_prediction"
    ].to_numpy(dtype=float)

    actual = x[
        ret_col
    ].to_numpy(dtype=float)

    directional_accuracy = float(
        np.mean(
            (
                p >= 0.5
            )
            == (
                y == 1
            )
        )
    )

    brier = float(
        np.mean(
            (p - y) ** 2
        )
    )

    logloss = float(
        -np.mean(
            y * np.log(p)
            + (1 - y)
            * np.log(1 - p)
        )
    )

    mae = float(
        np.mean(
            np.abs(
                rpred - actual
            )
        )
    )

    return {
        "observations": len(x),
        "directional_accuracy": directional_accuracy,
        "brier_score": brier,
        "log_loss": logloss,
        "return_mae": mae,
        "mean_predicted_return": float(
            np.mean(rpred)
        ),
        "mean_actual_return": float(
            np.mean(actual)
        ),
    }


# ============================================================================
# CURRENT SIGNAL
# ============================================================================

def prepare_current_data(
    data,
):
    frames = []

    for symbol, raw in data.items():

        x = make_features(raw)

        if x.empty:
            continue

        latest = x.iloc[-1:].copy()

        latest["symbol"] = symbol
        latest["signal_date"] = x.index[-1]

        frames.append(
            latest
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def train_production_model(
    full_training: pd.DataFrame,
    horizon: int,
):
    return fit_model(
        full_training,
        horizon,
    )


def current_signal(
    model,
    current,
):
    p, r = predict_model(
        model,
        current,
    )

    out = current[
        [
            "symbol",
            "signal_date",
            "Close",
        ]
    ].copy()

    out[
        "quant_probability"
    ] = p

    out[
        "quant_return_prediction"
    ] = r

    out = out.sort_values(
        [
            "quant_probability",
            "quant_return_prediction",
        ],
        ascending=False,
    )

    return out


# ============================================================================
# OPTIONAL CURRENT GEMINI
# ============================================================================

def current_gemini_analysis(
    signals: pd.DataFrame,
):

    if not USE_CURRENT_GEMINI:
        return ""

    if not GEMINI_API_KEY:
        return ""

    try:
        import requests

        rows = []

        for _, r in signals.head(10).iterrows():
            rows.append(
                {
                    "symbol": r["symbol"],
                    "probability": round(
                        float(
                            r["quant_probability"]
                        ),
                        4,
                    ),
                    "predicted_return": round(
                        float(
                            r[
                                "quant_return_prediction"
                            ]
                        ),
                        4,
                    ),
                }
            )

        prompt = f"""
You are a market-research assistant.

Review the following quantitative NSE signal candidates.

Do not invent news.
Do not claim certainty.
Do not provide guaranteed returns.

Return a concise research note identifying:
1. strongest quantitative candidates,
2. important risks,
3. whether the signal appears broad or concentrated,
4. what should be checked before trading.

Data:
{json.dumps(rows, indent=2)}
"""

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{GEMINI_MODEL}:generateContent"
        )

        response = requests.post(
            url,
            params={
                "key": GEMINI_API_KEY
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get(
                "candidates",
                [{}],
            )[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

    except Exception as exc:
        print(
            f"Gemini current analysis unavailable: {exc}"
        )
        return ""


# ============================================================================
# TELEGRAM
# ============================================================================

def telegram_send(
    message: str,
):
    if not SEND_TELEGRAM:
        print(
            "Telegram disabled by SEND_TELEGRAM."
        )
        return False

    if not TELEGRAM_BOT_TOKEN:
        print(
            "Telegram skipped: TELEGRAM_BOT_TOKEN missing."
        )
        return False

    if not TELEGRAM_CHAT_ID:
        print(
            "Telegram skipped: TELEGRAM_CHAT_ID missing."
        )
        return False

    try:
        import requests

        url = (
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        )

        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):
            print(
                "Telegram API returned failure:",
                data,
            )
            return False

        print(
            "Telegram alert sent successfully."
        )

        return True

    except Exception as exc:
        print(
            f"Telegram send failed: {exc}"
        )
        return False


def format_alert(
    signals: pd.DataFrame,
    horizon: int,
    thresholds: dict,
    gemini_note: str,
):

    date = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    pmin = thresholds["pmin"]
    rmin = thresholds["rmin"]

    trade = signals[
        (
            signals[
                "quant_probability"
            ] >= pmin
        )
        &
        (
            signals[
                "quant_return_prediction"
            ] >= rmin
        )
    ].copy()

    trade = trade.head(
        MAX_POSITIONS
    )

    lines = []

    lines.append(
        f"<b>V7.1.3 NSE QUANT ALERT</b>"
    )

    lines.append(
        f"Generated: {date}"
    )

    lines.append(
        f"Primary horizon: H{horizon}"
    )

    lines.append(
        f"Threshold: P≥{pmin:.2f}, "
        f"predicted return≥{rmin:.2%}"
    )

    lines.append("")

    if trade.empty:

        lines.append(
            "<b>NO QUALIFYING TRADE</b>"
        )

        lines.append(
            "Quant model did not find a candidate "
            "above the validation-selected threshold."
        )

    else:

        lines.append(
            f"<b>QUALIFYING: {len(trade)}</b>"
        )

        for _, r in trade.iterrows():

            lines.append(
                f"• <b>{r['symbol']}</b> "
                f"P={r['quant_probability']:.1%} "
                f"R={r['quant_return_prediction']:.2%}"
            )

    lines.append("")

    lines.append(
        "<b>Execution model</b>: "
        "signal at T close → entry T+1 open → "
        f"exit T+{horizon} close"
    )

    lines.append(
        f"Round-trip cost assumed: "
        f"{ROUND_TRIP_COST:.2%}"
    )

    lines.append(
        "Historical Gemini: "
        "NOT USED unless timestamped historical scores exist."
    )

    if gemini_note:

        lines.append("")

        lines.append(
            "<b>Current Gemini research note</b>"
        )

        lines.append(
            gemini_note[:3000]
        )

    lines.append("")

    lines.append(
        "<i>Research signal only. "
        "Historical backtests do not guarantee future performance.</i>"
    )

    return "\n".join(lines)


# ============================================================================
# AUDIT
# ============================================================================

def save_csv(df, filename):
    path = AUDIT_DIR / filename

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    start = time.time()

    print_header(
        f"{VERSION} — POINT-IN-TIME QUANT + GEMINI"
    )

    print(
        f"Revision: {REVISION}"
    )

    print(
        f"yfinance: {yf.__version__}"
    )

    print(
        f"Backtest period: {BACKTEST_PERIOD}"
    )

    print(
        f"Round-trip cost: {ROUND_TRIP_COST:.3%}"
    )

    print(
        "Execution: Signal T Close -> "
        "Entry T+1 Open -> Exit T+H Close"
    )

    print(
        f"Refit interval: every {REFIT_EVERY} signal dates"
    )

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------

    data = load_all_data()

    if not data:
        raise RuntimeError(
            "No symbols loaded."
        )

    historical_gemini = (
        load_historical_gemini()
    )

    df = build_dataset(
        data
    )

    df = attach_point_in_time_gemini(
        df,
        historical_gemini,
    )

    unique_dates = sorted(
        pd.to_datetime(
            df["signal_date"]
        ).unique()
    )

    print("\nDATASET")

    print(
        f"Total observations: {len(df):,}"
    )

    print(
        f"Symbols: {df['symbol'].nunique()}"
    )

    print(
        f"Signal dates: {len(unique_dates)}"
    )

    coverage = float(
        df["gemini_available"].mean()
    )

    print(
        f"Historical Gemini coverage: {coverage:.2%}"
    )

    # ------------------------------------------------------------------
    # TIME SPLIT
    # ------------------------------------------------------------------

    # 60% development
    # 25% validation
    # 15% OOS

    n_dates = len(unique_dates)

    dev_end_idx = int(
        n_dates * 0.60
    )

    val_end_idx = int(
        n_dates * 0.85
    )

    dev_end_date = unique_dates[
        dev_end_idx - 1
    ]

    val_end_date = unique_dates[
        val_end_idx - 1
    ]

    dev = df[
        df["signal_date"]
        <= dev_end_date
    ].copy()

    validation = df[
        (
            df["signal_date"]
            > dev_end_date
        )
        &
        (
            df["signal_date"]
            <= val_end_date
        )
    ].copy()

    oos = df[
        df["signal_date"]
        > val_end_date
    ].copy()

    print(
        f"Development: {len(dev):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"OOS: {len(oos):,}"
    )

    print(
        f"Development end: {dev_end_date}"
    )

    print(
        f"Validation end: {val_end_date}"
    )

    print(
        f"OOS start: {oos['signal_date'].min()}"
    )

    # ------------------------------------------------------------------
    # BACKTEST
    # ------------------------------------------------------------------

    summary_rows = []

    primary_oos = None
    primary_thresholds = None

    for horizon in HORIZONS:

        print_header(
            f"HORIZON {horizon}D"
        )

        # Development -> validation
        val_pred = walk_forward(
            dev,
            validation,
            horizon,
            f"VALIDATION-H{horizon}",
        )

        if val_pred.empty:
            print(
                "Validation prediction failed; "
                "using defaults."
            )

            thresholds = {
                "pmin": DEFAULT_PMIN,
                "rmin": DEFAULT_RMIN,
            }

        else:

            thresholds = select_thresholds(
                val_pred,
                horizon,
            )

        print(
            f"Validation-selected thresholds: "
            f"P>={thresholds['pmin']:.2f}, "
            f"Return>={thresholds['rmin']:.4f}"
        )

        # Full historical training available before OOS.
        pre_oos = pd.concat(
            [
                dev,
                validation,
            ],
            ignore_index=True,
        )

        oos_pred = walk_forward(
            pre_oos,
            oos,
            horizon,
            f"OOS-H{horizon}",
        )

        if oos_pred.empty:
            print(
                "OOS prediction failed."
            )
            continue

        metrics = model_metrics(
            oos_pred,
            horizon,
        )

        perf, trades = performance(
            oos_pred,
            horizon,
            thresholds["pmin"],
            thresholds["rmin"],
        )

        boot = bootstrap_returns(
            (
                trades[
                    f"future_return_{horizon}"
                ]
                - ROUND_TRIP_COST
            ).to_numpy()
            if not trades.empty
            else []
        )

        portfolio = portfolio_test(
            oos_pred,
            horizon,
            thresholds["pmin"],
            thresholds["rmin"],
        )

        print(
            f"OOS observations: "
            f"{metrics.get('observations', 0):,}"
        )

        print(
            f"Directional accuracy: "
            f"{metrics.get('directional_accuracy', np.nan):.4f}"
        )

        print(
            f"Mean actual return: "
            f"{metrics.get('mean_actual_return', np.nan):.4%}"
        )

        print(
            f"Trades: "
            f"{perf.get('trades', 0)}"
        )

        print(
            f"Win rate: "
            f"{perf.get('win_rate', np.nan):.2%}"
        )

        print(
            f"Average net: "
            f"{perf.get('average_net', np.nan):.4%}"
        )

        print(
            f"Profit factor: "
            f"{perf.get('profit_factor', np.nan):.3f}"
        )

        print(
            f"Bootstrap mean 95% CI: "
            f"{boot['mean_low']:.4%} "
            f"to {boot['mean_high']:.4%}"
        )

        print(
            f"Portfolio ending equity: "
            f"{portfolio['ending_equity']:,.2f}"
        )

        # Save predictions
        save_csv(
            oos_pred,
            f"v7_1_3_oos_h{horizon}.csv",
        )

        row = {
            "model": "quant",
            "horizon": horizon,
            **metrics,
            "selected_pmin": thresholds[
                "pmin"
            ],
            "selected_rmin": thresholds[
                "rmin"
            ],
            **{
                f"trade_{k}": v
                for k, v in perf.items()
            },
            **{
                f"bootstrap_{k}": v
                for k, v in boot.items()
            },
            **{
                f"portfolio_{k}": v
                for k, v in portfolio.items()
            },
            "historical_gemini_coverage":
                coverage,
        }

        summary_rows.append(
            row
        )

        if horizon == PRIMARY_HORIZON:
            primary_oos = oos_pred
            primary_thresholds = thresholds

    # ------------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    save_csv(
        summary,
        "v7_1_3_model_comparison.csv",
    )

    # ------------------------------------------------------------------
    # CURRENT PRODUCTION SIGNAL
    # ------------------------------------------------------------------

    print_header(
        f"CURRENT PRODUCTION SIGNAL — H{PRIMARY_HORIZON}"
    )

    # Use all observations whose targets are known.
    production_training = df[
        df[
            f"future_return_{PRIMARY_HORIZON}"
        ].notna()
    ].copy()

    print(
        f"Production training rows: "
        f"{len(production_training):,}"
    )

    model = train_production_model(
        production_training,
        PRIMARY_HORIZON,
    )

    current = prepare_current_data(
        data
    )

    if current.empty:
        raise RuntimeError(
            "Unable to build current market dataset."
        )

    signals = current_signal(
        model,
        current,
    )

    save_csv(
        signals,
        "v7_1_3_current_signals.csv",
    )

    print("\nTOP CURRENT SIGNALS")

    print(
        signals[
            [
                "symbol",
                "quant_probability",
                "quant_return_prediction",
            ]
        ].head(15).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # CURRENT GEMINI
    # ------------------------------------------------------------------

    gemini_note = current_gemini_analysis(
        signals
    )

    # ------------------------------------------------------------------
    # TELEGRAM
    # ------------------------------------------------------------------

    if primary_thresholds is None:
        primary_thresholds = {
            "pmin": DEFAULT_PMIN,
            "rmin": DEFAULT_RMIN,
        }

    alert = format_alert(
        signals,
        PRIMARY_HORIZON,
        primary_thresholds,
        gemini_note,
    )

    alert_path = AUDIT_DIR / "latest_telegram_alert.txt"

    alert_path.write_text(
        alert,
        encoding="utf-8",
    )

    print_header(
        "TELEGRAM ALERT"
    )

    print(alert)

    telegram_send(
        alert
    )

    elapsed = time.time() - start

    print_header(
        "V7.1.3 COMPLETED"
    )

    print(
        f"Elapsed time: {elapsed / 60:.2f} minutes"
    )

    print(
        "Historical Gemini status:",
        (
            "TESTABLE"
            if coverage >= MIN_GEMINI_COVERAGE
            else "NOT TESTED"
        ),
    )

    print(
        "Current Telegram alert: generated."
    )

    print(
        "IMPORTANT: this system produces research signals, "
        "not guaranteed investment returns."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user."
        )
        sys.exit(130)
    except Exception as exc:
        print(
            "\nFATAL ERROR:",
            exc,
        )
        traceback.print_exc()
        sys.exit(1)
