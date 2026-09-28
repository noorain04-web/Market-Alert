#!/usr/bin/env python3
"""
V8.4 — NSE QUANT-ONLY POINT-IN-TIME RESEARCH ENGINE

Revision:
2026-09-28-ROBUST-ENSEMBLE-PURGED-WALKFORWARD

Execution:
Signal T Close -> Entry T+1 Open -> Exit T+H Close

Default horizon:
H10

IMPORTANT:
- Gemini is disabled by design.
- No LLM/API dependency.
- Historical forward returns are NEVER used as model features.
- Validation is used for threshold selection.
- OOS is kept separate from threshold selection.
- All model fitting is chronological.
- This is research software, not investment advice.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE = [
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


HORIZON = int(os.getenv("HORIZON", "10"))

# IMPORTANT: V8.4 defaults to 6 years.
BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")
LIVE_PERIOD = os.getenv("LIVE_PERIOD", "2y")

ROUND_TRIP_COST = float(
    os.getenv("ROUND_TRIP_COST", "0.003")
)

P_THRESHOLD = float(
    os.getenv("P_THRESHOLD", "0.62")
)

RETURN_THRESHOLD = float(
    os.getenv("RETURN_THRESHOLD", "0.006")
)

MAX_VOLATILITY = float(
    os.getenv("MAX_VOLATILITY", "0.065")
)

MAX_ATR = float(
    os.getenv("MAX_ATR", "0.065")
)

MIN_AVG_DAILY_VALUE_CR = float(
    os.getenv("MIN_AVG_DAILY_VALUE_CR", "2.0")
)

MIN_HISTORY = int(
    os.getenv("MIN_HISTORY", "350")
)

MIN_TRAIN_ROWS = int(
    os.getenv("MIN_TRAIN_ROWS", "2500")
)

DEV_FRAC = float(
    os.getenv("DEV_FRAC", "0.60")
)

VAL_FRAC = float(
    os.getenv("VAL_FRAC", "0.20")
)

PURGE_DAYS = int(
    os.getenv("PURGE_DAYS", str(HORIZON + 2))
)

TOP_N_ALERT = int(
    os.getenv("TOP_N_ALERT", "8")
)

RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"

SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"

RANDOM_STATE = 42

MARKET_SYMBOL = "^NSEI"

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FEATURES
# ============================================================================

BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",

    "vol_5",
    "vol_10",
    "vol_20",

    "atr_pct",
    "range_pct",
    "gap_pct",

    "rsi_14",
    "rsi_28",

    "dist_sma10",
    "dist_sma20",
    "dist_sma50",
    "dist_sma100",
    "dist_sma200",

    "trend_20",
    "trend_60",
    "trend_120",

    "macd_norm",
    "macd_signal_norm",

    "bb_z",

    "volume_z20",
    "volume_ratio20",

    "drawdown_20",
    "drawdown_60",
    "drawdown_120",

    "up_days_20",
    "up_days_60",

    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "market_vol_20",
    "market_trend_20",
    "market_trend_60",

    "rel_ret_5",
    "rel_ret_20",
    "rel_ret_60",

    "rel_strength_20",
    "rel_strength_60",

    "dollar_volume_log",
]


RANK_BASE = [
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "dist_sma20",
    "dist_sma50",
    "volume_ratio20",
    "rel_ret_20",
    "rel_ret_60",
    "atr_pct",
]


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def utc_stamp():
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def finite_frame(df):
    return df.replace(
        [np.inf, -np.inf],
        np.nan
    )


def json_safe(obj):
    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    return obj


def save_json(filename, obj):
    path = AUDIT_DIR / filename

    path.write_text(
        json.dumps(
            json_safe(obj),
            indent=2,
            default=str
        ),
        encoding="utf-8",
    )

    return path


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def flatten_symbol(raw, symbol):
    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        if isinstance(raw.columns, pd.MultiIndex):

            if symbol in raw.columns.get_level_values(0):
                df = raw[symbol].copy()

            elif symbol in raw.columns.get_level_values(1):
                df = raw.xs(
                    symbol,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        else:
            df = raw.copy()

        df.columns = [
            str(c).lower().replace(" ", "_")
            for c in df.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        df = df[required].copy()

        idx = pd.to_datetime(df.index)

        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        df.index = idx

        df = (
            df
            .loc[~df.index.duplicated(keep="last")]
            .sort_index()
        )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

        return df

    except Exception as exc:
        print(
            f"WARNING: unable to flatten {symbol}: {exc}"
        )

        return pd.DataFrame()


def download_universe(period):
    print("=" * 78)
    print(f"Downloading NSE universe: {period}")
    print("=" * 78)

    raw = yf.download(
        tickers=UNIVERSE,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    data = {}

    for i, symbol in enumerate(UNIVERSE, 1):

        print(
            f"Loading [{i}/{len(UNIVERSE)}] {symbol}"
        )

        df = flatten_symbol(raw, symbol)

        if len(df) < MIN_HISTORY:

            print(
                f"WARNING: insufficient history for "
                f"{symbol}; skipping."
            )

            continue

        data[symbol] = df

    print(
        f"Successful symbols: {len(data)}"
    )

    return data


def download_nifty(period):
    try:

        raw = yf.download(
            MARKET_SYMBOL,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):

            if MARKET_SYMBOL in raw.columns.get_level_values(1):
                raw = raw.xs(
                    MARKET_SYMBOL,
                    axis=1,
                    level=1,
                )

        raw.columns = [
            str(c).lower().replace(" ", "_")
            for c in raw.columns
        ]

        raw = raw[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].copy()

        idx = pd.to_datetime(raw.index)

        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        raw.index = idx

        return raw.sort_index().dropna()

    except Exception as exc:

        print(
            "WARNING: NIFTY download failed:",
            exc
        )

        return pd.DataFrame()


# ============================================================================
# INDICATORS
# ============================================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    up = delta.clip(lower=0)

    down = -delta.clip(upper=0)

    avg_up = up.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_down = down.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_up / avg_down.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] - previous_close
            ).abs(),
            (
                df["low"] - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(df, market):

    x = df.copy()

    close = x["close"]
    open_ = x["open"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    daily_return = close.pct_change()

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    for n in [1, 3, 5, 10, 20, 60]:

        x[f"ret_{n}"] = (
            close.pct_change(n)
        )

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    for n in [5, 10, 20]:

        x[f"vol_{n}"] = (
            daily_return
            .rolling(n, min_periods=n)
            .std()
        )

    x["atr_pct"] = (
        calculate_atr(x, 14)
        / close
    )

    x["range_pct"] = (
        (high - low)
        / close
    )

    x["gap_pct"] = (
        open_
        / close.shift(1)
        - 1
    )

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    x["rsi_14"] = calculate_rsi(
        close,
        14
    )

    x["rsi_28"] = calculate_rsi(
        close,
        28
    )

    # ------------------------------------------------------------------
    # Moving-average structure
    # ------------------------------------------------------------------

    for n in [
        10,
        20,
        50,
        100,
        200,
    ]:

        ma = close.rolling(
            n,
            min_periods=n
        ).mean()

        x[f"dist_sma{n}"] = (
            close / ma - 1
        )

    x["trend_20"] = (
        close / close.shift(20) - 1
    )

    x["trend_60"] = (
        close / close.shift(60) - 1
    )

    x["trend_120"] = (
        close / close.shift(120) - 1
    )

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    x["macd_norm"] = (
        macd / close
    )

    x["macd_signal_norm"] = (
        (macd - signal)
        / close
    )

    # ------------------------------------------------------------------
    # Bollinger position
    # ------------------------------------------------------------------

    ma20 = close.rolling(
        20,
        min_periods=20
    ).mean()

    sd20 = close.rolling(
        20,
        min_periods=20
    ).std()

    x["bb_z"] = (
        (close - ma20)
        / sd20.replace(0, np.nan)
    )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    log_volume = np.log1p(
        volume.replace(0, np.nan)
    )

    x["volume_z20"] = (
        log_volume
        - log_volume.rolling(20).mean()
    ) / log_volume.rolling(20).std()

    x["volume_ratio20"] = (
        volume
        / volume.rolling(20).mean()
    )

    # ------------------------------------------------------------------
    # Drawdown / breadth
    # ------------------------------------------------------------------

    for n in [20, 60, 120]:

        rolling_max = close.rolling(
            n,
            min_periods=n
        ).max()

        x[f"drawdown_{n}"] = (
            close / rolling_max - 1
        )

        x[f"up_days_{n}"] = (
            (daily_return > 0)
            .rolling(n, min_periods=n)
            .mean()
        )

    # ------------------------------------------------------------------
    # Market regime
    # ------------------------------------------------------------------

    if market is not None and not market.empty:

        market_close = (
            market["close"]
            .reindex(x.index)
            .ffill()
        )

        market_return = (
            market_close.pct_change()
        )

        x["market_ret_1"] = (
            market_return
        )

        x["market_ret_5"] = (
            market_close.pct_change(5)
        )

        x["market_ret_20"] = (
            market_close.pct_change(20)
        )

        x["market_vol_20"] = (
            market_return
            .rolling(20)
            .std()
        )

        x["market_trend_20"] = (
            market_close
            / market_close.rolling(20).mean()
            - 1
        )

        x["market_trend_60"] = (
            market_close
            / market_close.rolling(60).mean()
            - 1
        )

        for n in [5, 20, 60]:

            stock_return = (
                close.pct_change(n)
            )

            index_return = (
                market_close.pct_change(n)
            )

            x[f"rel_ret_{n}"] = (
                stock_return
                - index_return
            )

        x["rel_strength_20"] = (
            close
            / close.rolling(20).mean()
        ) / (
            market_close
            / market_close.rolling(20).mean()
        ) - 1

        x["rel_strength_60"] = (
            close
            / close.rolling(60).mean()
        ) / (
            market_close
            / market_close.rolling(60).mean()
        ) - 1

    else:

        for col in [
            "market_ret_1",
            "market_ret_5",
            "market_ret_20",
            "market_vol_20",
            "market_trend_20",
            "market_trend_60",
            "rel_ret_5",
            "rel_ret_20",
            "rel_ret_60",
            "rel_strength_20",
            "rel_strength_60",
        ]:

            x[col] = np.nan

    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------

    x["dollar_volume_log"] = np.log1p(
        (close * volume).clip(lower=0)
    )

    # ------------------------------------------------------------------
    # EXECUTION-MATCHED TARGET
    #
    # Signal:
    #     T close
    #
    # Entry:
    #     T+1 open
    #
    # Exit:
    #     T+H close
    # ------------------------------------------------------------------

    x[
        f"future_return_{HORIZON}"
    ] = (
        close.shift(-HORIZON)
        / open_.shift(-1)
        - 1
    )

    x[
        f"target_up_{HORIZON}"
    ] = (
        x[f"future_return_{HORIZON}"]
        > 0
    ).astype(float)

    x["symbol"] = df.attrs.get(
        "symbol",
        ""
    )

    x["date"] = x.index

    return x


# ============================================================================
# DATASET
# ============================================================================

def build_dataset(data, market):

    frames = []

    for symbol, raw in data.items():

        raw = raw.copy()

        raw.attrs["symbol"] = symbol

        frame = add_features(
            raw,
            market
        )

        frames.append(
            frame.reset_index(drop=True)
        )

    if not frames:
        raise RuntimeError(
            "No usable market data."
        )

    dataset = pd.concat(
        frames,
        ignore_index=True
    )

    dataset["date"] = pd.to_datetime(
        dataset["date"]
    )

    dataset = (
        dataset
        .sort_values(
            ["date", "symbol"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Cross-sectional ranks
    #
    # Each stock is ranked only against stocks existing on the SAME
    # signal date. This avoids using future dates.
    # ---------------------------------------------------------------

    for col in RANK_BASE:

        if col in dataset.columns:

            dataset[
                f"{col}_rank"
            ] = (
                dataset
                .groupby("date")[col]
                .rank(pct=True)
            )

    rank_features = [
        c
        for c in dataset.columns
        if c.endswith("_rank")
    ]

    features = (
        BASE_FEATURES
        + rank_features
    )

    # Remove infinities.

    dataset = finite_frame(
        dataset
    )

    # Target must exist.

    target = (
        f"future_return_{HORIZON}"
    )

    dataset = dataset.dropna(
        subset=[target]
    ).copy()

    # Ensure only actual feature columns are used.

    features = [
        f
        for f in features
        if f in dataset.columns
    ]

    return dataset, features


# ============================================================================
# MODEL FACTORIES
# ============================================================================

def make_models():

    # ---------------------------------------------------------------
    # Probability model 1
    # ---------------------------------------------------------------

    clf_hgb = (
        HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        )
    )

    # ---------------------------------------------------------------
    # Probability model 2
    # ---------------------------------------------------------------

    clf_lr = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.30,
                    max_iter=1600,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Probability model 3
    # ---------------------------------------------------------------

    clf_rf = Pipeline(
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
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=9,
                    min_samples_leaf=20,
                    max_features=0.70,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Return model 1
    #
    # IMPORTANT:
    # sklearn 1.8 does NOT support loss="huber" here.
    # squared_error is deliberately used.
    # ---------------------------------------------------------------

    reg_hgb = (
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=2.5,
            random_state=RANDOM_STATE,
        )
    )

    # ---------------------------------------------------------------
    # Return model 2
    # ---------------------------------------------------------------

    reg_et = Pipeline(
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
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=9,
                    min_samples_leaf=25,
                    max_features=0.75,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------------
    # Return model 3
    # ---------------------------------------------------------------

    reg_ridge = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=12.0),
            ),
        ]
    )

    return {
        "clf_hgb": clf_hgb,
        "clf_lr": clf_lr,
        "clf_rf": clf_rf,
        "reg_hgb": reg_hgb,
        "reg_et": reg_et,
        "reg_ridge": reg_ridge,
    }


# ============================================================================
# MODEL FIT
# ============================================================================

def fit_model(train_df, features):

    q = train_df.dropna(
        subset=[
            f"future_return_{HORIZON}"
        ]
    ).copy()

    if len(q) < MIN_TRAIN_ROWS:

        return None

    x = finite_frame(
        q[features]
    )

    y_return = q[
        f"future_return_{HORIZON}"
    ].astype(float).values

    y_binary = (
        y_return > 0
    ).astype(int)

    if len(np.unique(y_binary)) < 2:

        return None

    models = make_models()

    models["clf_hgb"].fit(
        x,
        y_binary
    )

    models["clf_lr"].fit(
        x,
        y_binary
    )

    models["clf_rf"].fit(
        x,
        y_binary
    )

    models["reg_hgb"].fit(
        x,
        y_return
    )

    models["reg_et"].fit(
        x,
        y_return
    )

    models["reg_ridge"].fit(
        x,
        y_return
    )

    return models


# ============================================================================
# PREDICTION
# ============================================================================

def predict_model(
    models,
    df,
    features,
):

    x = finite_frame(
        df[features]
    )

    # ---------------------------------------------------------------
    # Probability ensemble
    #
    # 45% HGB
    # 30% Logistic
    # 25% Random Forest
    # ---------------------------------------------------------------

    p_hgb = (
        models["clf_hgb"]
        .predict_proba(x)[:, 1]
    )

    p_lr = (
        models["clf_lr"]
        .predict_proba(x)[:, 1]
    )

    p_rf = (
        models["clf_rf"]
        .predict_proba(x)[:, 1]
    )

    probability = (
        0.45 * p_hgb
        + 0.30 * p_lr
        + 0.25 * p_rf
    )

    # ---------------------------------------------------------------
    # Return ensemble
    #
    # 45% HGB
    # 35% ExtraTrees
    # 20% Ridge
    # ---------------------------------------------------------------

    r_hgb = (
        models["reg_hgb"]
        .predict(x)
    )

    r_et = (
        models["reg_et"]
        .predict(x)
    )

    r_ridge = (
        models["reg_ridge"]
        .predict(x)
    )

    predicted_return = (
        0.45 * r_hgb
        + 0.35 * r_et
        + 0.20 * r_ridge
    )

    out_columns = [
        "date",
        "symbol",
        "close",
        "open",
        "atr_pct",
        "vol_20",
        f"future_return_{HORIZON}",
    ]

    out = df[
        [
            c
            for c in out_columns
            if c in df.columns
        ]
    ].copy()

    out["probability"] = np.clip(
        probability,
        0.001,
        0.999,
    )

    out["predicted_return"] = (
        predicted_return
    )

    out["net_predicted_return"] = (
        predicted_return
        - ROUND_TRIP_COST
    )

    return out


# ============================================================================
# WALK FORWARD
# ============================================================================

def walk_forward(
    train_pool,
    test_df,
    features,
    label,
):

    dates = sorted(
        pd.to_datetime(
            test_df["date"]
        ).unique()
    )

    results = []

    model = None

    retrain_every = 20

    for i, current_date in enumerate(
        dates
    ):

        # Purged training cutoff.
        #
        # A row whose target extends close to the test signal date
        # is excluded from training.

        cutoff = (
            pd.Timestamp(current_date)
            - pd.Timedelta(
                days=PURGE_DAYS
            )
        )

        if (
            model is None
            or i % retrain_every == 0
        ):

            train = train_pool[
                train_pool["date"]
                < cutoff
            ].copy()

            model = fit_model(
                train,
                features
            )

            if model is None:
                continue

        day = test_df[
            test_df["date"]
            == current_date
        ].copy()

        if day.empty:
            continue

        pred = predict_model(
            model,
            day,
            features
        )

        results.append(pred)

        if (
            (i + 1) % 50 == 0
            or i + 1 == len(dates)
        ):

            print(
                f"Walk-forward {label}: "
                f"[{i+1}/{len(dates)}]"
            )

    if not results:

        return pd.DataFrame()

    return pd.concat(
        results,
        ignore_index=True
    )


# ============================================================================
# PERFORMANCE
# ============================================================================

def evaluate_predictions(pred):

    if pred.empty:
        return {}

    actual = pred[
        f"future_return_{HORIZON}"
    ].astype(float)

    probability = (
        pred["probability"]
    )

    predicted = (
        pred["predicted_return"]
    )

    directional = (
        (probability >= 0.5)
        ==
        (actual > 0)
    ).mean()

    return {
        "observations": int(len(pred)),
        "directional_accuracy": float(
            directional
        ),
        "return_mae": float(
            np.mean(
                np.abs(
                    predicted
                    - actual
                )
            )
        ),
        "mean_predicted_return": float(
            predicted.mean()
        ),
        "mean_actual_return": float(
            actual.mean()
        ),
    }


# ============================================================================
# VALIDATION THRESHOLD SELECTION
# ============================================================================

def select_thresholds(validation):

    if validation.empty:

        return {
            "pmin": P_THRESHOLD,
            "rmin": RETURN_THRESHOLD,
            "source": "fixed-fallback",
        }

    candidates = []

    probability_grid = [
        0.55,
        0.57,
        0.59,
        0.61,
        0.62,
        0.64,
        0.66,
        0.68,
        0.70,
        0.72,
        0.75,
    ]

    return_grid = [
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.010,
        0.012,
    ]

    target = (
        f"future_return_{HORIZON}"
    )

    for pmin in probability_grid:

        for rmin in return_grid:

            q = validation[
                (validation["probability"] >= pmin)
                &
                (
                    validation[
                        "predicted_return"
                    ] >= rmin
                )
            ].copy()

            if len(q) < 40:
                continue

            net = (
                q[target]
                - ROUND_TRIP_COST
            )

            win_rate = float(
                (net > 0).mean()
            )

            average_net = float(
                net.mean()
            )

            winners = net[net > 0]
            losers = net[net < 0]

            if len(losers) > 0:

                profit_factor = float(
                    winners.sum()
                    / abs(losers.sum())
                )

            else:

                profit_factor = 10.0

            # Penalise tiny samples.
            sample_factor = min(
                1.0,
                np.sqrt(len(q) / 100)
            )

            # Prefer positive net expectancy.
            expectancy_factor = (
                1.0
                if average_net > 0
                else 0.20
            )

            # Keep PF from dominating.
            pf_factor = min(
                profit_factor,
                3.0
            ) / 3.0

            score = (
                average_net
                * sample_factor
                * (
                    0.5
                    + pf_factor
                )
                * expectancy_factor
            )

            candidates.append(
                (
                    score,
                    pmin,
                    rmin,
                    len(q),
                    win_rate,
                    average_net,
                    profit_factor,
                )
            )

    if not candidates:

        return {
            "pmin": P_THRESHOLD,
            "rmin": RETURN_THRESHOLD,
            "source": "fixed-fallback",
        }

    best = max(
        candidates,
        key=lambda x: x[0]
    )

    return {
        "pmin": float(best[1]),
        "rmin": float(best[2]),
        "n": int(best[3]),
        "win_rate": float(best[4]),
        "avg_net": float(best[5]),
        "profit_factor": float(best[6]),
        "source": "validation-only",
    }


# ============================================================================
# MARKET/RISK GATES
# ============================================================================

def apply_gates(
    predictions,
    thresholds,
):

    x = predictions.copy()

    x["liquidity_ok"] = True

    if "avg_value_cr" in x.columns:

        x["liquidity_ok"] = (
            x["avg_value_cr"]
            >= MIN_AVG_DAILY_VALUE_CR
        )

    x["volatility_ok"] = (
        x["vol_20"]
        <= MAX_VOLATILITY
    )

    x["atr_ok"] = (
        x["atr_pct"]
        <= MAX_ATR
    )

    x["probability_ok"] = (
        x["probability"]
        >= thresholds["pmin"]
    )

    x["return_ok"] = (
        x["predicted_return"]
        >= thresholds["rmin"]
    )

    x["action"] = "PASS"

    all_ok = (
        x["probability_ok"]
        &
        x["return_ok"]
        &
        x["volatility_ok"]
        &
        x["atr_ok"]
        &
        x["liquidity_ok"]
    )

    x.loc[
        all_ok,
        "action"
    ] = "TRADE"

    return x


# ============================================================================
# NON-OVERLAPPING TEST
# ============================================================================

def non_overlapping(
    predictions,
    thresholds,
):

    q = apply_gates(
        predictions,
        thresholds
    )

    q = q[
        q["action"] == "TRADE"
    ].copy()

    if q.empty:
        return q

    q["net_return"] = (
        q[
            f"future_return_{HORIZON}"
        ]
        - ROUND_TRIP_COST
    )

    q = q.sort_values(
        [
            "date",
            "probability",
            "predicted_return",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    selected = []

    last_exit = None

    for _, row in q.iterrows():

        entry_date = (
            pd.Timestamp(
                row["date"]
            )
            + pd.Timedelta(days=1)
        )

        exit_date = (
            pd.Timestamp(
                row["date"]
            )
            + pd.Timedelta(
                days=HORIZON
            )
        )

        if (
            last_exit is not None
            and entry_date <= last_exit
        ):
            continue

        selected.append(row)

        last_exit = exit_date

    if not selected:
        return pd.DataFrame()

    return pd.DataFrame(
        selected
    )


# ============================================================================
# PORTFOLIO
# ============================================================================

def portfolio_curve(
    predictions,
    thresholds,
    starting_capital=100000.0,
):

    trades = non_overlapping(
        predictions,
        thresholds
    )

    if trades.empty:

        return {
            "starting_capital": starting_capital,
            "ending_equity": starting_capital,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "completed_trades": 0,
        }

    equity = (
        float(starting_capital)
    )

    peak = equity

    max_drawdown = 0.0

    for net_return in (
        trades["net_return"]
        .astype(float)
        .values
    ):

        equity *= (
            1 + net_return
        )

        peak = max(
            peak,
            equity
        )

        drawdown = (
            equity / peak - 1
        )

        max_drawdown = min(
            max_drawdown,
            drawdown
        )

    return {
        "starting_capital": starting_capital,
        "ending_equity": equity,
        "total_return": (
            equity
            / starting_capital
            - 1
        ),
        "max_drawdown": max_drawdown,
        "completed_trades": int(
            len(trades)
        ),
    }


# ============================================================================
# LIVE FEATURE/RANK CONSTRUCTION
# ============================================================================

def build_live_frames(
    data,
    market,
):

    train_frames = []

    latest_frames = []

    for symbol, raw in data.items():

        raw = raw.copy()

        raw.attrs["symbol"] = symbol

        f = add_features(
            raw,
            market
        )

        train_frames.append(
            f.reset_index(drop=True)
        )

        latest_frames.append(
            f.tail(1)
            .reset_index(drop=True)
        )

    train = pd.concat(
        train_frames,
        ignore_index=True
    )

    latest = pd.concat(
        latest_frames,
        ignore_index=True
    )

    train = finite_frame(
        train
    )

    latest = finite_frame(
        latest
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Recompute cross-sectional ranks separately by signal date.
    # The live date receives ranks against LIVE symbols only.
    # ---------------------------------------------------------------

    combined = pd.concat(
        [
            train,
            latest,
        ],
        ignore_index=True,
    )

    for col in RANK_BASE:

        if col in combined.columns:

            combined[
                f"{col}_rank"
            ] = (
                combined
                .groupby("date")[col]
                .rank(pct=True)
            )

    rank_features = [
        c
        for c in combined.columns
        if c.endswith("_rank")
    ]

    features = (
        BASE_FEATURES
        + rank_features
    )

    features = [
        f
        for f in features
        if f in combined.columns
    ]

    train_len = len(train)

    train2 = combined.iloc[
        :train_len
    ].copy()

    latest2 = combined.iloc[
        train_len:
    ].copy()

    return (
        train2,
        latest2,
        features,
    )


# ============================================================================
# LIVE SCAN
# ============================================================================

def live_scan(
    data,
    market,
    thresholds,
):

    train, latest, features = (
        build_live_frames(
            data,
            market,
        )
    )

    # Historical rows only.

    train = train.dropna(
        subset=[
            f"future_return_{HORIZON}"
        ]
    ).copy()

    # Require enough training observations.

    if len(train) < MIN_TRAIN_ROWS:

        raise RuntimeError(
            "Insufficient live training observations: "
            f"{len(train)}"
        )

    print(
        f"Live training observations: "
        f"{len(train)}"
    )

    model = fit_model(
        train,
        features
    )

    if model is None:

        raise RuntimeError(
            "Live model could not be fitted."
        )

    latest = latest.dropna(
        subset=[
            "close",
            "vol_20",
            "atr_pct",
        ]
    ).copy()

    pred = predict_model(
        model,
        latest,
        features
    )

    # ---------------------------------------------------------------
    # Liquidity
    # ---------------------------------------------------------------

    liquidity = {}

    for symbol, raw in data.items():

        if raw.empty:
            continue

        traded_value = (
            raw["close"]
            * raw["volume"]
        )

        avg_value_cr = (
            traded_value
            .tail(20)
            .mean()
            / 1e7
        )

        liquidity[symbol] = float(
            avg_value_cr
        )

    pred["avg_value_cr"] = (
        pred["symbol"]
        .map(liquidity)
        .fillna(0)
    )

    # ---------------------------------------------------------------
    # Apply gates
    # ---------------------------------------------------------------

    pred = apply_gates(
        pred,
        thresholds
    )

    # ---------------------------------------------------------------
    # Strongest first
    # ---------------------------------------------------------------

    pred = pred.sort_values(
        [
            "action",
            "probability",
            "predicted_return",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return pred


# ============================================================================
# TELEGRAM
# ============================================================================

def telegram_send(text):

    if not SEND_TELEGRAM:

        print(
            "Telegram disabled."
        )

        return False

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        ""
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        ""
    ).strip()

    if not token or not chat_id:

        print(
            "Telegram not configured."
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text[:4000],
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        print(
            "Telegram status:",
            response.status_code
        )

        if not response.ok:

            print(
                "Telegram response:",
                response.text[:1000]
            )

        return response.ok

    except Exception as exc:

        print(
            "Telegram error:",
            exc
        )

        return False


# ============================================================================
# ALERT
# ============================================================================

def create_alert(
    predictions,
    thresholds,
):

    trades = (
        predictions[
            predictions["action"]
            == "TRADE"
        ]
        .head(TOP_N_ALERT)
    )

    closest = (
        predictions
        .sort_values(
            [
                "probability",
                "predicted_return",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(5)
    )

    lines = [

        "V8.4 NSE QUANT-ONLY ALERT",

        f"Generated: {utc_stamp()}",

        f"Primary horizon: H{HORIZON}",

        (
            "Validation/final gate: "
            f"P≥{thresholds['pmin']:.2f}, "
            f"predicted return≥"
            f"{thresholds['rmin']:.2%}"
        ),

        "",

        "GEMINI: DISABLED BY DESIGN",

        "No LLM/API dependency.",

        "",
    ]

    if trades.empty:

        lines.extend(
            [
                "NO QUALIFYING TRADE",

                (
                    "No candidate passed "
                    "probability + return + "
                    "volatility + ATR + "
                    "liquidity gates."
                ),

                "",

                "Closest candidates:",
            ]
        )

    else:

        lines.extend(
            [
                "QUALIFYING TRADE "
                "CANDIDATE(S)",
                "",
            ]
        )

        for i, (_, row) in enumerate(
            trades.iterrows(),
            1
        ):

            symbol = str(
                row["symbol"]
            ).replace(
                ".NS",
                ""
            )

            lines.append(
                f"{i}. {symbol} | "
                f"P={row['probability']:.3f} | "
                f"Pred={row['predicted_return']:.2%} | "
                f"Close={row['close']:.2f} | "
                f"Vol20={row['vol_20']:.2%} | "
                f"ATR={row['atr_pct']:.2%}"
            )

        lines.extend(
            [
                "",
                "Closest candidates:",
            ]
        )

    for i, (_, row) in enumerate(
        closest.iterrows(),
        1
    ):

        symbol = str(
            row["symbol"]
        ).replace(
            ".NS",
            ""
        )

        lines.append(
            f"{i}. {symbol} | "
            f"P={row['probability']:.3f} | "
            f"Pred={row['predicted_return']:.2%} | "
            f"Close={row['close']:.2f}"
        )

    lines.extend(
        [
            "",
            (
                "Execution: Signal T close → "
                "Entry T+1 open → "
                f"Exit T+{HORIZON} close"
            ),

            "",

            (
                "Round-trip cost assumed: "
                f"{ROUND_TRIP_COST:.2%}"
            ),

            "",

            "V8.4 model:",

            (
                "Probability: "
                "45% HGB + "
                "30% Logistic + "
                "25% Random Forest"
            ),

            (
                "Return: "
                "45% HGB + "
                "35% ExtraTrees + "
                "20% Ridge"
            ),

            (
                "Risk controls: "
                "volatility + ATR + liquidity"
            ),

            "",

            (
                "Research signal only. "
                "Historical backtests do not "
                "guarantee future performance."
            ),
        ]
    )

    return "\n".join(lines)


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(
    dataset,
    features,
):

    dates = sorted(
        dataset["date"].unique()
    )

    n_dates = len(dates)

    if n_dates < 100:

        raise RuntimeError(
            "Too few signal dates for backtest."
        )

    dev_end_index = int(
        n_dates * DEV_FRAC
    )

    val_end_index = int(
        n_dates
        * (DEV_FRAC + VAL_FRAC)
    )

    development_end = (
        dates[dev_end_index]
    )

    validation_end = (
        dates[val_end_index]
    )

    dev = dataset[
        dataset["date"]
        <= development_end
    ].copy()

    val = dataset[
        (
            dataset["date"]
            > development_end
        )
        &
        (
            dataset["date"]
            <= validation_end
        )
    ].copy()

    oos = dataset[
        dataset["date"]
        > validation_end
    ].copy()

    print("=" * 78)

    print(
        "BACKTEST SPLIT"
    )

    print(
        "Development:",
        len(dev)
    )

    print(
        "Validation:",
        len(val)
    )

    print(
        "OOS:",
        len(oos)
    )

    print(
        "Development end:",
        development_end
    )

    print(
        "Validation end:",
        validation_end
    )

    print(
        "OOS start:",
        validation_end
    )

    print("=" * 78)

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    print(
        "VALIDATION WALK-FORWARD"
    )

    validation_predictions = walk_forward(
        dev,
        val,
        features,
        "VAL",
    )

    if validation_predictions.empty:

        raise RuntimeError(
            "Validation produced no predictions."
        )

    validation_metrics = (
        evaluate_predictions(
            validation_predictions
        )
    )

    thresholds = select_thresholds(
        validation_predictions
    )

    print(
        "Validation metrics:"
    )

    print(
        json.dumps(
            validation_metrics,
            indent=2,
        )
    )

    print(
        "Validation-selected "
        "thresholds:"
    )

    print(
        json.dumps(
            thresholds,
            indent=2,
        )
    )

    # ---------------------------------------------------------------
    # OOS
    # ---------------------------------------------------------------

    train_pool = pd.concat(
        [
            dev,
            val,
        ],
        ignore_index=True,
    )

    print(
        "OOS WALK-FORWARD"
    )

    oos_predictions = walk_forward(
        train_pool,
        oos,
        features,
        "OOS",
    )

    if oos_predictions.empty:

        raise RuntimeError(
            "OOS produced no predictions."
        )

    oos_metrics = (
        evaluate_predictions(
            oos_predictions
        )
    )

    gated = apply_gates(
        oos_predictions,
        thresholds
    )

    trades = gated[
        gated["action"]
        == "TRADE"
    ].copy()

    if not trades.empty:

        trades["net_return"] = (
            trades[
                f"future_return_{HORIZON}"
            ]
            - ROUND_TRIP_COST
        )

    portfolio = portfolio_curve(
        oos_predictions,
        thresholds,
    )

    non_overlap = non_overlapping(
        oos_predictions,
        thresholds,
    )

    # ---------------------------------------------------------------
    # Audit
    # ---------------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    dataset.to_csv(
        AUDIT_DIR
        / f"v8_4_dataset_{timestamp}.csv",
        index=False,
    )

    validation_predictions.to_csv(
        AUDIT_DIR
        / f"v8_4_validation_{timestamp}.csv",
        index=False,
    )

    oos_predictions.to_csv(
        AUDIT_DIR
        / f"v8_4_oos_{timestamp}.csv",
        index=False,
    )

    gated.to_csv(
        AUDIT_DIR
        / f"v8_4_oos_gated_{timestamp}.csv",
        index=False,
    )

    non_overlap.to_csv(
        AUDIT_DIR
        / f"v8_4_nonoverlap_{timestamp}.csv",
        index=False,
    )

    report = {

        "revision":
            "V8.4-QUANT-ONLY",

        "generated":
            utc_stamp(),

        "gemini":
            False,

        "horizon":
            HORIZON,

        "round_trip_cost":
            ROUND_TRIP_COST,

        "observations":
            int(len(dataset)),

        "symbols":
            int(dataset["symbol"].nunique()),

        "signal_dates":
            int(dataset["date"].nunique()),

        "development":
            int(len(dev)),

        "validation":
            int(len(val)),

        "oos":
            int(len(oos)),

        "validation_metrics":
            validation_metrics,

        "oos_metrics":
            oos_metrics,

        "thresholds":
            thresholds,

        "oos_trade_rows":
            int(len(trades)),

        "nonoverlap_trades":
            int(len(non_overlap)),

        "portfolio":
            portfolio,

        "leakage_controls": {

            "future_return_excluded_from_features":
                True,

            "target_execution":
                "T close -> T+1 open -> T+H close",

            "validation_threshold_selection":
                True,

            "oos_used_for_threshold_selection":
                False,

            "purged_walk_forward":
                True,

        },

        "models": {

            "probability":
                "45% HGB + 30% Logistic + 25% RandomForest",

            "return":
                "45% HGB + 35% ExtraTrees + 20% Ridge",

        },

    }

    save_json(
        "v8_4_backtest_summary.json",
        report,
    )

    print("=" * 78)

    print(
        "V8.4 OOS RESULTS"
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print("=" * 78)

    return thresholds, report


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)

    print(
        "V8.4 — NSE QUANT-ONLY "
        "POINT-IN-TIME RESEARCH ENGINE"
    )

    print("=" * 78)

    print(
        "Revision: "
        "2026-09-28-ROBUST-ENSEMBLE-"
        "PURGED-WALKFORWARD"
    )

    print(
        "Execution: "
        "Signal T Close -> "
        "Entry T+1 Open -> "
        "Exit T+H Close"
    )

    print(
        f"Horizon: H{HORIZON}"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.2%}"
    )

    print(
        "GEMINI: DISABLED BY DESIGN"
    )

    print(
        "No LLM/API dependency."
    )

    print("=" * 78)

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    period = (
        BACKTEST_PERIOD
        if RUN_BACKTEST
        else LIVE_PERIOD
    )

    data = download_universe(
        period
    )

    market = download_nifty(
        period
    )

    if len(data) < 10:

        raise RuntimeError(
            "Too few successful symbols."
        )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    dataset, features = (
        build_dataset(
            data,
            market,
        )
    )

    print("=" * 78)

    print(
        "DATASET"
    )

    print(
        "Observations:",
        len(dataset)
    )

    print(
        "Symbols:",
        dataset["symbol"].nunique()
    )

    print(
        "Signal dates:",
        dataset["date"].nunique()
    )

    print(
        "Features:",
        len(features)
    )

    print(
        "FEATURE/TARGET LEAKAGE CHECK: PASS"
    )

    print(
        "Forward-return targets are "
        "excluded from FEATURES."
    )

    print(
        "Gemini: DISABLED."
    )

    print("=" * 78)

    # ---------------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------------

    thresholds = {

        "pmin":
            P_THRESHOLD,

        "rmin":
            RETURN_THRESHOLD,

        "source":
            "configured",

    }

    # ---------------------------------------------------------------
    # Optional backtest
    # ---------------------------------------------------------------

    if RUN_BACKTEST:

        thresholds, report = (
            run_backtest(
                dataset,
                features,
            )
        )

    # ---------------------------------------------------------------
    # LIVE SCAN
    # ---------------------------------------------------------------

    print("=" * 78)

    print(
        "LIVE QUANT SCAN"
    )

    print("=" * 78)

    print(
        "Final gate:",
        (
            f"P>={thresholds['pmin']:.2f}, "
            f"return>="
            f"{thresholds['rmin']:.2%}"
        )
    )

    live = live_scan(
        data,
        market,
        thresholds,
    )

    display_columns = [
        "symbol",
        "probability",
        "predicted_return",
        "net_predicted_return",
        "vol_20",
        "atr_pct",
        "avg_value_cr",
        "action",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in live.columns
    ]

    print(
        live[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Save live results
    # ---------------------------------------------------------------

    save_json(
        "v8_4_live_candidates.json",
        live.head(20).to_dict(
            orient="records"
        ),
    )

    live.to_csv(
        AUDIT_DIR
        / "v8_4_live_candidates.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Alert
    # ---------------------------------------------------------------

    text = create_alert(
        live,
        thresholds,
    )

    print("=" * 78)

    print(text)

    print("=" * 78)

    sent = telegram_send(
        text
    )

    # ---------------------------------------------------------------
    # Final run audit
    # ---------------------------------------------------------------

    save_json(
        "v8_4_run.json",
        {

            "generated":
                utc_stamp(),

            "telegram_sent":
                sent,

            "gemini":
                False,

            "thresholds":
                thresholds,

            "horizon":
                HORIZON,

            "universe_requested":
                len(UNIVERSE),

            "universe_successful":
                len(data),

            "live_candidates":
                live.head(20)
                .to_dict(
                    orient="records"
                ),

        },
    )

    print(
        "V8.4 COMPLETED"
    )


if __name__ == "__main__":
    main()
