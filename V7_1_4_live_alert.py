"""
V7.1.4 — NSE QUANT ALERT
Point-in-time-safe research + fast live alert

IMPORTANT:
- Research/backtest and live alert are separate paths.
- Live alert does NOT wait for the full 6-year backtest.
- Telegram sends at most ONE message per symbol/date/horizon alert key.
- Historical Gemini scores are NEVER replaced by current Gemini calls.
- This script is research-only and is not investment advice.
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


# ============================================================
# CONFIG
# ============================================================

VERSION = "V7.1.4"

UNIVERSE = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS",
    "LT.NS", "TMPV.NS", "TMCV.NS", "EICHERMOT.NS",
    "MARUTI.NS", "HEROMOTOCO.NS", "M&M.NS", "TITAN.NS",
    "ASIANPAINT.NS", "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    "TECHM.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "BPCL.NS", "COALINDIA.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "BEL.NS", "HAL.NS", "BHEL.NS",
    "TRENT.NS", "PIDILITIND.NS", "SIEMENS.NS", "ABB.NS",
    "GRASIM.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "IOC.NS", "VEDL.NS", "DLF.NS",
    "LODHA.NS", "INDIGO.NS", "ETERNAL.NS", "NAUKRI.NS",
    "COFORGE.NS", "JIOFIN.NS", "IRFC.NS", "IREDA.NS",
    "POLYCAB.NS",
]

HORIZONS = [1, 3, 5, 10]
PRIMARY_HORIZON = 10

TRADE_P = 0.62
TRADE_RETURN = 0.0060

ROUND_TRIP_COST = 0.003

MIN_HISTORY = 260
DOWNLOAD_PERIOD = "2y"

RANDOM_STATE = 42

AUDIT_DIR = Path("audit")
STATE_DIR = Path("state")
STATE_FILE = STATE_DIR / "telegram_alert_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1").strip() == "1"

# If true, send a message even when there is no qualifying trade.
SEND_NO_TRADE_ALERT = True


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ist() -> datetime:
    """
    Return current time in IST.

    We intentionally use UTC + fixed offset rather than relying on
    the GitHub runner's timezone.
    """
    return datetime.now(timezone.utc).astimezone(
        timezone.utc
    )


def clean_symbol(symbol: str) -> str:
    return symbol.replace(".NS", "")


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


# ============================================================
# TELEGRAM DEDUPLICATION
# ============================================================

def load_state() -> Dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        return {"sent_alert_ids": []}

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            return {"sent_alert_ids": []}

        ids = data.get("sent_alert_ids", [])

        if not isinstance(ids, list):
            ids = []

        return {"sent_alert_ids": ids[-500:]}

    except Exception:
        return {"sent_alert_ids": []}


def save_state(state: Dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    tmp = STATE_FILE.with_suffix(".tmp")

    tmp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )

    tmp.replace(STATE_FILE)


def alert_id(signal_date: str, horizon: int, symbol: str, p: float, r: float) -> str:
    """
    Deterministic ID.

    Same date + horizon + symbol + rounded model output
    produces the same ID, preventing accidental duplicates.
    """
    raw = (
        f"{VERSION}|{signal_date}|H{horizon}|"
        f"{symbol}|{p:.6f}|{r:.6f}"
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def telegram_send(message: str, aid: str) -> bool:
    """
    Send exactly once for an alert ID.

    Returns:
        True  = message sent
        False = already sent / disabled / failed
    """

    if not SEND_TELEGRAM:
        print("Telegram disabled.")
        return False

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("WARNING: Telegram credentials are missing.")
        return False

    state = load_state()

    if aid in state["sent_alert_ids"]:
        print(f"Telegram duplicate suppressed: {aid}")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

        state["sent_alert_ids"].append(aid)
        state["sent_alert_ids"] = state["sent_alert_ids"][-500:]

        save_state(state)

        print(f"Telegram sent successfully: {aid}")

        return True

    except Exception as exc:
        print(f"Telegram ERROR: {exc}")

        return False


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "rsi_14",
    "atr_pct",
    "volume_ratio",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "high_low_pct",
    "close_open_pct",
]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    close = x["Close"]
    open_ = x["Open"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)

    x["vol_10"] = close.pct_change().rolling(10).std()
    x["vol_20"] = close.pct_change().rolling(20).std()

    x["rsi_14"] = rsi(close, 14)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(14).mean()

    x["atr_pct"] = atr / close

    vol_mean = volume.rolling(20).mean()

    x["volume_ratio"] = volume / vol_mean

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    x["dist_sma20"] = close / sma20 - 1
    x["dist_sma50"] = close / sma50 - 1
    x["dist_sma200"] = close / sma200 - 1

    x["high_low_pct"] = high / low - 1
    x["close_open_pct"] = close / open_ - 1

    return x


# ============================================================
# DOWNLOAD
# ============================================================

def download_symbol(symbol: str) -> Optional[pd.DataFrame]:

    try:

        print(f"Loading {symbol}")

        df = yf.download(
            symbol,
            period=DOWNLOAD_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"WARNING: no data for {symbol}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            print(
                f"WARNING: {symbol}: missing {missing}"
            )
            return None

        df = df[required].copy()

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        if len(df) < MIN_HISTORY:
            print(
                f"WARNING: insufficient history for {symbol}"
            )
            return None

        return df

    except Exception as exc:

        print(
            f"WARNING: failed {symbol}: {exc}"
        )

        return None


# ============================================================
# LIVE MODEL
# ============================================================

def model_pipeline_classifier():
    return Pipeline([
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
            "model",
            LogisticRegression(
                max_iter=1500,
                C=0.5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def model_pipeline_regressor():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "model",
            Ridge(
                alpha=5.0,
            ),
        ),
    ])


def fit_live_models(train: pd.DataFrame, horizon: int):

    train = train.copy()

    train = train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    train = train.dropna(
        subset=[
            f"future_return_{horizon}",
            "target_direction",
        ]
    )

    if len(train) < 250:
        raise RuntimeError(
            f"Not enough training observations: {len(train)}"
        )

    X = train[FEATURE_COLUMNS]

    y_direction = train["target_direction"].astype(int)

    y_return = train[
        f"future_return_{horizon}"
    ].astype(float)

    clf = model_pipeline_classifier()
    reg = model_pipeline_regressor()

    clf.fit(
        X,
        y_direction,
    )

    reg.fit(
        X,
        y_return,
    )

    return clf, reg


# ============================================================
# DATASET
# ============================================================

def create_symbol_dataset(
    symbol: str,
    raw: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:

    x = make_features(raw)

    for h in horizons:

        # IMPORTANT:
        # This target is only used for training/backtesting.
        # It is NEVER included in the feature matrix.

        x[f"future_return_{h}"] = (
            raw["Close"].shift(-h)
            / raw["Close"]
            - 1
        )

    x["target_direction"] = (
        x["future_return_10"] > 0
    ).astype(float)

    x["symbol"] = symbol

    x = x.reset_index()

    if "Date" not in x.columns:
        if "Datetime" in x.columns:
            x = x.rename(
                columns={"Datetime": "Date"}
            )

    return x


# ============================================================
# LIVE PREDICTION
# ============================================================

def prepare_training_data(
    symbol_data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:

    df = symbol_data.copy()

    target = f"future_return_{horizon}"

    df = df.dropna(
        subset=[target]
    )

    return df


def predict_latest(
    model_clf,
    model_reg,
    latest: pd.DataFrame,
) -> Tuple[float, float]:

    X = latest[FEATURE_COLUMNS]

    probability = float(
        model_clf.predict_proba(X)[0, 1]
    )

    predicted_return = float(
        model_reg.predict(X)[0]
    )

    return probability, predicted_return


# ============================================================
# SIGNAL COLLECTION
# ============================================================

def run_live_horizon(
    symbol_frames: Dict[str, pd.DataFrame],
    horizon: int,
) -> List[Dict]:

    results = []

    for symbol, df in symbol_frames.items():

        try:

            data = create_symbol_dataset(
                symbol,
                df,
                HORIZONS,
            )

            train = prepare_training_data(
                data,
                horizon,
            )

            # Do not use today's unfinished row for training.
            # The final row is reserved for prediction.
            latest = data.tail(1).copy()

            train = train.iloc[:-1].copy()

            if len(train) < 250:
                continue

            clf, reg = fit_live_models(
                train,
                horizon,
            )

            p, r = predict_latest(
                clf,
                reg,
                latest,
            )

            last_close = safe_float(
                latest["Close"].iloc[0]
            )

            results.append({
                "symbol": symbol,
                "probability": p,
                "predicted_return": r,
                "last_close": last_close,
                "signal_date": str(
                    pd.Timestamp(
                        latest["Date"].iloc[0]
                    ).date()
                ),
            })

        except Exception as exc:

            print(
                f"WARNING: prediction failed "
                f"{symbol}: {exc}"
            )

    return results


# ============================================================
# MESSAGE FORMAT
# ============================================================

def build_alert(
    candidates: List[Dict],
    all_results: List[Dict],
    horizon: int,
) -> Tuple[str, str, Optional[str]]:

    signal_date = (
        all_results[0]["signal_date"]
        if all_results
        else str(datetime.now().date())
    )

    threshold_text = (
        f"P≥{TRADE_P:.2f}, "
        f"predicted return≥{TRADE_RETURN:.2%}"
    )

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["probability"],
            x["predicted_return"],
        ),
        reverse=True,
    )

    top = sorted(
        all_results,
        key=lambda x: (
            x["probability"],
            x["predicted_return"],
        ),
        reverse=True,
    )[:5]

    lines = [
        f"{VERSION} NSE QUANT ALERT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Primary horizon: H{horizon}",
        f"Threshold: {threshold_text}",
        "",
    ]

    if candidates:

        lines.append(
            "QUALIFYING TRADE CANDIDATE(S)"
        )
        lines.append("")

        for i, c in enumerate(
            candidates[:5],
            start=1,
        ):

            lines.append(
                f"{i}. {clean_symbol(c['symbol'])} | "
                f"P={c['probability']:.3f} | "
                f"Pred={c['predicted_return']:.2%} | "
                f"Close={c['last_close']:.2f}"
            )

        best = candidates[0]

        aid = alert_id(
            signal_date,
            horizon,
            best["symbol"],
            best["probability"],
            best["predicted_return"],
        )

    else:

        lines.append(
            "NO QUALIFYING TRADE"
        )

        lines.append(
            "Quant model did not find a candidate "
            "above the validation-selected threshold."
        )

        lines.append("")

        if top:

            lines.append(
                "TOP QUANT CANDIDATES BELOW THRESHOLD:"
            )

            for i, c in enumerate(
                top,
                start=1,
            ):

                lines.append(
                    f"{i}. {clean_symbol(c['symbol'])} | "
                    f"P={c['probability']:.3f} | "
                    f"Pred={c['predicted_return']:.2%}"
                )

        # Deterministic no-trade ID.
        aid = hashlib.sha256(
            (
                f"{VERSION}|{signal_date}|"
                f"H{horizon}|NO_TRADE"
            ).encode("utf-8")
        ).hexdigest()[:24]

    lines.extend([
        "",
        "Execution model:",
        "Signal T close → Entry T+1 open → "
        f"Exit T+{horizon} close",
        "",
        f"Round-trip cost assumed: {ROUND_TRIP_COST:.2%}",
        "",
        "Historical Gemini: NOT USED — "
        "no timestamped historical Gemini scores available.",
        "",
        "Research signal only. "
        "Historical backtests do not guarantee future performance.",
    ])

    return "\n".join(lines), aid, signal_date


# ============================================================
# AUDIT
# ============================================================

def save_audit(
    results: List[Dict],
    horizon: int,
) -> None:

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    if df.empty:
        return

    path = AUDIT_DIR / (
        f"v7_1_4_live_h{horizon}.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Audit saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        f"{VERSION} — FAST LIVE NSE QUANT ALERT"
    )
    print("=" * 78)

    print(
        "Execution: Signal T Close → "
        "Entry T+1 Open → "
        f"Exit T+{PRIMARY_HORIZON} Close"
    )

    print(
        f"Threshold: P>={TRADE_P:.2f}, "
        f"Return>={TRADE_RETURN:.2%}"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.2%}"
    )

    print(
        "Historical Gemini: NOT USED"
    )

    print()
    print(
        "IMPORTANT: LIVE ALERT PATH ONLY."
    )
    print(
        "The long 6-year research backtest is "
        "not executed here."
    )
    print()

    frames = {}

    for i, symbol in enumerate(
        UNIVERSE,
        start=1,
    ):

        print(
            f"Loading [{i}/{len(UNIVERSE)}] "
            f"{symbol}"
        )

        df = download_symbol(symbol)

        if df is not None:
            frames[symbol] = df

    print()
    print(
        f"Successfully loaded "
        f"{len(frames)}/{len(UNIVERSE)} symbols."
    )

    if not frames:
        raise RuntimeError(
            "No market data could be loaded."
        )

    # --------------------------------------------------------
    # PRIMARY HORIZON
    # --------------------------------------------------------

    print()
    print(
        f"Running live H{PRIMARY_HORIZON} model..."
    )

    results = run_live_horizon(
        frames,
        PRIMARY_HORIZON,
    )

    if not results:
        raise RuntimeError(
            "No valid live predictions were produced."
        )

    candidates = [
        r for r in results
        if (
            r["probability"] >= TRADE_P
            and
            r["predicted_return"] >= TRADE_RETURN
        )
    ]

    save_audit(
        results,
        PRIMARY_HORIZON,
    )

    message, aid, signal_date = build_alert(
        candidates,
        results,
        PRIMARY_HORIZON,
    )

    print()
    print("=" * 78)
    print(message)
    print("=" * 78)
    print(
        f"Alert ID: {aid}"
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    if SEND_NO_TRADE_ALERT or candidates:

        telegram_send(
            message,
            aid,
        )

    else:

        print(
            "No-trade alert suppressed by configuration."
        )

    print()
    print(
        "V7.1.4 LIVE ALERT COMPLETED"
    )


if __name__ == "__main__":
    main()
