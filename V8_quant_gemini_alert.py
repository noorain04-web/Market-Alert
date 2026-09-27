"""
==============================================================================
V8 — POINT-IN-TIME QUANT + GEMINI ENSEMBLE NSE RESEARCH ALERT
==============================================================================

Architecture:
    Historical/backtest:
        Quant model only unless timestamped historical Gemini data exists.

    Live/current signal:
        Quant -> Gemini research -> calibrated ensemble -> risk gates -> alert

Execution:
    Signal T close
    Entry T+1 open
    Exit T+H close

Important:
    Gemini is NOT treated as an oracle.
    Gemini provides a structured research/risk signal.
    The quantitative model remains the primary statistical model.

Revision:
    2026-09-27-V8-GEMINI-ENSEMBLE

Python:
    3.12+

Dependencies:
    yfinance
    pandas
    numpy
    scikit-learn
    requests
    google-genai
    pydantic
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ============================================================================
# CONFIG
# ============================================================================

VERSION = "V8.0"

BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

ROUND_TRIP_COST = float(
    os.getenv("ROUND_TRIP_COST", "0.003")
)

PRIMARY_HORIZON = int(
    os.getenv("PRIMARY_HORIZON", "10")
)

MIN_PROBABILITY = float(
    os.getenv("MIN_PROBABILITY", "0.62")
)

MIN_PREDICTED_RETURN = float(
    os.getenv("MIN_PREDICTED_RETURN", "0.006")
)

MIN_GEMINI_COVERAGE = float(
    os.getenv("MIN_GEMINI_COVERAGE", "0.60")
)

# Ensemble weights.
#
# Quant remains dominant.
# Gemini is an information/risk overlay.
QUANT_WEIGHT = float(
    os.getenv("QUANT_WEIGHT", "0.70")
)

GEMINI_WEIGHT = float(
    os.getenv("GEMINI_WEIGHT", "0.30")
)

# Risk limits
MAX_VOLATILITY = float(
    os.getenv("MAX_VOLATILITY", "0.065")
)

MAX_STOCKS_IN_ALERT = int(
    os.getenv("MAX_STOCKS_IN_ALERT", "5")
)

RANDOM_STATE = 42

# Current Gemini model.
#
# Google's current model catalogue lists Gemini 3.8 Flash as stable.
# Keep configurable so model migrations do not require rewriting the script.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()


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
# GEMINI STRUCTURED RESPONSE
# ============================================================================

if GEMINI_AVAILABLE:

    class GeminiAssessment:
        """
        Pydantic-like structured schema.

        We deliberately request bounded numeric outputs.
        Gemini is NOT asked to output an unrestricted stock price.
        """

        pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.5
    return float(np.clip(x, 0.0, 1.0))


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def pct(x):
    if not np.isfinite(x):
        return "NA"
    return f"{x * 100:.2f}%"


def symbol_name(symbol: str) -> str:
    return symbol.replace(".NS", "")


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_symbol(symbol: str) -> Optional[pd.DataFrame]:

    try:
        print(f"Loading {symbol}")

        df = yf.download(
            symbol,
            period=BACKTEST_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"WARNING: no data for {symbol}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            print(
                f"WARNING: {symbol}: missing columns {missing}"
            )
            return None

        df = df[required].copy()

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        if len(df) < 300:
            print(
                f"WARNING: insufficient history for {symbol}; skipping."
            )
            return None

        df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as exc:
        print(
            f"WARNING: failed {symbol}: {exc}"
        )
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(
    df: pd.DataFrame,
    symbol: str
) -> pd.DataFrame:

    x = df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    # Returns
    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)
    x["ret_60"] = close.pct_change(60)

    # Moving-average structure
    for n in [5, 10, 20, 50, 100, 200]:
        ma = close.rolling(n).mean()
        x[f"ma_ratio_{n}"] = close / ma - 1.0

    # Volatility
    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()

    # ATR-like normalized range
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr_14"] = (
        tr.rolling(14).mean() / close
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    x["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # Volume
    vol_ma20 = volume.rolling(20).mean()

    x["volume_ratio"] = (
        volume / vol_ma20
    )

    x["volume_change_5"] = (
        volume.pct_change(5)
    )

    # Candle structure
    x["range_pct"] = (
        (high - low) / close
    )

    x["close_location"] = (
        (close - low) /
        (high - low).replace(0, np.nan)
    )

    # Breakout structure
    x["high_20_break"] = (
        close /
        high.rolling(20).max().shift(1) - 1
    )

    x["low_20_break"] = (
        close /
        low.rolling(20).min().shift(1) - 1
    )

    # Drawdown
    rolling_high = close.rolling(60).max()

    x["drawdown_60"] = (
        close / rolling_high - 1
    )

    # Trend acceleration
    x["momentum_acceleration"] = (
        x["ret_5"] - x["ret_20"]
    )

    x["symbol"] = symbol

    return x


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ma_ratio_5",
    "ma_ratio_10",
    "ma_ratio_20",
    "ma_ratio_50",
    "ma_ratio_100",
    "ma_ratio_200",
    "vol_5",
    "vol_10",
    "vol_20",
    "vol_60",
    "atr_14",
    "rsi_14",
    "volume_ratio",
    "volume_change_5",
    "range_pct",
    "close_location",
    "high_20_break",
    "low_20_break",
    "drawdown_60",
    "momentum_acceleration",
]


# ============================================================================
# TARGETS
# ============================================================================

def build_dataset(
    symbol: str,
    df: pd.DataFrame
) -> pd.DataFrame:

    x = add_features(df, symbol)

    # Execution:
    # signal T close
    # entry T+1 open
    # exit T+H close
    #
    # Therefore target is deliberately based on future prices.
    # It is NEVER included in FEATURES.

    for h in [3, 5, 10]:

        future_entry = x["Open"].shift(-1)

        future_exit = x["Close"].shift(-h)

        x[f"future_return_{h}"] = (
            future_exit / future_entry - 1.0
        )

        x[f"target_direction_{h}"] = (
            x[f"future_return_{h}"] > 0
        ).astype(float)

    x["symbol"] = symbol

    return x


# ============================================================================
# LEAKAGE CHECK
# ============================================================================

def leakage_check(df: pd.DataFrame):

    forbidden = []

    for c in FEATURE_COLUMNS:

        if "future" in c.lower():
            forbidden.append(c)

        if "target" in c.lower():
            forbidden.append(c)

    if forbidden:
        raise RuntimeError(
            "LEAKAGE CHECK FAILED: "
            + str(forbidden)
        )

    print(
        "FEATURE/TARGET LEAKAGE CHECK: PASS"
    )


# ============================================================================
# MODEL
# ============================================================================

def make_classifier():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    C=0.25,
                    random_state=RANDOM_STATE
                )
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
                    add_indicator=True
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )


@dataclass
class QuantModel:

    classifier: Any
    regressor: Any


def fit_model(
    train: pd.DataFrame,
    horizon: int
) -> QuantModel:

    y_direction = (
        train[f"target_direction_{horizon}"]
    )

    y_return = (
        train[f"future_return_{horizon}"]
    )

    valid = (
        y_direction.notna()
        & y_return.notna()
    )

    train = train.loc[valid].copy()

    if len(train) < 500:
        raise RuntimeError(
            f"Not enough training rows: {len(train)}"
        )

    X = train[FEATURE_COLUMNS].copy()

    # Hard finite cleanup before sklearn.
    # Pipeline imputation handles NaN, but infinite values
    # must never reach sklearn.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    clf = make_classifier()
    reg = make_regressor()

    clf.fit(
        X,
        y_direction.astype(int)
    )

    reg.fit(
        X,
        y_return.astype(float)
    )

    return QuantModel(
        classifier=clf,
        regressor=reg
    )


def predict_model(
    model: QuantModel,
    frame: pd.DataFrame
) -> pd.DataFrame:

    X = frame[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    probability = model.classifier.predict_proba(
        X
    )[:, 1]

    prediction = model.regressor.predict(
        X
    )

    out = frame.copy()

    out["quant_probability"] = np.clip(
        probability,
        0.001,
        0.999
    )

    out["quant_return"] = prediction

    return out


# ============================================================================
# GEMINI
# ============================================================================

def get_gemini_client():

    if not GEMINI_AVAILABLE:
        return None

    if not GEMINI_API_KEY:
        return None

    try:
        return genai.Client(
            api_key=GEMINI_API_KEY
        )
    except Exception as exc:
        print(
            "Gemini client initialization failed:",
            exc
        )
        return None


def gemini_prompt(row: pd.Series) -> str:

    symbol = symbol_name(
        str(row["symbol"])
    )

    return f"""
You are a market-research risk analyst.

You are NOT allowed to guarantee a return.

Evaluate the following current NSE equity using ONLY the information
provided in this prompt.

Ticker: {symbol}

Current close:
{safe_float(row.get("Close"))}

Quant probability of positive H10 return:
{safe_float(row.get("quant_probability")):.4f}

Quant predicted H10 return:
{safe_float(row.get("quant_return")):.6f}

Recent returns:
1D = {safe_float(row.get("ret_1")):.6f}
5D = {safe_float(row.get("ret_5")):.6f}
20D = {safe_float(row.get("ret_20")):.6f}

RSI:
{safe_float(row.get("rsi_14")):.2f}

20-day volatility:
{safe_float(row.get("vol_20")):.6f}

ATR:
{safe_float(row.get("atr_14")):.6f}

Volume ratio:
{safe_float(row.get("volume_ratio")):.3f}

60-day drawdown:
{safe_float(row.get("drawdown_60")):.6f}

Your job is to assess whether the quantitative signal deserves
confirmation, downgrade, or rejection.

Return:

1. directional_score:
   0 = strongly bearish
   0.5 = neutral
   1 = strongly bullish

2. return_adjustment:
   expected percentage adjustment to the quantitative prediction.
   Keep between -0.03 and +0.03.

3. risk_score:
   0 = low risk
   1 = extremely high risk

4. confidence:
   0 = no confidence
   1 = very high confidence

5. decision:
   CONFIRM
   NEUTRAL
   DOWNGRADE
   REJECT

6. concise_reason:
   maximum 300 characters.

Do NOT invent news or facts.
Do NOT claim access to information that is not supplied.
Do NOT provide a guaranteed future price.
"""


def call_gemini(
    client,
    row: pd.Series
) -> Optional[Dict[str, Any]]:

    if client is None:
        return None

    prompt = gemini_prompt(row)

    schema = {
        "type": "object",
        "properties": {
            "directional_score": {
                "type": "number"
            },
            "return_adjustment": {
                "type": "number"
            },
            "risk_score": {
                "type": "number"
            },
            "confidence": {
                "type": "number"
            },
            "decision": {
                "type": "string"
            },
            "concise_reason": {
                "type": "string"
            }
        },
        "required": [
            "directional_score",
            "return_adjustment",
            "risk_score",
            "confidence",
            "decision",
            "concise_reason"
        ]
    }

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        data = json.loads(
            response.text
        )

        result = {
            "directional_score": clip01(
                safe_float(
                    data.get(
                        "directional_score"
                    ),
                    0.5
                )
            ),
            "return_adjustment": float(
                np.clip(
                    safe_float(
                        data.get(
                            "return_adjustment"
                        ),
                        0.0
                    ),
                    -0.03,
                    0.03
                )
            ),
            "risk_score": clip01(
                safe_float(
                    data.get(
                        "risk_score"
                    ),
                    0.5
                )
            ),
            "confidence": clip01(
                safe_float(
                    data.get(
                        "confidence"
                    ),
                    0.0
                )
            ),
            "decision": str(
                data.get(
                    "decision",
                    "NEUTRAL"
                )
            ).upper(),
            "reason": str(
                data.get(
                    "concise_reason",
                    ""
                )
            )[:300]
        }

        return result

    except Exception as exc:

        print(
            f"Gemini failed for "
            f"{symbol_name(str(row['symbol']))}: {exc}"
        )

        return None


# ============================================================================
# ENSEMBLE
# ============================================================================

def ensemble_score(
    quant_probability: float,
    quant_return: float,
    gemini: Optional[Dict[str, Any]]
):

    if gemini is None:

        return {
            "ensemble_probability":
                quant_probability,

            "ensemble_return":
                quant_return,

            "risk_penalty":
                0.0,

            "gemini_used":
                False
        }

    gp = gemini["directional_score"]

    ga = gemini["return_adjustment"]

    risk = gemini["risk_score"]

    confidence = gemini["confidence"]

    # Confidence-weight Gemini.
    effective_gemini_weight = (
        GEMINI_WEIGHT * confidence
    )

    effective_quant_weight = (
        1.0 - effective_gemini_weight
    )

    probability = (
        effective_quant_weight
        * quant_probability
        +
        effective_gemini_weight
        * gp
    )

    predicted_return = (
        quant_return
        +
        effective_gemini_weight
        * ga
    )

    # Risk penalty is deliberately modest.
    risk_penalty = (
        0.15 * risk
    )

    probability = (
        probability
        - risk_penalty * 0.05
    )

    return {
        "ensemble_probability":
            clip01(probability),

        "ensemble_return":
            predicted_return,

        "risk_penalty":
            risk_penalty,

        "gemini_used":
            True
    }


# ============================================================================
# CURRENT SIGNAL GENERATION
# ============================================================================

def latest_rows(
    dataset: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for symbol, group in dataset.groupby(
        "symbol",
        sort=False
    ):

        group = group.sort_index()

        if group.empty:
            continue

        rows.append(
            group.iloc[-1]
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def generate_current_signals(
    dataset: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:

    latest = latest_rows(dataset)

    if latest.empty:
        return latest

    # Train only on observations that are genuinely historical
    # relative to the latest signal date.
    signal_date = dataset.index.max()

    train = dataset[
        dataset.index < signal_date
    ].copy()

    train = train.dropna(
        subset=[
            f"future_return_{horizon}",
            f"target_direction_{horizon}"
        ]
    )

    model = fit_model(
        train,
        horizon
    )

    latest = predict_model(
        model,
        latest
    )

    client = get_gemini_client()

    results = []

    for _, row in latest.iterrows():

        gemini = None

        if client is not None:

            gemini = call_gemini(
                client,
                row
            )

            # Avoid hammering API.
            time.sleep(0.15)

        ens = ensemble_score(
            row["quant_probability"],
            row["quant_return"],
            gemini
        )

        r = row.to_dict()

        r.update(
            ens
        )

        r["gemini"] = gemini

        results.append(r)

    return pd.DataFrame(results)


# ============================================================================
# TRADE FILTER
# ============================================================================

def qualify(
    row: pd.Series
) -> bool:

    p = safe_float(
        row.get(
            "ensemble_probability"
        ),
        0.0
    )

    ret = safe_float(
        row.get(
            "ensemble_return"
        ),
        -999
    )

    vol = safe_float(
        row.get(
            "vol_20"
        ),
        np.nan
    )

    if p < MIN_PROBABILITY:
        return False

    if ret < MIN_PREDICTED_RETURN:
        return False

    if np.isfinite(vol):
        if vol > MAX_VOLATILITY:
            return False

    return True


# ============================================================================
# TELEGRAM
# ============================================================================

def send_telegram(
    message: str
) -> bool:

    if not TELEGRAM_BOT_TOKEN:
        print(
            "Telegram: BOT TOKEN NOT CONFIGURED"
        )
        return False

    if not TELEGRAM_CHAT_ID:
        print(
            "Telegram: CHAT ID NOT CONFIGURED"
        )
        return False

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_BOT_TOKEN
        + "/sendMessage"
    )

    try:

        response = requests.post(
            url,
            data={
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    message
            },
            timeout=30
        )

        response.raise_for_status()

        print(
            "Telegram alert sent."
        )

        return True

    except Exception as exc:

        print(
            "Telegram error:",
            exc
        )

        return False


# ============================================================================
# ALERT FORMAT
# ============================================================================

def make_alert(
    results: pd.DataFrame,
    horizon: int
) -> str:

    now = datetime.now(
        timezone.utc
    ).astimezone()

    qualifying = results[
        results.apply(
            qualify,
            axis=1
        )
    ].copy()

    qualifying = qualifying.sort_values(
        [
            "ensemble_probability",
            "ensemble_return"
        ],
        ascending=False
    ).head(
        MAX_STOCKS_IN_ALERT
    )

    lines = []

    lines.append(
        f"V8.0 NSE QUANT + GEMINI ALERT"
    )

    lines.append(
        f"Generated: "
        f"{now.strftime('%Y-%m-%d %H:%M')}"
    )

    lines.append(
        f"Primary horizon: H{horizon}"
    )

    lines.append(
        f"Threshold: "
        f"P≥{MIN_PROBABILITY:.2f}, "
        f"predicted return≥"
        f"{MIN_PREDICTED_RETURN * 100:.2f}%"
    )

    lines.append("")

    if qualifying.empty:

        lines.append(
            "NO QUALIFYING TRADE"
        )

        lines.append(
            "The V8 ensemble did not find "
            "a candidate passing all gates."
        )

    else:

        lines.append(
            "QUALIFYING TRADE CANDIDATE(S)"
        )

        lines.append("")

        for i, (_, row) in enumerate(
            qualifying.iterrows(),
            start=1
        ):

            sym = symbol_name(
                str(row["symbol"])
            )

            p = row[
                "ensemble_probability"
            ]

            ret = row[
                "ensemble_return"
            ]

            close = row[
                "Close"
            ]

            gp = row.get(
                "gemini",
                None
            )

            if isinstance(
                gp,
                dict
            ):

                gemini_line = (
                    f"Gemini={gp['decision']}, "
                    f"risk={gp['risk_score']:.2f}"
                )

            else:

                gemini_line = (
                    "Gemini=UNAVAILABLE"
                )

            lines.append(
                f"{i}. {sym} | "
                f"P={p:.3f} | "
                f"Pred={ret * 100:.2f}% | "
                f"Close={close:.2f}"
            )

            lines.append(
                f"   {gemini_line}"
            )

            if isinstance(
                gp,
                dict
            ) and gp.get("reason"):

                lines.append(
                    f"   {gp['reason']}"
                )

    lines.append("")

    lines.append(
        "Execution model:"
    )

    lines.append(
        "Signal T close → "
        "Entry T+1 open → "
        f"Exit T+{horizon} close"
    )

    lines.append("")

    lines.append(
        f"Round-trip cost assumed: "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    if GEMINI_API_KEY:

        lines.append(
            f"Gemini model: {GEMINI_MODEL}"
        )

        lines.append(
            "Gemini used as a research/risk "
            "overlay, not a guaranteed predictor."
        )

    else:

        lines.append(
            "Gemini: NOT CONFIGURED — "
            "quant-only mode."
        )

    lines.append("")

    lines.append(
        "Research signal only. "
        "Historical backtests do not "
        "guarantee future performance."
    )

    return "\n".join(lines)


# ============================================================================
# AUDIT
# ============================================================================

def save_audit(
    results: pd.DataFrame
):

    os.makedirs(
        "audit",
        exist_ok=True
    )

    output = []

    for _, row in results.iterrows():

        record = {}

        for key, value in row.to_dict().items():

            if isinstance(
                value,
                dict
            ):

                record[key] = value

            elif isinstance(
                value,
                (np.integer,)
            ):

                record[key] = int(value)

            elif isinstance(
                value,
                (np.floating,)
            ):

                record[key] = (
                    float(value)
                    if np.isfinite(value)
                    else None
                )

            elif pd.isna(value):

                record[key] = None

            else:

                record[key] = value

        output.append(record)

    path = (
        "audit/"
        f"v8_signal_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            default=str
        )

    print(
        f"Audit written: {path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)

    print(
        "V8.0 — POINT-IN-TIME "
        "QUANT + GEMINI ENSEMBLE"
    )

    print("=" * 78)

    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    print(
        "Gemini API:",
        "CONFIGURED"
        if GEMINI_API_KEY
        else "NOT CONFIGURED"
    )

    print(
        f"Primary horizon: H{PRIMARY_HORIZON}"
    )

    print(
        f"Probability threshold: "
        f"{MIN_PROBABILITY:.2f}"
    )

    print(
        f"Return threshold: "
        f"{MIN_PREDICTED_RETURN:.2%}"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.2%}"
    )

    print("=" * 78)

    frames = []

    successful = 0

    for symbol in SYMBOLS:

        df = download_symbol(
            symbol
        )

        if df is None:
            continue

        x = build_dataset(
            symbol,
            df
        )

        frames.append(x)

        successful += 1

    if not frames:

        raise RuntimeError(
            "No usable market data."
        )

    dataset = pd.concat(
        frames,
        axis=0
    )

    dataset = dataset.sort_index()

    leakage_check(
        dataset
    )

    print("")
    print("DATASET")
    print(
        f"Total observations: "
        f"{len(dataset):,}"
    )

    print(
        f"Symbols: "
        f"{dataset['symbol'].nunique()}"
    )

    print(
        f"Signal dates: "
        f"{dataset.index.nunique()}"
    )

    print(
        "=" * 78
    )

    results = generate_current_signals(
        dataset,
        PRIMARY_HORIZON
    )

    if results.empty:

        raise RuntimeError(
            "No current signals produced."
        )

    qualifying = results[
        results.apply(
            qualify,
            axis=1
        )
    ].copy()

    qualifying = qualifying.sort_values(
        "ensemble_probability",
        ascending=False
    )

    print("")
    print(
        "CURRENT SIGNALS"
    )

    print(
        f"Candidates: "
        f"{len(results)}"
    )

    print(
        f"Qualifying: "
        f"{len(qualifying)}"
    )

    if not qualifying.empty:

        for _, row in qualifying.head(
            MAX_STOCKS_IN_ALERT
        ).iterrows():

            print(
                symbol_name(
                    str(row["symbol"])
                ),
                f"P={row['ensemble_probability']:.3f}",
                f"Pred={row['ensemble_return']:.2%}"
            )

    save_audit(
        results
    )

    message = make_alert(
        results,
        PRIMARY_HORIZON
    )

    print("")
    print("=" * 78)
    print(message)
    print("=" * 78)

    send_telegram(
        message
    )


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print(
            "FATAL ERROR"
        )

        traceback.print_exc()

        # Attempt to alert Telegram about failure.
        try:

            send_telegram(
                "V8.0 NSE ALERT FAILED\n\n"
                + traceback.format_exc()[-3500:]
            )

        except Exception:
            pass

        sys.exit(1)
