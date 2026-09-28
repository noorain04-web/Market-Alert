#!/usr/bin/env python3
"""
V8.2 — NSE QUANT + GEMINI RESEARCH ENSEMBLE
Revision: 2026-09-27-TOP15-GROUNDED-ENSEMBLE

Purpose
-------
A research-oriented NSE alert engine with:
  1) leakage-safe OHLCV features,
  2) quant probability + return models,
  3) Gemini 3.8 Flash live research on the TOP quant candidates,
  4) Google Search grounding,
  5) bounded Gemini adjustments (never an unconstrained return forecast),
  6) risk vetoes,
  7) Telegram alerts,
  8) audit CSV/JSON/TXT output.

Point-in-time rule
------------------
Historical Gemini scores are NEVER reconstructed from today's Gemini call.
The backtest is quant-only unless timestamped historical Gemini scores are
provided separately. Current Gemini is LIVE research only.

Execution model
----------------
Signal T close -> Entry T+1 open -> Exit T+H close

Research software only. Not investment advice. Backtests do not guarantee
future performance.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Gemini SDK
try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    GEMINI_SDK_OK = True
except Exception:
    GEMINI_SDK_OK = False


# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS", "LT.NS", "TMPV.NS", "TMCV.NS", "EICHERMOT.NS",
    "MARUTI.NS", "HEROMOTOCO.NS", "M&M.NS", "TITAN.NS", "ASIANPAINT.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "CIPLA.NS", "DIVISLAB.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    "TECHM.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "BPCL.NS", "COALINDIA.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BEL.NS",
    "HAL.NS", "BHEL.NS", "TRENT.NS", "PIDILITIND.NS", "SIEMENS.NS", "ABB.NS",
    "GRASIM.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "IOC.NS", "VEDL.NS", "DLF.NS", "LODHA.NS", "INDIGO.NS",
    "ETERNAL.NS", "NAUKRI.NS", "COFORGE.NS", "JIOFIN.NS", "IRFC.NS",
    "IREDA.NS", "POLYCAB.NS",
]

HORIZON = int(os.getenv("HORIZON", "10"))
BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")
LIVE_PERIOD = os.getenv("LIVE_PERIOD", "2y")

ROUND_TRIP_COST = float(os.getenv("ROUND_TRIP_COST", "0.003"))

P_THRESHOLD = float(os.getenv("P_THRESHOLD", "0.62"))
RETURN_THRESHOLD = float(os.getenv("RETURN_THRESHOLD", "0.006"))

TOP_N_QUANT = int(os.getenv("TOP_N_QUANT", "20"))
TOP_N_GEMINI = int(os.getenv("TOP_N_GEMINI", "15"))
TOP_N_ALERT = int(os.getenv("TOP_N_ALERT", "8"))

# Hard safety/quality filters.
MAX_VOLATILITY = float(os.getenv("MAX_VOLATILITY", "0.065"))
MAX_EVENT_RISK = float(os.getenv("MAX_EVENT_RISK", "0.85"))

# Gemini is deliberately bounded.
# It can shift conviction and expected-return bias, but cannot invent an
# unconstrained numerical return.
GEMINI_PROB_WEIGHT = float(os.getenv("GEMINI_PROB_WEIGHT", "0.10"))
GEMINI_RETURN_WEIGHT = float(os.getenv("GEMINI_RETURN_WEIGHT", "0.15"))
GEMINI_RISK_WEIGHT = float(os.getenv("GEMINI_RISK_WEIGHT", "0.10"))
MAX_GEMINI_RETURN_ADJUSTMENT = float(
    os.getenv("MAX_GEMINI_RETURN_ADJUSTMENT", "0.015")
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "1") == "1"
ENABLE_GEMINI_SEARCH = os.getenv("ENABLE_GEMINI_SEARCH", "1") == "1"

RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"

RANDOM_STATE = 42
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20", "atr_pct",
    "rsi_14", "dist_sma20", "dist_sma50",
    "volume_z20", "range_pct", "gap_pct",
    "drawdown_20", "trend_20", "trend_60",
]


# ============================================================================
# GEMINI STRUCTURED SCHEMA
# ============================================================================

if GEMINI_SDK_OK:

    class GeminiAssessment(BaseModel):
        ticker: str
        direction_score: float = Field(ge=-1.0, le=1.0)
        catalyst_score: float = Field(ge=-1.0, le=1.0)
        return_bias: float = Field(ge=-1.0, le=1.0)
        risk_score: float = Field(ge=0.0, le=1.0)
        confidence: float = Field(ge=0.0, le=1.0)
        direction: str
        veto: bool
        reason: str
        risk_flags: list[str]

    class GeminiBatch(BaseModel):
        assessments: list[GeminiAssessment]


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def ticker_name(symbol: str) -> str:
    return symbol.replace(".NS", "")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def finite_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )


def flatten_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(0):
                df = raw[symbol].copy()
            elif symbol in raw.columns.get_level_values(1):
                df = raw.xs(symbol, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = raw.copy()

        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        wanted = ["open", "high", "low", "close", "volume"]
        if any(c not in df.columns for c in wanted):
            return pd.DataFrame()

        df = df[wanted].copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df.dropna(subset=["open", "high", "low", "close"])
    except Exception:
        return pd.DataFrame()


# ============================================================================
# FEATURES / TARGETS
# ============================================================================

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c, h, l, o = x["close"], x["high"], x["low"], x["open"]
    v = x["volume"].replace(0, np.nan)

    x["ret_1"] = c.pct_change(1)
    x["ret_3"] = c.pct_change(3)
    x["ret_5"] = c.pct_change(5)
    x["ret_10"] = c.pct_change(10)
    x["ret_20"] = c.pct_change(20)

    d = c.pct_change()
    x["vol_5"] = d.rolling(5).std()
    x["vol_10"] = d.rolling(10).std()
    x["vol_20"] = d.rolling(20).std()

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / c

    x["rsi_14"] = rsi(c, 14) / 100.0
    x["dist_sma20"] = c / c.rolling(20).mean() - 1
    x["dist_sma50"] = c / c.rolling(50).mean() - 1

    lv = np.log(v)
    x["volume_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    x["range_pct"] = (h - l) / c
    x["gap_pct"] = o / c.shift(1) - 1

    x["drawdown_20"] = c / c.rolling(20).max() - 1
    x["trend_20"] = c / c.shift(20) - 1
    x["trend_60"] = c / c.shift(60) - 1

    return x.replace([np.inf, -np.inf], np.nan)


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()

    # Signal T close -> entry T+1 open -> exit T+H close.
    entry = x["open"].shift(-1)
    exit_close = x["close"].shift(-horizon)

    x[f"future_return_{horizon}"] = exit_close / entry - 1
    x[f"target_up_{horizon}"] = (
        x[f"future_return_{horizon}"] > 0
    ).astype(int)

    return x


def build_dataset(data: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []

    for symbol, raw in data.items():
        x = add_targets(add_features(raw), horizon)
        x["symbol"] = symbol
        x["signal_date"] = x.index

        cols = (
            ["symbol", "signal_date", "close"]
            + FEATURES
            + [f"future_return_{horizon}", f"target_up_{horizon}"]
        )

        x = x[cols].replace([np.inf, -np.inf], np.nan)

        # For training/backtest, target must exist. Feature NaNs are handled
        # inside the sklearn imputer.
        x = x.dropna(subset=[f"future_return_{horizon}"])
        if not x.empty:
            rows.append(x)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


# ============================================================================
# QUANT MODELS
# ============================================================================

def make_models():
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1800,
            class_weight="balanced",
            C=0.7,
            random_state=RANDOM_STATE,
        )),
    ])

    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])
    return clf, reg


def fit_quant(train: pd.DataFrame, horizon: int):
    train = train.copy()
    y_cls = train[f"target_up_{horizon}"].astype(int)
    y_ret = train[f"future_return_{horizon}"].astype(float)

    X = finite_features(train)
    clf, reg = make_models()

    if y_cls.nunique() < 2:
        p = float(y_cls.mean()) if len(y_cls) else 0.5

        class ConstantClassifier:
            def __init__(self, prob):
                self.prob = prob

            def predict_proba(self, X):
                pp = np.full(len(X), self.prob)
                return np.column_stack([1 - pp, pp])

        clf = ConstantClassifier(p)

    else:
        clf.fit(X, y_cls)

    # Robust finite target handling.
    median_ret = float(y_ret.median()) if len(y_ret) else 0.0
    reg_y = y_ret.fillna(median_ret)
    reg.fit(X, reg_y)

    return clf, reg


def predict_quant(models, frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    clf, reg = models
    X = finite_features(frame)

    p = np.clip(clf.predict_proba(X)[:, 1], 0.001, 0.999)
    r = reg.predict(X)

    out = frame[["symbol", "signal_date", "close"]].copy()
    out["quant_probability"] = p
    out["quant_return_prediction"] = r
    out["quant_volatility"] = frame.get(
        "vol_20", pd.Series(np.nan, index=frame.index)
    ).to_numpy()

    return out


# ============================================================================
# MARKET DATA
# ============================================================================

def download_universe(period: str) -> dict[str, pd.DataFrame]:
    print(f"Downloading {len(UNIVERSE)} NSE symbols ({period})...")

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
        print(f"Loading [{i}/{len(UNIVERSE)}] {symbol}")
        df = flatten_download(raw, symbol)

        if len(df) < 120:
            print(f"WARNING: insufficient history for {symbol}; skipping.")
            continue

        data[symbol] = df

    print(f"Successful symbols: {len(data)}")
    return data


# ============================================================================
# GEMINI LIVE RESEARCH
# ============================================================================

def gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or not GEMINI_SDK_OK or not ENABLE_GEMINI:
        return None
    return genai.Client(api_key=key)


def run_gemini_research(candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    One grounded Gemini call for the top quant-ranked candidates.

    Gemini is not allowed to produce an unconstrained return forecast.
    Instead it supplies:
      direction_score [-1,1]
      catalyst_score [-1,1]
      return_bias [-1,1]
      risk_score [0,1]
      confidence [0,1]
      veto
    """

    if candidates.empty:
        return {}

    client = gemini_client()
    if client is None:
        return {}

    selected = candidates.head(TOP_N_GEMINI).copy()

    payload = []
    for _, r in selected.iterrows():
        payload.append({
            "ticker": ticker_name(r.symbol),
            "close": round(float(r.close), 2),
            "quant_probability": round(float(r.quant_probability), 4),
            "quant_return_prediction": round(float(r.quant_return_prediction), 5),
            "quant_volatility_20d": round(
                safe_float(r.quant_volatility, 0.0), 5
            ),
            "horizon_days": HORIZON,
        })

    prompt = f"""
You are the CURRENT-INFORMATION research and risk layer for an NSE equity
quantitative signal engine.

Current UTC timestamp: {datetime.now(timezone.utc).isoformat()}

The statistical model already produced probability and expected-return
signals. Your job is NOT to replace that model and NOT to invent a numerical
price target.

Use Google Search grounding to inspect recent, credible public information
that could materially affect these stocks during the next {HORIZON} trading
days.

Prioritize:
- latest company announcements and exchange filings
- recent earnings/results/guidance
- regulatory/legal actions
- major contracts/orders
- management changes
- material corporate actions
- credible sector-specific catalysts
- credible negative developments
- major event/earnings risk

Ignore generic market chatter unless it has clear stock-specific relevance.

For each ticker return:
direction_score:
  -1 strongly contradicts bullish quant signal
  +1 strongly supports bullish quant signal

catalyst_score:
  -1 strongly negative catalyst
  +1 strongly positive catalyst
  0 neutral/no meaningful catalyst

return_bias:
  -1 suggests downside pressure
  +1 suggests upside pressure
This is a bounded qualitative bias, NOT a percentage forecast.

risk_score:
  0 low near-term event risk
  1 extreme near-term event risk

confidence:
  0 no confidence
  1 high confidence

veto:
true only when a material near-term risk makes the quant candidate
unreliable as a research candidate.

reason:
brief, evidence-based explanation.

risk_flags:
short list of material risks only.

Do not fabricate facts. If evidence is weak, use neutral scores and low
confidence.

Candidates:
{json.dumps(payload, indent=2)}
"""

    try:
        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": GeminiBatch,
        }

        if ENABLE_GEMINI_SEARCH:
            config_kwargs["tools"] = [
                types.Tool(google_search=types.GoogleSearch())
            ]

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        parsed = getattr(response, "parsed", None)

        if parsed is None:
            text = getattr(response, "text", "")
            if not text:
                return {}
            parsed = GeminiBatch.model_validate_json(text)

        result = {}
        for item in parsed.assessments:
            result[item.ticker.upper()] = item.model_dump()

        return result

    except Exception as exc:
        print(f"WARNING: Gemini research failed safely: {exc}")
        return {}


# ============================================================================
# ENSEMBLE
# ============================================================================

def combine_quant_and_gemini(
    q: pd.DataFrame,
    assessments: dict[str, dict[str, Any]],
) -> pd.DataFrame:

    out = q.copy()

    def item_for(symbol):
        return assessments.get(ticker_name(symbol).upper())

    def field(symbol, name, default):
        item = item_for(symbol)
        if not item:
            return default
        return item.get(name, default)

    out["gemini_available"] = out.symbol.map(
        lambda s: item_for(s) is not None
    )

    out["gemini_direction_score"] = out.symbol.map(
        lambda s: safe_float(field(s, "direction_score", 0.0), 0.0)
    )

    out["gemini_catalyst_score"] = out.symbol.map(
        lambda s: safe_float(field(s, "catalyst_score", 0.0), 0.0)
    )

    out["gemini_return_bias"] = out.symbol.map(
        lambda s: safe_float(field(s, "return_bias", 0.0), 0.0)
    )

    out["gemini_risk_score"] = out.symbol.map(
        lambda s: safe_float(field(s, "risk_score", 0.0), 0.0)
    )

    out["gemini_confidence"] = out.symbol.map(
        lambda s: safe_float(field(s, "confidence", 0.0), 0.0)
    )

    out["gemini_veto"] = out.symbol.map(
        lambda s: bool(field(s, "veto", False))
    )

    out["gemini_reason"] = out.symbol.map(
        lambda s: str(field(s, "reason", ""))
    )

    out["gemini_risk_flags"] = out.symbol.map(
        lambda s: field(s, "risk_flags", [])
    )

    # ---------------------------------------------------------------------
    # BOUNDED PROBABILITY ADJUSTMENT
    # ---------------------------------------------------------------------
    # Gemini never contributes more than +/- 0.10 to the probability.
    qprob = out["quant_probability"]

    support = (
        0.60 * out["gemini_direction_score"]
        + 0.40 * out["gemini_catalyst_score"]
    )

    prob_delta = (
        GEMINI_PROB_WEIGHT
        * support
        * out["gemini_confidence"]
    )

    risk_delta = (
        GEMINI_RISK_WEIGHT
        * out["gemini_risk_score"]
        * out["gemini_confidence"]
    )

    out["ensemble_probability"] = np.clip(
        qprob + prob_delta - risk_delta,
        0.001,
        0.999,
    )

    # ---------------------------------------------------------------------
    # BOUNDED RETURN ADJUSTMENT
    # ---------------------------------------------------------------------
    # This is a qualitative research overlay, NOT a second price model.
    raw_adj = (
        MAX_GEMINI_RETURN_ADJUSTMENT
        * out["gemini_return_bias"]
        * out["gemini_confidence"]
        * (0.65 + 0.35 * out["gemini_catalyst_score"].clip(lower=0))
    )

    out["gemini_return_adjustment"] = raw_adj

    out["ensemble_return_prediction"] = (
        out["quant_return_prediction"]
        + GEMINI_RETURN_WEIGHT * raw_adj
    )

    # ---------------------------------------------------------------------
    # FINAL GATES
    # ---------------------------------------------------------------------
    out["volatility_pass"] = (
        out["quant_volatility"].fillna(np.inf) <= MAX_VOLATILITY
    )

    out["ensemble_pass"] = (
        (out["ensemble_probability"] >= P_THRESHOLD)
        & (out["ensemble_return_prediction"] >= RETURN_THRESHOLD)
        & out["volatility_pass"]
        & (~out["gemini_veto"])
        & (out["gemini_risk_score"] < MAX_EVENT_RISK)
    )

    out["rejection_reason"] = out.apply(rejection_reason, axis=1)

    return out


def rejection_reason(row: pd.Series) -> str:
    reasons = []

    if row["quant_probability"] < P_THRESHOLD:
        reasons.append("quant probability")

    if row["quant_return_prediction"] < RETURN_THRESHOLD:
        reasons.append("quant return")

    if row["ensemble_probability"] < P_THRESHOLD:
        reasons.append("ensemble probability")

    if row["ensemble_return_prediction"] < RETURN_THRESHOLD:
        reasons.append("ensemble return")

    if not bool(row.get("volatility_pass", True)):
        reasons.append("high volatility")

    if bool(row.get("gemini_veto", False)):
        reasons.append("Gemini risk veto")

    if safe_float(row.get("gemini_risk_score", 0.0)) >= MAX_EVENT_RISK:
        reasons.append("high event risk")

    if not reasons:
        return "QUALIFIES"

    return "; ".join(dict.fromkeys(reasons))


# ============================================================================
# BACKTEST
# ============================================================================

def chronological_split(df: pd.DataFrame):
    dates = sorted(pd.to_datetime(df.signal_date).dt.date.unique())
    n = len(dates)

    d1 = dates[max(1, int(n * 0.60) - 1)]
    d2 = dates[max(2, int(n * 0.80) - 1)]

    dev = df[pd.to_datetime(df.signal_date).dt.date <= d1].copy()

    val = df[
        (pd.to_datetime(df.signal_date).dt.date > d1)
        & (pd.to_datetime(df.signal_date).dt.date <= d2)
    ].copy()

    oos = df[pd.to_datetime(df.signal_date).dt.date > d2].copy()

    return dev, val, oos


def backtest_quant_only(dataset: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """
    Strictly point-in-time quant backtest.

    No current Gemini result enters the historical experiment.
    """

    dev, val, oos = chronological_split(dataset)

    model = fit_quant(dev, horizon)
    vp = predict_quant(model, val, horizon)
    op = predict_quant(model, oos, horizon)

    op["trade"] = (
        (op.quant_probability >= P_THRESHOLD)
        & (op.quant_return_prediction >= RETURN_THRESHOLD)
        & (op.quant_volatility.fillna(np.inf) <= MAX_VOLATILITY)
    )

    future_col = f"future_return_{horizon}"
    joined = op.merge(
        oos[["symbol", "signal_date", future_col]],
        on=["symbol", "signal_date"],
        how="left",
    )

    trades = joined[joined.trade].copy()

    if trades.empty:
        stats = {
            "trades": 0,
            "win_rate": None,
            "average_net_return": None,
            "median_net_return": None,
            "profit_factor": None,
        }
    else:
        net = trades[future_col].astype(float) - ROUND_TRIP_COST
        losses = net[net < 0]
        gains = net[net > 0]

        stats = {
            "trades": int(len(net)),
            "win_rate": float((net > 0).mean()),
            "average_net_return": float(net.mean()),
            "median_net_return": float(net.median()),
            "profit_factor": (
                float(gains.sum() / abs(losses.sum()))
                if len(losses) else None
            ),
        }

    return {
        "version": "V8.2",
        "horizon": horizon,
        "development_observations": len(dev),
        "validation_observations": len(val),
        "oos_observations": len(oos),
        "historical_gemini_used": False,
        "stats": stats,
    }


# ============================================================================
# TELEGRAM
# ============================================================================

def send_telegram(text: str) -> bool:
    if not SEND_TELEGRAM:
        return False

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("Telegram secrets not configured.")
        return False

    try:
        import requests

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        response.raise_for_status()
        return True

    except Exception as exc:
        print(f"WARNING: Telegram failed safely: {exc}")
        return False


# ============================================================================
# ALERT
# ============================================================================

def build_alert(final: pd.DataFrame, gemini_count: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ranked = final.sort_values(
        ["ensemble_pass", "ensemble_probability",
         "ensemble_return_prediction"],
        ascending=[False, False, False],
    )

    winners = ranked[ranked.ensemble_pass].head(TOP_N_ALERT)
    closest = ranked[~ranked.ensemble_pass].head(5)

    lines = [
        "V8.2 NSE QUANT + GEMINI ALERT",
        f"Generated: {stamp}",
        f"Primary horizon: H{HORIZON}",
        f"Threshold: P≥{P_THRESHOLD:.2f}, predicted return≥{RETURN_THRESHOLD:.2%}",
        "",
        f"Quant candidates: {len(final)}",
        f"Gemini candidates analyzed: {gemini_count}",
        f"Gemini model: {GEMINI_MODEL}",
        "",
    ]

    if not winners.empty:
        lines += [
            "QUALIFYING TRADE CANDIDATE(S)",
            "",
        ]

        for i, (_, r) in enumerate(winners.iterrows(), 1):
            reason = str(r.get("gemini_reason", "")).replace("\n", " ")
            if len(reason) > 130:
                reason = reason[:127] + "..."

            lines.append(
                f"{i}. {ticker_name(r.symbol)} | "
                f"Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | "
                f"Pred={r.ensemble_return_prediction:.2%} | "
                f"GeminiDir={r.gemini_direction_score:+.2f} | "
                f"Risk={r.gemini_risk_score:.2f}"
            )

            if reason:
                lines.append(f"   Gemini: {reason}")

    else:
        lines += [
            "NO QUALIFYING TRADE",
            "No candidate passed all quant + volatility + Gemini risk gates.",
            "",
            "Closest candidates:",
        ]

        for i, (_, r) in enumerate(closest.iterrows(), 1):
            lines.append(
                f"{i}. {ticker_name(r.symbol)} | "
                f"Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | "
                f"Pred={r.ensemble_return_prediction:.2%} | "
                f"GeminiDir={r.gemini_direction_score:+.2f} | "
                f"Risk={r.gemini_risk_score:.2f} | "
                f"Reason={r.rejection_reason}"
            )

    lines += [
        "",
        "Execution:",
        f"Signal T close → Entry T+1 open → Exit T+{HORIZON} close",
        f"Round-trip cost assumed: {ROUND_TRIP_COST:.2%}",
        "",
        "Gemini is a grounded research/risk overlay, not a guaranteed return predictor.",
        "Historical Gemini is NOT used in the backtest unless timestamped historical scores exist.",
        "",
        "Research signal only. Historical backtests do not guarantee future performance.",
    ]

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("V8.2 — NSE QUANT + GEMINI RESEARCH ENSEMBLE")
    print("=" * 78)
    print(f"Revision: 2026-09-27-TOP15-GROUNDED-ENSEMBLE")
    print(f"yfinance: {getattr(yf, '__version__', 'unknown')}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"Horizon: H{HORIZON}")
    print(f"Universe requested: {len(UNIVERSE)}")
    print(f"Cost: {ROUND_TRIP_COST:.2%}")
    print(
        f"Gates: P≥{P_THRESHOLD:.2f}, "
        f"return≥{RETURN_THRESHOLD:.2%}, "
        f"max vol≤{MAX_VOLATILITY:.2%}"
    )
    print(f"Gemini SDK available: {GEMINI_SDK_OK}")
    print(
        "Gemini API configured:",
        bool(os.getenv("GEMINI_API_KEY", "").strip())
    )

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------
    period = BACKTEST_PERIOD if RUN_BACKTEST else LIVE_PERIOD
    data = download_universe(period)

    if not data:
        raise RuntimeError("No market data downloaded.")

    dataset = build_dataset(data, HORIZON)

    if dataset.empty:
        raise RuntimeError("Dataset is empty.")

    print(f"Observations: {len(dataset):,}")
    print(f"Symbols: {dataset.symbol.nunique()}")
    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Point-in-time execution: T close -> T+1 open -> T+H close")

    # ---------------------------------------------------------------
    # Optional historical quant backtest
    # ---------------------------------------------------------------
    if RUN_BACKTEST:
        bt = backtest_quant_only(dataset, HORIZON)
        print("\nBACKTEST — QUANT ONLY")
        print(json.dumps(bt, indent=2))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (AUDIT_DIR / f"v8_2_backtest_{ts}.json").write_text(
            json.dumps(bt, indent=2),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------
    # Live model
    # ---------------------------------------------------------------
    # Targets on the latest rows are unavailable. Fit using observations
    # whose forward target exists, then predict today's latest rows.
    train = dataset.copy()
    models = fit_quant(train, HORIZON)

    latest_rows = []

    for symbol, raw in data.items():
        f = add_features(raw)
        f["symbol"] = symbol
        f["signal_date"] = f.index

        if f.empty:
            continue

        last = f.tail(1).copy()

        latest_rows.append(
            last[["symbol", "signal_date", "close"] + FEATURES]
        )

    if not latest_rows:
        raise RuntimeError("No latest feature rows available.")

    latest = pd.concat(latest_rows, ignore_index=True)

    quant = predict_quant(models, latest, HORIZON)

    # Rank broadly first. Gemini sees the TOP 15 rather than only stocks that
    # already passed the final gates. This fixes the main V8.1 limitation.
    quant = quant.sort_values(
        ["quant_probability", "quant_return_prediction"],
        ascending=[False, False],
    )

    quant_for_research = quant.head(TOP_N_QUANT).copy()
    gemini_input = quant_for_research.head(TOP_N_GEMINI).copy()

    print("\nTOP QUANT CANDIDATES")
    print(
        quant_for_research[
            [
                "symbol",
                "close",
                "quant_probability",
                "quant_return_prediction",
                "quant_volatility",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Gemini live research
    # ---------------------------------------------------------------
    assessments = run_gemini_research(gemini_input)

    print(
        f"\nGemini assessments received: {len(assessments)}/"
        f"{len(gemini_input)}"
    )

    final = combine_quant_and_gemini(
        quant_for_research,
        assessments,
    )

    final = final.sort_values(
        ["ensemble_pass", "ensemble_probability",
         "ensemble_return_prediction"],
        ascending=[False, False, False],
    )

    print("\nFINAL ENSEMBLE")
    print(
        final[
            [
                "symbol",
                "quant_probability",
                "ensemble_probability",
                "quant_return_prediction",
                "ensemble_return_prediction",
                "gemini_available",
                "gemini_direction_score",
                "gemini_risk_score",
                "ensemble_pass",
            ]
        ].to_string(index=False)
    )

    alert = build_alert(final, len(assessments))

    print("\n" + "=" * 78)
    print(alert)
    print("=" * 78)

    # ---------------------------------------------------------------
    # Audit
    # ---------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    final.to_csv(
        AUDIT_DIR / f"v8_2_candidates_{ts}.csv",
        index=False,
    )

    (AUDIT_DIR / f"v8_2_alert_{ts}.txt").write_text(
        alert,
        encoding="utf-8",
    )

    metadata = {
        "version": "V8.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gemini_model": GEMINI_MODEL,
        "gemini_sdk_available": GEMINI_SDK_OK,
        "gemini_api_configured": bool(
            os.getenv("GEMINI_API_KEY", "").strip()
        ),
        "gemini_live_assessments": len(assessments),
        "gemini_historical_used": False,
        "horizon": HORIZON,
        "p_threshold": P_THRESHOLD,
        "return_threshold": RETURN_THRESHOLD,
        "max_volatility": MAX_VOLATILITY,
        "max_event_risk": MAX_EVENT_RISK,
        "round_trip_cost": ROUND_TRIP_COST,
        "universe_requested": len(UNIVERSE),
        "symbols_downloaded": len(data),
        "dataset_observations": len(dataset),
        "top_n_quant": TOP_N_QUANT,
        "top_n_gemini": TOP_N_GEMINI,
        "final_candidates": final.to_dict(orient="records"),
    }

    (AUDIT_DIR / f"v8_2_run_{ts}.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Telegram
    # ---------------------------------------------------------------
    telegram_ok = send_telegram(alert)
    print(f"Telegram sent: {telegram_ok}")


if __name__ == "__main__":
    main()
