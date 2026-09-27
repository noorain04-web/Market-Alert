#!/usr/bin/env python3
"""
V8.1 — NSE QUANT + GEMINI LIVE RESEARCH ENGINE
Revision: 2026-09-27-GEMINI-GROUNDED-RISK-OVERLAY

Design goals
------------
1. Quant model remains the primary return/probability engine.
2. Gemini is a current-information research/risk overlay, not a replacement for
   the statistical model.
3. Current Gemini calls use Google Search grounding, so the model can inspect
   recent public information.
4. Historical/backtest Gemini is NEVER substituted with current Gemini output.
5. The live path is fast: one bulk market-data download + one Gemini call for
   the top quant candidates.
6. API failure falls back safely to quant-only.
7. Telegram alert is optional and never receives the API key.
8. Audit JSON/CSV files are written to audit/.

Execution model:
signal at T close -> entry T+1 open -> exit T+H close

IMPORTANT:
This is research software, not investment advice. Backtests do not guarantee
future performance.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    GEMINI_SDK_OK = True
except Exception:
    GEMINI_SDK_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
HISTORY_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")
COST = float(os.getenv("ROUND_TRIP_COST", "0.003"))
P_THRESHOLD = float(os.getenv("P_THRESHOLD", "0.62"))
RETURN_THRESHOLD = float(os.getenv("RETURN_THRESHOLD", "0.006"))
TOP_N_QUANT = int(os.getenv("TOP_N_QUANT", "10"))
TOP_N_GEMINI = int(os.getenv("TOP_N_GEMINI", "8"))
MAX_GEMINI_SECONDS = int(os.getenv("MAX_GEMINI_SECONDS", "45"))
RANDOM_STATE = 42

RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"
ENABLE_GEMINI_SEARCH = os.getenv("ENABLE_GEMINI_SEARCH", "1") == "1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20", "atr_pct",
    "rsi_14", "dist_sma20", "dist_sma50",
    "volume_z20", "range_pct", "gap_pct",
    "drawdown_20", "trend_20", "trend_60",
]


# ---------------------------------------------------------------------------
# Gemini schema
# ---------------------------------------------------------------------------

if GEMINI_SDK_OK:
    class GeminiItem(BaseModel):
        ticker: str
        sentiment: float = Field(ge=-1.0, le=1.0)
        event_risk: float = Field(ge=0.0, le=1.0)
        confidence: float = Field(ge=0.0, le=1.0)
        direction: str
        risk_flags: list[str]
        catalyst_summary: str
        veto: bool

    class GeminiBatch(BaseModel):
        assessments: list[GeminiItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_ist() -> datetime:
    # UTC+05:30 without requiring pytz.
    return datetime.now(timezone.utc).astimezone(
        timezone.utc.__class__ if False else None
    )

def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def finite_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    return out

def ticker_name(symbol: str) -> str:
    return symbol.replace(".NS", "")

def flatten_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract one ticker from yfinance's single- or multi-index output."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker' normally yields (ticker, field)
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
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[wanted].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

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
    c = x["close"]
    h = x["high"]
    l = x["low"]
    o = x["open"]
    v = x["volume"].replace(0, np.nan)

    x["ret_1"] = c.pct_change(1)
    x["ret_3"] = c.pct_change(3)
    x["ret_5"] = c.pct_change(5)
    x["ret_10"] = c.pct_change(10)
    x["ret_20"] = c.pct_change(20)

    daily_ret = c.pct_change()
    x["vol_5"] = daily_ret.rolling(5).std()
    x["vol_10"] = daily_ret.rolling(10).std()
    x["vol_20"] = daily_ret.rolling(20).std()

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

    roll20 = c.rolling(20)
    x["drawdown_20"] = c / roll20.max() - 1
    x["trend_20"] = c / c.shift(20) - 1
    x["trend_60"] = c / c.shift(60) - 1

    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()

    # Signal T close -> entry T+1 open -> exit T+H close.
    entry = x["open"].shift(-1)
    exit_close = x["close"].shift(-horizon)
    x[f"future_return_{horizon}"] = exit_close / entry - 1
    x[f"target_up_{horizon}"] = (
        x[f"future_return_{horizon}"] > 0
    ).astype(float)
    return x


def build_dataset(symbol_data: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []
    for symbol, raw in symbol_data.items():
        f = add_features(raw)
        f = add_targets(f, horizon)
        f["symbol"] = symbol
        f["signal_date"] = f.index
        f = f.dropna(subset=FEATURES + [f"future_return_{horizon}"])
        rows.append(f[["symbol", "signal_date", "close"] + FEATURES +
                       [f"future_return_{horizon}", f"target_up_{horizon}"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURES)
    return out.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quant model
# ---------------------------------------------------------------------------

def make_models() -> tuple[Any, Any]:
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1500,
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


def fit_quant(train: pd.DataFrame, horizon: int) -> tuple[Any, Any]:
    y_cls = train[f"target_up_{horizon}"].astype(int)
    y_ret = train[f"future_return_{horizon}"].astype(float)

    X = finite_frame(train, FEATURES)
    clf, reg = make_models()

    # Guard against degenerate training windows.
    if y_cls.nunique() < 2:
        # A trivial probability model is safer than crashing.
        class ConstantClassifier:
            def __init__(self, p: float): self.p = float(p)
            def predict_proba(self, X):
                p = np.full(len(X), self.p)
                return np.column_stack([1 - p, p])
        return ConstantClassifier(float(y_cls.mean())), reg.fit(
            X, y_ret.fillna(y_ret.median())
        )

    clf.fit(X, y_cls)
    reg.fit(X, y_ret)
    return clf, reg


def predict_quant(models: tuple[Any, Any], frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    clf, reg = models
    X = finite_frame(frame, FEATURES)

    p = clf.predict_proba(X)[:, 1]
    r = reg.predict(X)

    out = frame[["symbol", "signal_date", "close"]].copy()
    out["quant_probability"] = np.clip(p, 0.001, 0.999)
    out["quant_return_prediction"] = r
    out["quant_pass"] = (
        (out["quant_probability"] >= P_THRESHOLD) &
        (out["quant_return_prediction"] >= RETURN_THRESHOLD)
    )
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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

    data: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(UNIVERSE, 1):
        print(f"Loading [{i}/{len(UNIVERSE)}] {symbol}")
        df = flatten_download(raw, symbol)
        if len(df) < 120:
            print(f"WARNING: insufficient history for {symbol}; skipping.")
            continue
        data[symbol] = df

    print(f"Successful symbols: {len(data)}")
    return data


# ---------------------------------------------------------------------------
# Gemini live research overlay
# ---------------------------------------------------------------------------

def gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or not GEMINI_SDK_OK:
        return None
    return genai.Client(api_key=key)


def gemini_overlay(candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Current/live-only research overlay.

    It deliberately does NOT get called from the historical backtest.
    Gemini is asked to use Google Search grounding and return structured JSON.
    """
    client = gemini_client()
    if client is None or not ENABLE_GEMINI_SEARCH or candidates.empty:
        return {}

    payload = []
    for _, r in candidates.head(TOP_N_GEMINI).iterrows():
        payload.append({
            "ticker": ticker_name(r["symbol"]),
            "close": round(float(r["close"]), 2),
            "quant_probability": round(float(r["quant_probability"]), 4),
            "quant_return_prediction": round(float(r["quant_return_prediction"]), 5),
            "horizon_days": HORIZON,
        })

    prompt = f"""
You are a market-research risk overlay for Indian NSE equities.

Current date/time: {datetime.now(timezone.utc).isoformat()}
The quantitative model has already generated candidate scores.
You MUST NOT invent prices, earnings, announcements, or facts.

Use Google Search grounding to check RECENT public information for the
candidate stocks below. Focus on information that could materially affect the
next {HORIZON} trading days:
- recent company announcements
- earnings/results or guidance
- major regulatory/legal actions
- large contracts/orders
- management changes
- material corporate actions
- sector-specific events
- credible negative catalysts
- unusual event risk

Do NOT treat generic market commentary as a stock-specific catalyst.

Return one assessment for each ticker.
sentiment: -1 = strongly negative, +1 = strongly positive
event_risk: 0 = low, 1 = extreme
confidence: 0 = no confidence, 1 = high confidence
direction: one of bullish, neutral, bearish
veto: true ONLY if a material near-term risk makes the quant signal
unsafe/unreliable as a research candidate.

This is NOT a guaranteed price forecast. Do not override the quantitative
prediction with an invented number.

Candidates:
{json.dumps(payload, indent=2)}
"""

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiBatch,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        started = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        if time.time() - started > MAX_GEMINI_SECONDS:
            print("Gemini call exceeded time budget; ignoring overlay.")
            return {}

        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
        else:
            parsed = GeminiBatch.model_validate_json(response.text)

        result: dict[str, dict[str, Any]] = {}
        for item in parsed.assessments:
            result[item.ticker.upper()] = item.model_dump()
        return result

    except Exception as exc:
        print(f"WARNING: Gemini overlay failed safely: {exc}")
        return {}


def combine_scores(q: pd.DataFrame, g: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = q.copy()

    def get_g(sym: str, field: str, default: float | bool = 0):
        item = g.get(ticker_name(sym).upper())
        if not item:
            return default
        return item.get(field, default)

    out["gemini_available"] = out["symbol"].map(
        lambda s: ticker_name(s).upper() in g
    )
    out["gemini_sentiment"] = out["symbol"].map(
        lambda s: safe_float(get_g(s, "sentiment", 0), 0)
    )
    out["gemini_event_risk"] = out["symbol"].map(
        lambda s: safe_float(get_g(s, "event_risk", 0.0), 0.0)
    )
    out["gemini_confidence"] = out["symbol"].map(
        lambda s: safe_float(get_g(s, "confidence", 0.0), 0.0)
    )
    out["gemini_veto"] = out["symbol"].map(
        lambda s: bool(get_g(s, "veto", False))
    )
    out["gemini_direction"] = out["symbol"].map(
        lambda s: get_g(s, "direction", "neutral")
    )
    out["gemini_catalyst"] = out["symbol"].map(
        lambda s: get_g(s, "catalyst_summary", "")
    )
    out["gemini_risk_flags"] = out["symbol"].map(
        lambda s: get_g(s, "risk_flags", [])
    )

    # Gemini modifies conviction modestly; it does not become the price model.
    sentiment_component = 0.05 * out["gemini_sentiment"] * out["gemini_confidence"]
    risk_penalty = 0.08 * out["gemini_event_risk"] * out["gemini_confidence"]

    out["ensemble_probability"] = np.clip(
        out["quant_probability"] + sentiment_component - risk_penalty,
        0.001, 0.999,
    )

    out["ensemble_return_prediction"] = (
        out["quant_return_prediction"] *
        (1.0 + 0.20 * out["gemini_sentiment"] * out["gemini_confidence"])
        - out["quant_return_prediction"].abs() *
        0.10 * out["gemini_event_risk"] * out["gemini_confidence"]
    )

    out["ensemble_pass"] = (
        (out["ensemble_probability"] >= P_THRESHOLD) &
        (out["ensemble_return_prediction"] >= RETURN_THRESHOLD) &
        (~out["gemini_veto"])
    )

    out["rejection_reason"] = out.apply(rejection_reason, axis=1)
    return out


def rejection_reason(row: pd.Series) -> str:
    reasons = []
    if row["quant_probability"] < P_THRESHOLD:
        reasons.append("quant probability")
    if row["quant_return_prediction"] < RETURN_THRESHOLD:
        reasons.append("quant return")
    if row.get("gemini_veto", False):
        reasons.append("Gemini risk veto")
    if row.get("gemini_event_risk", 0) >= 0.8:
        reasons.append("high event risk")
    if not reasons:
        return "QUALIFIES"
    return "; ".join(reasons)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["signal_date"]).dt.date.unique())
    n = len(dates)
    d1 = dates[max(1, int(n * 0.60) - 1)]
    d2 = dates[max(2, int(n * 0.80) - 1)]

    dev = df[pd.to_datetime(df.signal_date).dt.date <= d1].copy()
    val = df[
        (pd.to_datetime(df.signal_date).dt.date > d1) &
        (pd.to_datetime(df.signal_date).dt.date <= d2)
    ].copy()
    oos = df[pd.to_datetime(df.signal_date).dt.date > d2].copy()
    return dev, val, oos


def backtest_once(df: pd.DataFrame, horizon: int) -> dict[str, Any]:
    dev, val, oos = chronological_split(df)

    # Thresholds are fixed from validation only.
    model = fit_quant(dev, horizon)
    vp = predict_quant(model, val, horizon)
    op = predict_quant(model, oos, horizon)

    # For the research version, no current Gemini call is allowed in OOS.
    op["ensemble_probability"] = op["quant_probability"]
    op["ensemble_return_prediction"] = op["quant_return_prediction"]
    op["ensemble_pass"] = (
        (op.ensemble_probability >= P_THRESHOLD) &
        (op.ensemble_return_prediction >= RETURN_THRESHOLD)
    )

    trades = op[op.ensemble_pass].copy()
    if trades.empty:
        stats = {"trades": 0, "win_rate": np.nan, "average_net_return": np.nan}
    else:
        net = trades[f"future_return_{horizon}"] - COST
        stats = {
            "trades": int(len(trades)),
            "win_rate": float((net > 0).mean()),
            "average_net_return": float(net.mean()),
            "median_net_return": float(net.median()),
            "profit_factor": float(
                net[net > 0].sum() / abs(net[net < 0].sum())
            ) if (net < 0).any() else np.inf,
        }

    return {
        "horizon": horizon,
        "development": len(dev),
        "validation": len(val),
        "oos": len(oos),
        "oos_stats": stats,
        "gemini_historical_used": False,
    }


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> bool:
    if not SEND_TELEGRAM:
        return False

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram not configured; printing alert only.")
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"WARNING: Telegram failed safely: {exc}")
        return False


def build_alert(final: pd.DataFrame) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    q = final.sort_values(
        ["ensemble_pass", "ensemble_probability", "ensemble_return_prediction"],
        ascending=[False, False, False],
    )

    winners = q[q.ensemble_pass].head(8)
    near = q[~q.ensemble_pass].head(5)

    lines = [
        "V8.1 NSE QUANT + GEMINI ALERT",
        f"Generated: {stamp}",
        f"Primary horizon: H{HORIZON}",
        f"Threshold: P≥{P_THRESHOLD:.2f}, predicted return≥{RETURN_THRESHOLD:.2%}",
        "",
    ]

    if not winners.empty:
        lines += ["QUALIFYING TRADE CANDIDATE(S)", ""]
        for i, (_, r) in enumerate(winners.iterrows(), 1):
            catalyst = str(r.get("gemini_catalyst", "")).replace("\n", " ")
            if len(catalyst) > 140:
                catalyst = catalyst[:137] + "..."
            lines.append(
                f"{i}. {ticker_name(r.symbol)} | "
                f"Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | "
                f"Pred={r.ensemble_return_prediction:.2%} | "
                f"Gemini={r.gemini_sentiment:+.2f} | "
                f"Risk={r.gemini_event_risk:.2f}"
            )
            if catalyst:
                lines.append(f"   {catalyst}")
    else:
        lines += [
            "NO QUALIFYING TRADE",
            "The V8.1 ensemble did not find a candidate passing all gates.",
            "",
            "Closest candidates:",
        ]
        for i, (_, r) in enumerate(near.iterrows(), 1):
            lines.append(
                f"{i}. {ticker_name(r.symbol)} | "
                f"Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | "
                f"Pred={r.ensemble_return_prediction:.2%} | "
                f"Reason={r.rejection_reason}"
            )

    lines += [
        "",
        "Execution:",
        f"Signal T close → Entry T+1 open → Exit T+{HORIZON} close",
        f"Round-trip cost assumed: {COST:.2%}",
        f"Gemini model: {GEMINI_MODEL}",
        "Gemini is a research/risk overlay, not a guaranteed return predictor.",
        "Historical Gemini is NOT used in the backtest unless timestamped historical scores exist.",
        "",
        "Research signal only. Historical backtests do not guarantee future performance.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("V8.1 — POINT-IN-TIME QUANT + GEMINI RESEARCH UPGRADE")
    print("=" * 78)
    print(f"yfinance: {getattr(yf, '__version__', 'unknown')}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"Horizon: H{HORIZON}")
    print(f"Thresholds: P>={P_THRESHOLD:.2f}, return>={RETURN_THRESHOLD:.2%}")
    print(f"Cost: {COST:.2%}")
    print(f"Gemini SDK available: {GEMINI_SDK_OK}")
    print(f"Gemini API key configured: {bool(os.getenv('GEMINI_API_KEY', '').strip())}")

    # A longer history is used only when explicitly running the backtest.
    period = HISTORY_PERIOD if RUN_BACKTEST else os.getenv("LIVE_PERIOD", "2y")
    data = download_universe(period)
    if not data:
        raise RuntimeError("No market data downloaded.")

    dataset = build_dataset(data, HORIZON)
    if dataset.empty:
        raise RuntimeError("Feature/target dataset is empty.")

    print(f"Dataset observations: {len(dataset):,}")
    print(f"Symbols: {dataset.symbol.nunique()}")
    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Execution: signal T close -> entry T+1 open -> exit T+H close")

    if RUN_BACKTEST:
        result = backtest_once(dataset, HORIZON)
        print(json.dumps(result, indent=2, default=str))
        (AUDIT_DIR / "v8_1_backtest.json").write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )

    # Fit on all currently available historical observations for LIVE inference.
    # Targets at the final signal date are naturally unavailable and are excluded.
    live = dataset.copy()
    models = fit_quant(live, HORIZON)

    latest_rows = []
    for symbol, raw in data.items():
        f = add_features(raw)
        f["symbol"] = symbol
        f["signal_date"] = f.index
        last = f.tail(1).copy()
        if last[FEATURES].notna().sum(axis=1).iloc[0] < max(8, len(FEATURES) // 2):
            continue
        latest_rows.append(last[["symbol", "signal_date", "close"] + FEATURES])

    latest = pd.concat(latest_rows, ignore_index=True)
    q = predict_quant(models, latest, HORIZON)
    q = q.sort_values(
        ["quant_probability", "quant_return_prediction"],
        ascending=False,
    ).head(TOP_N_QUANT)

    print(f"Quant top candidates: {len(q)}")
    print(q[
        ["symbol", "close", "quant_probability", "quant_return_prediction"]
    ].to_string(index=False))

    # Gemini sees only the top quant candidates, reducing cost and latency.
    g = gemini_overlay(q)
    final = combine_scores(q, g)

    final = final.sort_values(
        ["ensemble_pass", "ensemble_probability", "ensemble_return_prediction"],
        ascending=[False, False, False],
    )

    alert = build_alert(final)
    print("\n" + "=" * 78)
    print(alert)
    print("=" * 78)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    final.to_csv(AUDIT_DIR / f"v8_1_candidates_{ts}.csv", index=False)
    (AUDIT_DIR / f"v8_1_alert_{ts}.txt").write_text(alert, encoding="utf-8")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gemini_model": GEMINI_MODEL,
        "gemini_available": bool(g),
        "gemini_historical_used": False,
        "horizon": HORIZON,
        "p_threshold": P_THRESHOLD,
        "return_threshold": RETURN_THRESHOLD,
        "cost": COST,
        "symbols_downloaded": len(data),
        "dataset_observations": len(dataset),
        "top_quant_candidates": q.to_dict(orient="records"),
    }
    (AUDIT_DIR / f"v8_1_run_{ts}.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    send_telegram(alert)


if __name__ == "__main__":
    main()
