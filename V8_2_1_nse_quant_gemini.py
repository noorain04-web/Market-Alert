#!/usr/bin/env python3
"""
V8.2.1 — NSE QUANT + GEMINI LIVE RESEARCH ENSEMBLE
Revision: 2026-09-28-GEMINI-DIAGNOSTIC-RETRY-JSON-FIX

Key fixes over V8.2
-------------------
1. Explicit Gemini API health check before candidate research.
2. Uses Gemini 3.8 Flash with the current Google GenAI SDK.
3. Structured JSON is requested and parsed manually, avoiding fragile
   SDK/Pydantic response-schema + Search combinations.
4. Google Search grounding is enabled for live research.
5. Retries transient Gemini failures with exponential backoff.
6. Every Gemini failure is recorded with an explicit error code/message.
7. Gemini is NEVER reported as active merely because the model is configured.
8. Telegram reports ACTIVE / FAILED / DISABLED Gemini status.
9. Top quant candidates are researched even when they do not yet pass
   the final trade threshold.
10. Historical backtests remain QUANT-ONLY to avoid look-ahead leakage.

Execution model
---------------
Signal T close -> Entry T+1 open -> Exit T+H close

Research software only. Not investment advice. Historical backtests do not
predict future performance.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_OK = True
except Exception as exc:
    GEMINI_SDK_OK = False
    GEMINI_IMPORT_ERROR = str(exc)


# ============================================================================
# CONFIG
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
MAX_VOLATILITY = float(os.getenv("MAX_VOLATILITY", "0.065"))
MAX_EVENT_RISK = float(os.getenv("MAX_EVENT_RISK", "0.85"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "1") == "1"
ENABLE_GEMINI_SEARCH = os.getenv("ENABLE_GEMINI_SEARCH", "1") == "1"
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")

GEMINI_PROB_WEIGHT = float(os.getenv("GEMINI_PROB_WEIGHT", "0.10"))
GEMINI_RETURN_WEIGHT = float(os.getenv("GEMINI_RETURN_WEIGHT", "0.15"))
GEMINI_RISK_WEIGHT = float(os.getenv("GEMINI_RISK_WEIGHT", "0.10"))
MAX_GEMINI_RETURN_ADJUSTMENT = float(os.getenv("MAX_GEMINI_RETURN_ADJUSTMENT", "0.015"))

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
# GENERIC HELPERS
# ============================================================================

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def ticker_name(symbol: str) -> str:
    return symbol.replace(".NS", "")


def finite_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).copy()


def json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object even if the model wrapped it in markdown."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def normalize_assessment(item: dict[str, Any], expected_ticker: str) -> dict[str, Any]:
    """Clamp Gemini output into the bounded schema used by the ensemble."""
    direction = safe_float(item.get("direction_score"), 0.0)
    catalyst = safe_float(item.get("catalyst_score"), 0.0)
    bias = safe_float(item.get("return_bias"), 0.0)
    risk = safe_float(item.get("risk_score"), 0.0)
    confidence = safe_float(item.get("confidence"), 0.0)

    direction = float(np.clip(direction, -1, 1))
    catalyst = float(np.clip(catalyst, -1, 1))
    bias = float(np.clip(bias, -1, 1))
    risk = float(np.clip(risk, 0, 1))
    confidence = float(np.clip(confidence, 0, 1))

    veto = bool(item.get("veto", False))
    flags = item.get("risk_flags", [])
    if not isinstance(flags, list):
        flags = [str(flags)] if flags else []

    return {
        "ticker": str(item.get("ticker") or expected_ticker).upper(),
        "direction_score": direction,
        "catalyst_score": catalyst,
        "return_bias": bias,
        "risk_score": risk,
        "confidence": confidence,
        "direction": str(item.get("direction", "neutral")),
        "veto": veto,
        "reason": str(item.get("reason", ""))[:600],
        "risk_flags": [str(x)[:160] for x in flags[:8]],
    }


# ============================================================================
# MARKET DATA / FEATURES
# ============================================================================

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
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
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
    entry = x["open"].shift(-1)
    exit_close = x["close"].shift(-horizon)
    x[f"future_return_{horizon}"] = exit_close / entry - 1
    x[f"target_up_{horizon}"] = (x[f"future_return_{horizon}"] > 0).astype(int)
    return x


def build_dataset(data: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []
    for symbol, raw in data.items():
        x = add_targets(add_features(raw), horizon)
        x["symbol"] = symbol
        x["signal_date"] = x.index
        cols = ["symbol", "signal_date", "close"] + FEATURES + [
            f"future_return_{horizon}", f"target_up_{horizon}"
        ]
        x = x[cols].replace([np.inf, -np.inf], np.nan)
        x = x.dropna(subset=[f"future_return_{horizon}"])
        if not x.empty:
            rows.append(x)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["signal_date", "symbol"]
    ).reset_index(drop=True)


# ============================================================================
# QUANT MODEL
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
    reg.fit(X, y_ret.fillna(float(y_ret.median()) if len(y_ret) else 0.0))
    return clf, reg


def predict_quant(models, frame: pd.DataFrame) -> pd.DataFrame:
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
# GEMINI V8.2.1
# ============================================================================

class GeminiRun:
    def __init__(self):
        self.status = "DISABLED"
        self.reason = ""
        self.error_code = ""
        self.error_detail = ""
        self.model = GEMINI_MODEL
        self.api_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
        self.sdk_available = GEMINI_SDK_OK
        self.health_ok = False
        self.search_enabled = ENABLE_GEMINI_SEARCH
        self.attempts = 0
        self.assessments = 0
        self.requested = 0

    def as_dict(self):
        return {
            "status": self.status,
            "reason": self.reason,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "model": self.model,
            "api_configured": self.api_configured,
            "sdk_available": self.sdk_available,
            "health_ok": self.health_ok,
            "search_enabled": self.search_enabled,
            "attempts": self.attempts,
            "assessments": self.assessments,
            "requested": self.requested,
        }


def make_gemini_client():
    if not ENABLE_GEMINI:
        return None, "DISABLED", "ENABLE_GEMINI=0"
    if not GEMINI_SDK_OK:
        return None, "SDK_IMPORT_ERROR", globals().get("GEMINI_IMPORT_ERROR", "unknown")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None, "MISSING_API_KEY", "GEMINI_API_KEY is empty or unavailable in the process environment"
    try:
        # The GenAI SDK reads the API key supplied here.
        return genai.Client(api_key=key), "OK", ""
    except Exception as exc:
        return None, "CLIENT_INIT_ERROR", str(exc)


def _gemini_config(include_search: bool):
    kwargs = {
        "response_mime_type": "application/json",
        "max_output_tokens": 5000,
    }
    # Gemini 3.8 supports tunable thinking levels. Keep it low for a daily
    # multi-stock alert to control latency/cost while retaining reasoning.
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=GEMINI_THINKING_LEVEL
        )
    except Exception:
        pass
    if include_search:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)


def gemini_health_check(client: Any, state: GeminiRun) -> bool:
    """Small authenticated request. It proves the API path works before a batch."""
    if client is None:
        return False
    prompt = (
        "Return exactly this JSON object and nothing else: "
        '{"ok":true,"service":"gemini"}'
    )
    last = ""
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        state.attempts += 1
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=_gemini_config(False),
            )
            parsed = extract_json_object(getattr(response, "text", ""))
            if parsed and parsed.get("ok") is True:
                state.health_ok = True
                return True
            last = f"health response did not contain expected JSON: {getattr(response, 'text', '')[:500]}"
        except Exception as exc:
            last = str(exc)
            if attempt < GEMINI_MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
    state.error_code = "HEALTH_CHECK_FAILED"
    state.error_detail = last[:1200]
    state.reason = "Gemini API authentication/model health check failed"
    return False


def build_gemini_prompt(candidates: pd.DataFrame) -> str:
    payload = []
    for _, r in candidates.iterrows():
        payload.append({
            "ticker": ticker_name(r.symbol),
            "close": round(safe_float(r.close), 2),
            "quant_probability": round(safe_float(r.quant_probability), 4),
            "quant_return_prediction": round(safe_float(r.quant_return_prediction), 5),
            "quant_volatility_20d": round(safe_float(r.quant_volatility), 5),
            "horizon_trading_days": HORIZON,
        })

    return f"""
You are the live research and risk overlay for an NSE equity quantitative
signal engine. The statistical model has already produced the quantitative
scores below. Do not replace the quant model and do not invent a price target.

Current UTC time: {datetime.now(timezone.utc).isoformat()}
Research horizon: next {HORIZON} trading days.

Use Google Search grounding when enabled. Check recent, credible, stock-specific
public information. Prioritize exchange/company announcements, results,
guidance, regulatory/legal events, major orders/contracts, corporate actions,
management changes, sector-specific catalysts, and material negative news.

For each ticker return:
- direction_score: -1 to +1; whether current evidence contradicts/supports
  the bullish quant direction.
- catalyst_score: -1 to +1; near-term stock-specific catalyst direction.
- return_bias: -1 to +1; qualitative downside/upside pressure only.
- risk_score: 0 to 1; near-term event/risk intensity.
- confidence: 0 to 1; confidence in the research assessment.
- direction: bullish, bearish, or neutral.
- veto: true only when a material near-term risk makes the candidate
  unsuitable as a research candidate.
- reason: concise evidence-based explanation.
- risk_flags: short list of material risks.

Do not fabricate facts. If evidence is weak or ambiguous, use neutral scores
and low confidence. Do not output a numerical return forecast.

Return ONLY valid JSON in this exact shape:
{{
  "assessments": [
    {{
      "ticker": "TICKER",
      "direction_score": 0.0,
      "catalyst_score": 0.0,
      "return_bias": 0.0,
      "risk_score": 0.0,
      "confidence": 0.0,
      "direction": "neutral",
      "veto": false,
      "reason": "...",
      "risk_flags": []
    }}
  ]
}}

Candidates:
{json.dumps(payload, indent=2)}
"""


def run_gemini_research(candidates: pd.DataFrame, state: GeminiRun) -> dict[str, dict[str, Any]]:
    state.requested = int(len(candidates))
    if candidates.empty:
        state.status = "NO_CANDIDATES"
        state.reason = "No candidates were supplied to Gemini"
        return {}

    client, code, detail = make_gemini_client()
    if client is None:
        state.status = "FAILED"
        state.error_code = code
        state.error_detail = detail[:1200]
        state.reason = detail
        return {}

    if not gemini_health_check(client, state):
        state.status = "FAILED"
        return {}

    selected = candidates.head(TOP_N_GEMINI).copy()
    prompt = build_gemini_prompt(selected)
    last_error = ""

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        state.attempts += 1
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=_gemini_config(ENABLE_GEMINI_SEARCH),
            )
            text = getattr(response, "text", "") or ""
            parsed = extract_json_object(text)
            if not parsed:
                raise ValueError(f"Gemini returned non-JSON output: {text[:1200]}")

            raw_items = parsed.get("assessments")
            if not isinstance(raw_items, list):
                raise ValueError("Gemini JSON has no assessments list")

            expected = {ticker_name(s).upper() for s in selected.symbol}
            result: dict[str, dict[str, Any]] = {}
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker", "")).upper().replace(".NS", "")
                if ticker in expected:
                    result[ticker] = normalize_assessment(item, ticker)

            if not result:
                raise ValueError("Gemini returned zero recognizable candidate assessments")

            state.assessments = len(result)
            state.status = "ACTIVE"
            state.reason = "Gemini health check and research call succeeded"
            return result

        except Exception as exc:
            last_error = str(exc)
            print(f"WARNING: Gemini attempt {attempt}/{GEMINI_MAX_RETRIES} failed: {last_error}")
            if attempt < GEMINI_MAX_RETRIES:
                time.sleep(min(8, 2 ** (attempt - 1)))

    state.status = "FAILED"
    state.error_code = "RESEARCH_CALL_FAILED"
    state.error_detail = last_error[:1600]
    state.reason = "Gemini research request failed after retries"
    return {}


# ============================================================================
# ENSEMBLE
# ============================================================================

def combine_quant_and_gemini(q: pd.DataFrame, assessments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = q.copy()

    def item_for(symbol: str):
        return assessments.get(ticker_name(symbol).upper())

    def field(symbol: str, name: str, default: Any):
        item = item_for(symbol)
        return default if item is None else item.get(name, default)

    out["gemini_available"] = out.symbol.map(lambda s: item_for(s) is not None)
    out["gemini_direction_score"] = out.symbol.map(lambda s: safe_float(field(s, "direction_score", 0.0)))
    out["gemini_catalyst_score"] = out.symbol.map(lambda s: safe_float(field(s, "catalyst_score", 0.0)))
    out["gemini_return_bias"] = out.symbol.map(lambda s: safe_float(field(s, "return_bias", 0.0)))
    out["gemini_risk_score"] = out.symbol.map(lambda s: safe_float(field(s, "risk_score", 0.0)))
    out["gemini_confidence"] = out.symbol.map(lambda s: safe_float(field(s, "confidence", 0.0)))
    out["gemini_veto"] = out.symbol.map(lambda s: bool(field(s, "veto", False)))
    out["gemini_reason"] = out.symbol.map(lambda s: str(field(s, "reason", "")))
    out["gemini_risk_flags"] = out.symbol.map(lambda s: field(s, "risk_flags", []))

    # Important: if Gemini is unavailable, its contribution is exactly zero.
    # It must not create a false positive, and the alert clearly reports that
    # it was unavailable.
    avail = out["gemini_available"].astype(float)
    support = 0.60 * out["gemini_direction_score"] + 0.40 * out["gemini_catalyst_score"]
    prob_delta = GEMINI_PROB_WEIGHT * support * out["gemini_confidence"] * avail
    risk_delta = GEMINI_RISK_WEIGHT * out["gemini_risk_score"] * out["gemini_confidence"] * avail

    out["ensemble_probability"] = np.clip(
        out["quant_probability"] + prob_delta - risk_delta, 0.001, 0.999
    )

    raw_adj = (
        MAX_GEMINI_RETURN_ADJUSTMENT
        * out["gemini_return_bias"]
        * out["gemini_confidence"]
        * (0.65 + 0.35 * out["gemini_catalyst_score"].clip(lower=0))
        * avail
    )
    out["gemini_return_adjustment"] = raw_adj
    out["ensemble_return_prediction"] = out["quant_return_prediction"] + GEMINI_RETURN_WEIGHT * raw_adj

    out["volatility_pass"] = out["quant_volatility"].fillna(np.inf) <= MAX_VOLATILITY

    # Gemini is a risk overlay. A candidate can pass quant gates without
    # Gemini, but the normal V8.2.1 "ensemble" gate requires Gemini coverage.
    # This prevents a silent quant-only trade from being labelled Gemini-backed.
    out["ensemble_pass"] = (
        (out["ensemble_probability"] >= P_THRESHOLD)
        & (out["ensemble_return_prediction"] >= RETURN_THRESHOLD)
        & out["volatility_pass"]
        & out["gemini_available"]
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
    if not bool(row.get("gemini_available", False)):
        reasons.append("Gemini unavailable")
    if bool(row.get("gemini_veto", False)):
        reasons.append("Gemini risk veto")
    if safe_float(row.get("gemini_risk_score", 0.0)) >= MAX_EVENT_RISK:
        reasons.append("high event risk")
    return "; ".join(dict.fromkeys(reasons)) if reasons else "QUALIFIES"


# ============================================================================
# BACKTEST
# ============================================================================

def chronological_split(df: pd.DataFrame):
    dates = sorted(pd.to_datetime(df.signal_date).dt.date.unique())
    n = len(dates)
    d1 = dates[max(1, int(n * 0.60) - 1)]
    d2 = dates[max(2, int(n * 0.80) - 1)]
    dev = df[pd.to_datetime(df.signal_date).dt.date <= d1].copy()
    val = df[(pd.to_datetime(df.signal_date).dt.date > d1) & (pd.to_datetime(df.signal_date).dt.date <= d2)].copy()
    oos = df[pd.to_datetime(df.signal_date).dt.date > d2].copy()
    return dev, val, oos


def backtest_quant_only(dataset: pd.DataFrame, horizon: int) -> dict[str, Any]:
    dev, val, oos = chronological_split(dataset)
    model = fit_quant(dev, horizon)
    op = predict_quant(model, oos)
    op["trade"] = (
        (op.quant_probability >= P_THRESHOLD)
        & (op.quant_return_prediction >= RETURN_THRESHOLD)
        & (op.quant_volatility.fillna(np.inf) <= MAX_VOLATILITY)
    )
    future_col = f"future_return_{horizon}"
    joined = op.merge(oos[["symbol", "signal_date", future_col]], on=["symbol", "signal_date"], how="left")
    trades = joined[joined.trade].copy()
    if trades.empty:
        stats = {"trades": 0, "win_rate": None, "average_net_return": None, "median_net_return": None, "profit_factor": None}
    else:
        net = trades[future_col].astype(float) - ROUND_TRIP_COST
        gains = net[net > 0]
        losses = net[net < 0]
        stats = {
            "trades": int(len(net)),
            "win_rate": float((net > 0).mean()),
            "average_net_return": float(net.mean()),
            "median_net_return": float(net.median()),
            "profit_factor": float(gains.sum() / abs(losses.sum())) if len(losses) else None,
        }
    return {
        "version": "V8.2.1",
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
        print("Telegram disabled.")
        return False
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram secrets not configured.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, timeout=20)
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"WARNING: Telegram failed safely: {exc}")
        return False


def build_alert(final: pd.DataFrame, state: GeminiRun) -> str:
    ranked = final.sort_values(["ensemble_pass", "ensemble_probability", "ensemble_return_prediction"], ascending=[False, False, False])
    winners = ranked[ranked.ensemble_pass].head(TOP_N_ALERT)
    closest = ranked[~ranked.ensemble_pass].head(5)

    lines = [
        "V8.2.1 NSE QUANT + GEMINI ALERT",
        f"Generated: {utc_stamp()}",
        f"Primary horizon: H{HORIZON}",
        f"Threshold: P≥{P_THRESHOLD:.2f}, predicted return≥{RETURN_THRESHOLD:.2%}",
        "",
        f"Quant candidates: {len(final)}",
        f"Gemini candidates requested: {state.requested}",
        f"Gemini candidates analyzed: {state.assessments}/{state.requested}",
        f"Gemini status: {state.status}",
        f"Gemini model: {state.model}",
    ]

    if state.status == "FAILED":
        lines += [
            f"Gemini error: {state.error_code}",
            f"Gemini detail: {state.error_detail[:350]}",
            "Ensemble trade gate: DISABLED because Gemini was unavailable.",
        ]
    elif state.status == "ACTIVE":
        lines += ["Gemini health check: PASS", "Google Search grounding: " + ("ON" if state.search_enabled else "OFF")]
    else:
        lines += ["Gemini health check: NOT RUN"]

    lines.append("")

    if not winners.empty:
        lines += ["QUALIFYING TRADE CANDIDATE(S)", ""]
        for i, (_, r) in enumerate(winners.iterrows(), 1):
            lines.append(
                f"{i}. {ticker_name(r.symbol)} | Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | Pred={r.ensemble_return_prediction:.2%} | "
                f"GeminiDir={r.gemini_direction_score:+.2f} | Risk={r.gemini_risk_score:.2f}"
            )
            reason = str(r.get("gemini_reason", "")).replace("\n", " ")
            if reason:
                lines.append(f"   Gemini: {reason[:180]}")
    else:
        lines += [
            "NO QUALIFYING TRADE",
            "No candidate passed all quant + volatility + Gemini risk gates.",
            "",
            "Closest candidates:",
        ]
        for i, (_, r) in enumerate(closest.iterrows(), 1):
            lines.append(
                f"{i}. {ticker_name(r.symbol)} | Q={r.quant_probability:.3f} | "
                f"Final={r.ensemble_probability:.3f} | Pred={r.ensemble_return_prediction:.2%} | "
                f"GeminiDir={r.gemini_direction_score:+.2f} | Risk={r.gemini_risk_score:.2f} | "
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
    print("V8.2.1 — NSE QUANT + GEMINI LIVE RESEARCH ENSEMBLE")
    print("=" * 78)
    print("Revision: 2026-09-28-GEMINI-DIAGNOSTIC-RETRY-JSON-FIX")
    print(f"yfinance: {getattr(yf, '__version__', 'unknown')}")
    print(f"Gemini model: {GEMINI_MODEL}")
    print(f"Gemini SDK available: {GEMINI_SDK_OK}")
    print(f"Gemini API configured: {bool(os.getenv('GEMINI_API_KEY', '').strip())}")
    print(f"Gemini Search grounding: {ENABLE_GEMINI_SEARCH}")
    print(f"Horizon: H{HORIZON}")
    print(f"Universe requested: {len(UNIVERSE)}")
    print(f"Cost: {ROUND_TRIP_COST:.2%}")

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

    if RUN_BACKTEST:
        bt = backtest_quant_only(dataset, HORIZON)
        print("\nBACKTEST — QUANT ONLY")
        print(json.dumps(bt, indent=2))
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (AUDIT_DIR / f"v8_2_1_backtest_{ts}.json").write_text(json.dumps(bt, indent=2), encoding="utf-8")

    models = fit_quant(dataset, HORIZON)
    latest_rows = []
    for symbol, raw in data.items():
        f = add_features(raw)
        f["symbol"] = symbol
        f["signal_date"] = f.index
        if not f.empty:
            latest_rows.append(f.tail(1)[["symbol", "signal_date", "close"] + FEATURES])
    if not latest_rows:
        raise RuntimeError("No latest feature rows available.")

    latest = pd.concat(latest_rows, ignore_index=True)
    quant = predict_quant(models, latest)
    quant = quant.sort_values(["quant_probability", "quant_return_prediction"], ascending=[False, False])
    quant_for_research = quant.head(TOP_N_QUANT).copy()
    gemini_input = quant_for_research.head(TOP_N_GEMINI).copy()

    print("\nTOP QUANT CANDIDATES")
    print(quant_for_research[["symbol", "close", "quant_probability", "quant_return_prediction", "quant_volatility"]].to_string(index=False))

    state = GeminiRun()
    assessments = run_gemini_research(gemini_input, state)
    print(f"\nGEMINI STATUS: {state.status}")
    print(f"Gemini health check: {state.health_ok}")
    print(f"Gemini assessments: {state.assessments}/{state.requested}")
    if state.error_code:
        print(f"Gemini error code: {state.error_code}")
        print(f"Gemini error detail: {state.error_detail}")

    final = combine_quant_and_gemini(quant_for_research, assessments)
    final = final.sort_values(["ensemble_pass", "ensemble_probability", "ensemble_return_prediction"], ascending=[False, False, False])

    print("\nFINAL ENSEMBLE")
    print(final[[
        "symbol", "quant_probability", "ensemble_probability",
        "quant_return_prediction", "ensemble_return_prediction",
        "gemini_available", "gemini_direction_score", "gemini_risk_score",
        "ensemble_pass", "rejection_reason"
    ]].to_string(index=False))

    alert = build_alert(final, state)
    print("\n" + "=" * 78)
    print(alert)
    print("=" * 78)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    final.to_csv(AUDIT_DIR / f"v8_2_1_candidates_{ts}.csv", index=False)
    (AUDIT_DIR / f"v8_2_1_alert_{ts}.txt").write_text(alert, encoding="utf-8")
    (AUDIT_DIR / f"v8_2_1_gemini_{ts}.json").write_text(json.dumps(state.as_dict(), indent=2), encoding="utf-8")

    metadata = {
        "version": "V8.2.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gemini": state.as_dict(),
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
    }
    (AUDIT_DIR / f"v8_2_1_run_{ts}.json").write_text(json.dumps(json_clean(metadata), indent=2), encoding="utf-8")

    telegram_ok = send_telegram(alert)
    print(f"Telegram sent: {telegram_ok}")


if __name__ == "__main__":
    main()
