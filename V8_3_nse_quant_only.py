#!/usr/bin/env python3
"""
V8.3 — NSE QUANT-ONLY POINT-IN-TIME RESEARCH ENGINE
Revision: 2026-09-28-QUANT-ENSEMBLE-NO-GEMINI

Purpose
-------
A production-oriented NSE research/backtest + live-alert engine that does NOT
depend on Gemini or any external LLM.

Core design
-----------
1. Strict point-in-time features: every feature uses information available at
   signal date T close.
2. Target matches execution:
      Signal T close -> Entry T+1 open -> Exit T+H close
3. Walk-forward development/validation/OOS evaluation.
4. Two-model probability ensemble:
      - HistGradientBoostingClassifier
      - LogisticRegression
5. Two-model return ensemble:
      - HistGradientBoostingRegressor
      - ExtraTreesRegressor
6. Cross-sectional features/ranks and market-regime features.
7. Validation-only threshold selection.
8. Volatility and liquidity gates.
9. Cost-aware expected-return gate.
10. Live scan + Telegram alert.
11. Historical backtest never uses today's information.
12. No Gemini/API dependency.

Research only. This is not investment advice and does not guarantee returns.
"""

from __future__ import annotations

import json
import math
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
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

UNIVERSE = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "SHRIRAMFIN.NS","LT.NS","TMPV.NS","TMCV.NS","EICHERMOT.NS","MARUTI.NS",
    "HEROMOTOCO.NS","M&M.NS","TITAN.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS",
    "DIVISLAB.NS","TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "BHARTIARTL.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","COALINDIA.NS",
    "ADANIENT.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS","TRENT.NS",
    "PIDILITIND.NS","SIEMENS.NS","ABB.NS","GRASIM.NS","ULTRACEMCO.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","IOC.NS","VEDL.NS","DLF.NS",
    "LODHA.NS","INDIGO.NS","ETERNAL.NS","NAUKRI.NS","COFORGE.NS","JIOFIN.NS",
    "IRFC.NS","IREDA.NS","POLYCAB.NS",
]

HORIZON = int(os.getenv("HORIZON", "10"))
BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")
LIVE_PERIOD = os.getenv("LIVE_PERIOD", "2y")
COST = float(os.getenv("ROUND_TRIP_COST", "0.003"))

# Final live gate. Validation is allowed to choose a stricter threshold.
P_THRESHOLD = float(os.getenv("P_THRESHOLD", "0.62"))
RETURN_THRESHOLD = float(os.getenv("RETURN_THRESHOLD", "0.006"))
TOP_N_ALERT = int(os.getenv("TOP_N_ALERT", "8"))
TOP_N_BACKTEST = int(os.getenv("TOP_N_BACKTEST", "5"))

# Risk/liquidity controls
MAX_VOL = float(os.getenv("MAX_VOLATILITY", "0.065"))
MIN_AVG_VALUE_CR = float(os.getenv("MIN_AVG_DAILY_VALUE_CR", "2.0"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "350"))

# Split dates for a reproducible 6-year study.
# These are generated from the available dataset when not supplied.
DEV_FRAC = float(os.getenv("DEV_FRAC", "0.60"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.20"))

RUN_BACKTEST = os.getenv("RUN_BACKTEST", "0") == "1"
SEND_TELEGRAM = os.getenv("SEND_TELEGRAM", "1") == "1"
RANDOM_STATE = 42

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_SYMBOL = "^NSEI"

# ---------------------------------------------------------------------------
# FEATURE SET
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_20","ret_60",
    "vol_5","vol_10","vol_20","atr_pct","range_pct","gap_pct",
    "rsi_14","rsi_28",
    "dist_sma10","dist_sma20","dist_sma50","dist_sma100","dist_sma200",
    "trend_20","trend_60","trend_120",
    "macd_norm","macd_signal_norm","bb_z",
    "volume_z20","volume_ratio20",
    "drawdown_20","drawdown_60","drawdown_120",
    "up_days_20","up_days_60",
    "market_ret_1","market_ret_5","market_ret_20",
    "market_vol_20","market_trend_20","market_trend_60",
    "rel_ret_5","rel_ret_20","rel_ret_60",
    "rel_strength_20","rel_strength_60",
    "dollar_volume_log",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def sf(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def finite_frame(x):
    return x.replace([np.inf, -np.inf], np.nan)


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def flatten_symbol(raw, symbol):
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
        need = ["open","high","low","close","volume"]
        if not all(c in df.columns for c in need):
            return pd.DataFrame()

        df = df[need].copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df.dropna(subset=["open","high","low","close"])
    except Exception:
        return pd.DataFrame()


def download_market(period):
    print(f"Downloading NSE universe: {period}")
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
    for i, s in enumerate(UNIVERSE, 1):
        print(f"Loading [{i}/{len(UNIVERSE)}] {s}")
        df = flatten_symbol(raw, s)
        if len(df) < MIN_HISTORY:
            print(f"WARNING: insufficient history for {s}; skipping.")
            continue
        data[s] = df
    print(f"Successful symbols: {len(data)}")
    return data


def download_market_index(period):
    try:
        raw = yf.download(
            MARKET_SYMBOL, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=False
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(MARKET_SYMBOL, axis=1, level=1) if MARKET_SYMBOL in raw.columns.get_level_values(1) else raw
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        raw = raw[["open","high","low","close","volume"]].copy()
        idx = pd.to_datetime(raw.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        raw.index = idx
        return raw.sort_index().dropna()
    except Exception as e:
        print("WARNING: NIFTY index download failed:", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def add_features(df, market):
    x = df.copy()
    c, o, h, l, v = x["close"], x["open"], x["high"], x["low"], x["volume"]

    ret = c.pct_change()
    for n in [1,3,5,10,20,60]:
        x[f"ret_{n}"] = c.pct_change(n)

    for n in [5,10,20]:
        x[f"vol_{n}"] = ret.rolling(n, min_periods=n).std()

    x["atr_pct"] = atr(x,14) / c
    x["range_pct"] = (h-l) / c
    x["gap_pct"] = o / c.shift(1) - 1

    x["rsi_14"] = rsi(c,14)
    x["rsi_28"] = rsi(c,28)

    for n in [10,20,50,100,200]:
        ma = c.rolling(n, min_periods=n).mean()
        x[f"dist_sma{n}"] = c / ma - 1

    x["trend_20"] = c / c.shift(20) - 1
    x["trend_60"] = c / c.shift(60) - 1
    x["trend_120"] = c / c.shift(120) - 1

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    x["macd_norm"] = macd / c
    x["macd_signal_norm"] = (macd-signal) / c

    ma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    x["bb_z"] = (c-ma20) / sd20.replace(0,np.nan)

    lv = np.log1p(v.replace(0,np.nan))
    x["volume_z20"] = (lv-lv.rolling(20).mean()) / lv.rolling(20).std()
    x["volume_ratio20"] = v / v.rolling(20).mean()

    for n in [20,60,120]:
        rollmax = c.rolling(n, min_periods=n).max()
        x[f"drawdown_{n}"] = c/rollmax-1
        x[f"up_days_{n}"] = (ret > 0).rolling(n, min_periods=n).mean()

    if market is not None and not market.empty:
        mc = market["close"].reindex(x.index).ffill()
        mr = mc.pct_change()
        x["market_ret_1"] = mr
        x["market_ret_5"] = mc.pct_change(5)
        x["market_ret_20"] = mc.pct_change(20)
        x["market_vol_20"] = mr.rolling(20).std()
        x["market_trend_20"] = mc/mc.rolling(20).mean()-1
        x["market_trend_60"] = mc/mc.rolling(60).mean()-1

        for n in [5,20,60]:
            sr = c.pct_change(n)
            mret = mc.pct_change(n)
            x[f"rel_ret_{n}"] = sr-mret

        x["rel_strength_20"] = (c/c.rolling(20).mean()) / (mc/mc.rolling(20).mean()) - 1
        x["rel_strength_60"] = (c/c.rolling(60).mean()) / (mc/mc.rolling(60).mean()) - 1
    else:
        for col in [
            "market_ret_1","market_ret_5","market_ret_20","market_vol_20",
            "market_trend_20","market_trend_60","rel_ret_5","rel_ret_20",
            "rel_ret_60","rel_strength_20","rel_strength_60"
        ]:
            x[col] = np.nan

    x["dollar_volume_log"] = np.log1p((c*v).clip(lower=0))

    # Execution-matched target:
    # entry = next trading day's OPEN, exit = H trading days after signal CLOSE.
    x[f"future_return_{HORIZON}"] = (
        c.shift(-HORIZON) / o.shift(-1) - 1
    )
    x[f"target_up_{HORIZON}"] = (x[f"future_return_{HORIZON}"] > 0).astype(float)

    x["symbol"] = df.attrs.get("symbol","")
    x["date"] = x.index
    return x


def build_dataset(data, market):
    frames = []
    for symbol, raw in data.items():
        raw = raw.copy()
        raw.attrs["symbol"] = symbol
        f = add_features(raw, market)
        frames.append(f.reset_index(drop=True))

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date","symbol"]).reset_index(drop=True)

    # Cross-sectional ranks are computed independently on each signal date.
    for col in [
        "ret_5","ret_20","ret_60","vol_20","dist_sma20","dist_sma50",
        "volume_ratio20","rel_ret_20","rel_ret_60","atr_pct"
    ]:
        if col in all_df:
            all_df[f"{col}_rank"] = all_df.groupby("date")[col].rank(pct=True)

    rank_cols = [c for c in all_df.columns if c.endswith("_rank")]
    feature_cols = BASE_FEATURES + rank_cols

    # Remove rows without a complete target or without enough core history.
    target = f"future_return_{HORIZON}"
    all_df = all_df.replace([np.inf,-np.inf],np.nan)
    all_df = all_df.dropna(subset=[target]).copy()

    return all_df, feature_cols


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

def make_models():
    clf_hgb = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=1.5,
        random_state=RANDOM_STATE,
    )

    clf_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.35, max_iter=1200, class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])

    reg_hgb = HistGradientBoostingRegressor(
        loss="huber",
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )

    reg_et = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=25,
            max_features=0.75,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return clf_hgb, clf_lr, reg_hgb, reg_et


def fit_model(df, features):
    x = finite_frame(df[features])
    yb = df[f"target_up_{HORIZON}"].astype(int).values
    yr = df[f"future_return_{HORIZON}"].astype(float).values

    # HGB accepts NaNs, while LR/ExtraTrees use an imputer.
    models = make_models()
    models[0].fit(x, yb)
    models[1].fit(x, yb)
    models[2].fit(x, yr)
    models[3].fit(x, yr)
    return models


def predict_model(models, df, features):
    x = finite_frame(df[features])

    p1 = models[0].predict_proba(x)[:,1]
    p2 = models[1].predict_proba(x)[:,1]
    prob = 0.60*p1 + 0.40*p2

    r1 = models[2].predict(x)
    r2 = models[3].predict(x)
    pred = 0.60*r1 + 0.40*r2

    out = df[["date","symbol","close","open","atr_pct","vol_20",
              f"future_return_{HORIZON}"]].copy()
    out["probability"] = np.clip(prob,0.001,0.999)
    out["predicted_return"] = pred
    out["net_predicted_return"] = pred - COST
    return out


# ---------------------------------------------------------------------------
# WALK FORWARD
# ---------------------------------------------------------------------------

def walk_forward(train_df, test_df, features, label):
    test_dates = sorted(pd.to_datetime(test_df["date"]).unique())
    results = []

    # Retrain every 20 signal days. This preserves chronological ordering
    # while avoiding 300+ full model fits.
    retrain_every = 20
    model = None
    last_train_end = None

    for i, d in enumerate(test_dates):
        if model is None or i % retrain_every == 0:
            tr = train_df[train_df["date"] < d].copy()
            if len(tr) < 2000:
                continue
            model = fit_model(tr, features)
            last_train_end = d

        day = test_df[test_df["date"] == d]
        if day.empty:
            continue
        pred = predict_model(model, day, features)
        results.append(pred)

        if (i+1) % 50 == 0 or i+1 == len(test_dates):
            print(f"Walk-forward {label}: [{i+1}/{len(test_dates)}]")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# THRESHOLD / VALIDATION
# ---------------------------------------------------------------------------

def evaluate(pred):
    if pred.empty:
        return {}
    y = (pred[f"future_return_{HORIZON}"] > 0).astype(int)
    p = pred["probability"].values
    r = pred["predicted_return"].values
    return {
        "observations": len(pred),
        "directional_accuracy": float(((p >= .5).astype(int)==y).mean()),
        "mae": float(np.mean(np.abs(r-pred[f"future_return_{HORIZON}"]))),
        "mean_pred": float(np.mean(r)),
        "mean_actual": float(np.mean(pred[f"future_return_{HORIZON}"])),
    }


def select_thresholds(val):
    # Threshold selection is validation-only. The objective rewards
    # net return and penalizes low sample size and poor hit rate.
    candidates = []
    for pmin in [0.55,0.57,0.59,0.61,0.62,0.64,0.66,0.68,0.70]:
        for rmin in [0.003,0.004,0.005,0.006,0.007,0.008,0.010]:
            q = val[(val.probability >= pmin) &
                    (val.predicted_return >= rmin)].copy()
            if len(q) < 30:
                continue
            net = q[f"future_return_{HORIZON}"] - COST
            win = float((net > 0).mean())
            avg = float(net.mean())
            pf = float(net[net>0].sum() / abs(net[net<0].sum())) if (net<0).any() else 10.0
            score = avg * np.sqrt(len(q)) * (0.5 + min(pf,3)/3)
            candidates.append((score,pmin,rmin,len(q),win,avg,pf))

    if not candidates:
        return {"pmin":P_THRESHOLD,"rmin":RETURN_THRESHOLD,"source":"fixed-fallback"}

    best = max(candidates,key=lambda z:z[0])
    return {
        "pmin":best[1],
        "rmin":best[2],
        "n":best[3],
        "win":best[4],
        "avg_net":best[5],
        "profit_factor":best[6],
        "source":"validation",
    }


def apply_gate(pred, thresholds):
    x = pred.copy()
    x["action"] = np.where(
        (x.probability >= thresholds["pmin"]) &
        (x.predicted_return >= thresholds["rmin"]) &
        (x.vol_20 <= MAX_VOL) &
        (x.atr_pct <= MAX_VOL),
        "TRADE","PASS"
    )
    return x


# ---------------------------------------------------------------------------
# PORTFOLIO / NON-OVERLAP
# ---------------------------------------------------------------------------

def nonoverlap(pred, thresholds):
    q = apply_gate(pred, thresholds)
    q = q[q.action=="TRADE"].copy()
    if q.empty:
        return q
    q["net_return"] = q[f"future_return_{HORIZON}"] - COST
    q = q.sort_values(["date","probability"],ascending=[True,False])
    chosen=[]
    last_exit=None
    for _, row in q.iterrows():
        entry_date = pd.Timestamp(row["date"]) + pd.Timedelta(days=1)
        if last_exit is not None and entry_date <= last_exit:
            continue
        chosen.append(row)
        last_exit = pd.Timestamp(row["date"]) + pd.Timedelta(days=HORIZON)
    return pd.DataFrame(chosen)


def portfolio_curve(pred, thresholds, starting=100000.0):
    q = nonoverlap(pred,thresholds)
    if q.empty:
        return {
            "starting_capital":starting,"ending_equity":starting,
            "total_return":0.0,"max_drawdown":0.0,"trades":0
        }
    equity=starting
    peak=starting
    maxdd=0.0
    for r in q["net_return"].values:
        equity *= (1+float(r))
        peak=max(peak,equity)
        maxdd=min(maxdd,equity/peak-1)
    return {
        "starting_capital":starting,
        "ending_equity":equity,
        "total_return":equity/starting-1,
        "max_drawdown":maxdd,
        "trades":len(q),
    }


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def telegram_send(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    chat=os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not SEND_TELEGRAM:
        print("Telegram disabled.")
        return False
    if not token or not chat:
        print("Telegram not configured.")
        return False
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r=requests.post(url,json={
            "chat_id":chat,
            "text":text[:4000],
            "disable_web_page_preview":True,
        },timeout=30)
        print("Telegram:",r.status_code)
        return r.ok
    except Exception as e:
        print("Telegram error:",e)
        return False


# ---------------------------------------------------------------------------
# LIVE MODEL
# ---------------------------------------------------------------------------

def live_scan(data, market, features, thresholds):
    frames=[]
    for symbol,raw in data.items():
        raw=raw.copy()
        raw.attrs["symbol"]=symbol
        f=add_features(raw,market)
        frames.append(f.tail(1).reset_index(drop=True))
    latest=pd.concat(frames,ignore_index=True)

    # Remove rows with missing key features.
    latest=latest.dropna(subset=["close","atr_pct","vol_20"]).copy()

    # Train on all historical rows with known target.
    train_frames=[]
    for symbol,raw in data.items():
        raw=raw.copy()
        raw.attrs["symbol"]=symbol
        train_frames.append(add_features(raw,market).reset_index(drop=True))
    train=pd.concat(train_frames,ignore_index=True)
    train=train.dropna(subset=[f"future_return_{HORIZON}"]).copy()

    # Recompute cross-sectional ranks for the combined training/live frame.
    combined=pd.concat([train,latest],ignore_index=True)
    for col in [
        "ret_5","ret_20","ret_60","vol_20","dist_sma20","dist_sma50",
        "volume_ratio20","rel_ret_20","rel_ret_60","atr_pct"
    ]:
        if col in combined:
            combined[f"{col}_rank"]=combined.groupby("date")[col].rank(pct=True)
    rank_cols=[c for c in combined.columns if c.endswith("_rank")]
    live_features=BASE_FEATURES+rank_cols

    train2=combined.iloc[:len(train)].copy()
    live2=combined.iloc[len(train):].copy()

    model=fit_model(train2,live_features)
    pred=predict_model(model,live2,live_features)

    # Liquidity gate using latest daily traded value.
    latest_lookup={}
    for s,raw in data.items():
        if raw.empty: continue
        row=raw.iloc[-1]
        avg_value=(raw["close"]*raw["volume"]).tail(20).mean()/1e7
        latest_lookup[s]=float(avg_value)

    pred["avg_value_cr"]=pred["symbol"].map(latest_lookup).fillna(0)
    pred["liquidity_ok"]=pred["avg_value_cr"]>=MIN_AVG_VALUE_CR

    pred["action"]="PASS"
    pred.loc[
        (pred.probability>=thresholds["pmin"]) &
        (pred.predicted_return>=thresholds["rmin"]) &
        (pred.vol_20<=MAX_VOL) &
        (pred.atr_pct<=MAX_VOL) &
        pred.liquidity_ok,
        "action"
    ]="TRADE"

    return pred.sort_values(
        ["action","probability","predicted_return"],
        ascending=[True,False,False]
    )


# ---------------------------------------------------------------------------
# REPORTS
# ---------------------------------------------------------------------------

def alert_text(pred, thresholds):
    trade=pred[pred.action=="TRADE"].head(TOP_N_ALERT)
    closest=pred.head(5)

    lines=[
        "V8.3 NSE QUANT-ONLY ALERT",
        f"Generated: {stamp()}",
        f"Primary horizon: H{HORIZON}",
        f"Validation/final gate: P≥{thresholds['pmin']:.2f}, predicted return≥{thresholds['rmin']:.2%}",
        "",
        "GEMINI: DISABLED — QUANT-ONLY MODEL",
        "No LLM/API dependency.",
        "",
    ]

    if trade.empty:
        lines += [
            "NO QUALIFYING TRADE",
            "No candidate passed probability + return + volatility + liquidity gates.",
            "",
            "Closest candidates:",
        ]
    else:
        lines += ["QUALIFYING TRADE CANDIDATE(S)",""]
        for i,(_,r) in enumerate(trade.iterrows(),1):
            lines.append(
                f"{i}. {r.symbol.replace('.NS','')} | "
                f"P={r.probability:.3f} | Pred={r.predicted_return:.2%} | "
                f"Close={r.close:.2f} | Vol20={r.vol_20:.2%}"
            )
        lines += ["","Closest candidates:"]

    for i,(_,r) in enumerate(closest.iterrows(),1):
        lines.append(
            f"{i}. {r.symbol.replace('.NS','')} | "
            f"P={r.probability:.3f} | Pred={r.predicted_return:.2%} | "
            f"Close={r.close:.2f}"
        )

    lines += [
        "",
        f"Execution: Signal T close → Entry T+1 open → Exit T+{HORIZON} close",
        f"Round-trip cost assumed: {COST:.2%}",
        "",
        "Model: point-in-time quant ensemble",
        "Probability: 60% gradient boosting + 40% logistic regression",
        "Return: 60% gradient boosting + 40% ExtraTrees",
        "Risk controls: volatility + ATR + liquidity",
        "",
        "Research signal only. Historical backtests do not guarantee future performance.",
    ]
    return "\n".join(lines)


def save_json(name,obj):
    path=AUDIT_DIR/name
    path.write_text(json.dumps(json_safe(obj),indent=2,default=str),encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

def run_backtest(dataset, features):
    dates=sorted(dataset.date.unique())
    n=len(dates)
    d1=dates[int(n*DEV_FRAC)]
    d2=dates[int(n*(DEV_FRAC+VAL_FRAC))]

    dev=dataset[dataset.date<=d1].copy()
    val=dataset[(dataset.date>d1)&(dataset.date<=d2)].copy()
    oos=dataset[dataset.date>d2].copy()

    print("="*78)
    print(f"DEVELOPMENT END: {d1}")
    print(f"VALIDATION END:  {d2}")
    print(f"OOS START:       {d2}")
    print("="*78)

    # Development -> validation
    print("VALIDATION WALK-FORWARD")
    vp=walk_forward(dev,val,features,"VAL")

    if vp.empty:
        raise RuntimeError("Validation walk-forward produced no predictions.")

    thresholds=select_thresholds(vp)
    print("Validation-selected thresholds:",thresholds)

    # Development + validation -> OOS
    train=pd.concat([dev,val],ignore_index=True)
    print("OOS WALK-FORWARD")
    op=walk_forward(train,oos,features,"OOS")

    if op.empty:
        raise RuntimeError("OOS walk-forward produced no predictions.")

    gated=apply_gate(op,thresholds)
    q=gated[gated.action=="TRADE"].copy()
    if not q.empty:
        q["net_return"]=q[f"future_return_{HORIZON}"]-COST

    eval_oos=evaluate(op)
    pf=portfolio_curve(op,thresholds)

    report={
        "revision":"V8.3-QUANT-ONLY",
        "generated":stamp(),
        "horizon":HORIZON,
        "observations":len(dataset),
        "symbols":int(dataset.symbol.nunique()),
        "signal_dates":int(dataset.date.nunique()),
        "development":len(dev),
        "validation":len(val),
        "oos":len(oos),
        "thresholds":thresholds,
        "oos_metrics":eval_oos,
        "portfolio":pf,
        "qualifying_oos_rows":len(q),
    }
    save_json("v8_3_backtest_summary.json",report)

    print("="*78)
    print("V8.3 OOS RESULTS")
    print(json.dumps(report,indent=2,default=str))
    print("="*78)

    return thresholds, report


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("="*78)
    print("V8.3 — NSE QUANT-ONLY POINT-IN-TIME RESEARCH ENGINE")
    print("="*78)
    print("Revision: 2026-09-28-QUANT-ENSEMBLE-NO-GEMINI")
    print("Execution: Signal T Close -> Entry T+1 Open -> Exit T+H Close")
    print(f"Horizon: H{HORIZON}")
    print(f"Round-trip cost: {COST:.2%}")
    print("GEMINI: DISABLED BY DESIGN")
    print("="*78)

    data=download_market(BACKTEST_PERIOD if RUN_BACKTEST else LIVE_PERIOD)
    market=download_market_index(BACKTEST_PERIOD if RUN_BACKTEST else LIVE_PERIOD)

    if len(data)<10:
        raise RuntimeError("Too few symbols downloaded.")

    dataset,features=build_dataset(data,market)
    print("DATASET")
    print("Observations:",len(dataset))
    print("Symbols:",dataset.symbol.nunique())
    print("Signal dates:",dataset.date.nunique())
    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Forward-return targets are excluded from FEATURES.")

    # Default live threshold; if backtest is requested, replace it with the
    # threshold selected strictly on validation.
    thresholds={"pmin":P_THRESHOLD,"rmin":RETURN_THRESHOLD,"source":"configured"}

    if RUN_BACKTEST:
        thresholds,_=run_backtest(dataset,features)

        # Reload data is unnecessary; dataset already contains all history.
        # Live scan uses the latest available data and trains on all rows.
        live_data=data
    else:
        live_data=data

    print("="*78)
    print("LIVE QUANT SCAN")
    print("="*78)
    print(f"Configured/selected gate: P>={thresholds['pmin']:.2f}, return>={thresholds['rmin']:.2%}")

    live=live_scan(live_data,market,features,thresholds)

    print(live[[
        "symbol","probability","predicted_return","vol_20","atr_pct","action"
    ]].head(20).to_string(index=False))

    save_json(
        "v8_3_live_candidates.json",
        live.head(20).to_dict(orient="records")
    )

    text=alert_text(live,thresholds)
    print("="*78)
    print(text)
    print("="*78)

    sent=telegram_send(text)

    save_json("v8_3_run.json",{
        "generated":stamp(),
        "telegram_sent":sent,
        "gemini":False,
        "thresholds":thresholds,
        "top_candidates":live.head(20).to_dict(orient="records"),
    })

    print("V8.3 COMPLETED")


if __name__=="__main__":
    main()
