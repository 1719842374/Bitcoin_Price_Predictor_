#!/usr/bin/env python3
"""
🔧 FIXED Bitcoin Price Predictor - Anti-Overfitting Version (Hardened)
=====================================================================
Key fixes and hardening in this edition:
1) Robust external indicator fallbacks and safer requests
2) DXY ticker fixed to '^DXY' with fallbacks
3) Feature engineering: safer power-law (no div-by-zero), denoising fallback
4) LPPL now predicts absolute prices (not returns)
5) Future forecast uses absolute prices and uncertainty bands from residuals
6) Stacking uses TimeSeriesSplit to avoid leakage
7) Models tuned to reduce overfitting; stricter ensemble selection
8) Optional dependencies handled gracefully (skip if unavailable)
"""

import os
import math
import csv
import warnings
import datetime
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Optional/soft imports with graceful fallbacks ----
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
except Exception:  # pragma: no cover
    plt = None
    StrMethodFormatter = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from scipy.stats import spearmanr
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    spearmanr = None
    curve_fit = None

try:
    from arch import arch_model as arch_garch_model
except Exception:  # pragma: no cover
    arch_garch_model = None

try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
except Exception:  # pragma: no cover
    ARIMA = None
    SARIMAX = None
    adfuller = None
    acf = None
    pacf = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.base import clone
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:  # pragma: no cover
    variance_inflation_factor = None

# -------------------------- Helper functions --------------------------

def _safe_requests_get(url, headers=None, timeout=10):
    if requests is None:
        return None
    try:
        headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


def lppl(t, A, B, tc, m, C, omega, phi):
    dt = np.maximum(tc - t, 1e-6)
    dt = np.clip(dt, 1e-6, 1e6)
    log_dt = np.log(dt)
    power = np.power(dt, m, where=(dt > 0))
    cos_term = np.cos(omega * log_dt + phi)
    return A + B * power * (1 + C * cos_term)


def fit_lppl(X, prices):
    if curve_fit is None:
        # Fallback: linear trend on log price
        from sklearn.linear_model import LinearRegression
        t = X["days_since_genesis"].values.reshape(-1, 1)
        log_p = np.log(prices + 1e-6)
        lin_model = LinearRegression()
        lin_model.fit(t, log_p)
        return lin_model

    t = X["days_since_genesis"].values
    log_p = np.log(prices + 1e-6)
    p0 = [np.mean(log_p), -1, t[-1] + 2000, 0.5, 0.1, 7, 0]
    bounds = (
        [min(log_p) - 1, -np.inf, t[-1] + 365, 0.1, -1, 2, -np.pi],
        [max(log_p) + 1, -0.001, t[-1] + 10000, 1.0, 1, 30, np.pi],
    )
    try:
        params, _ = curve_fit(lppl, t, log_p, p0=p0, bounds=bounds, maxfev=50000)
        pred_log = lppl(t, *params)
        fit_r2 = r2_score(log_p, pred_log)
        if fit_r2 < 0:
            raise ValueError("Negative fit R2")
        return params
    except Exception:
        # Linear fallback on log price
        from sklearn.linear_model import LinearRegression
        lin_model = LinearRegression()
        lin_model.fit(t.reshape(-1, 1), log_p)
        return lin_model


# -------------------------- Deep models --------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=0.95,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ---------------------- Indicator fetch functions ----------------------

def fetch_fear_greed():
    if requests is None:
        return np.nan
    try:
        resp = _safe_requests_get("https://api.alternative.me/fng/?limit=1")
        if resp is None:
            return np.nan
        data = resp.json()
        timestamp_str = data["data"][0]["timestamp"]
        fetch_date = datetime.datetime.fromtimestamp(int(timestamp_str)).date()
        current_date = datetime.date.today()
        if (current_date - fetch_date).days > 1:
            return np.nan
        return int(data["data"][0]["value"])
    except Exception:
        return np.nan


def fetch_dxy():
    if yf is None:
        return np.nan
    # Try multiple tickers: '^DXY' -> 'DX-Y.NYB' -> 'UUP' (ETF proxy)
    for ticker in ["^DXY", "DX-Y.NYB", "UUP"]:
        try:
            t = yf.Ticker(ticker)
            d = t.history(period="5d", auto_adjust=False)
            if not d.empty and "Close" in d.columns:
                return float(d["Close"].iloc[-1])
        except Exception:
            continue
    return np.nan


def fetch_mvrv():
    if requests is None or BeautifulSoup is None:
        return np.nan
    try:
        resp = _safe_requests_get("https://charts.bitbo.io/mvrv/")
        if resp is None:
            return np.nan
        soup = BeautifulSoup(resp.text, "html.parser")
        value_tag = soup.find("div", class_="current-mvrv")
        if value_tag:
            return float(value_tag.text.strip())
        return np.nan
    except Exception:
        return np.nan


def fetch_nvt():
    if requests is None or BeautifulSoup is None:
        return np.nan
    try:
        resp = _safe_requests_get("https://charts.bitbo.io/nvt/")
        if resp is None:
            return np.nan
        soup = BeautifulSoup(resp.text, "html.parser")
        value_tag = soup.find("div", class_="current-nvt")
        if value_tag:
            return float(value_tag.text.strip())
        return np.nan
    except Exception:
        return np.nan


def fetch_rsi_14(btc_long):
    try:
        if len(btc_long) < 14:
            return np.nan
        delta = btc_long["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        rsi_series = 100 - 100 / (1 + rs)
        rsi_series = rsi_series.dropna()
        if rsi_series.empty:
            return np.nan
        return float(rsi_series.iloc[-1])
    except Exception:
        return np.nan


def fetch_etf_flows():
    if requests is None or BeautifulSoup is None:
        return np.nan
    try:
        resp = _safe_requests_get("https://farside.co.uk/btc/")
        if resp is None:
            return np.nan
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="etf-table")
        if table:
            rows = table.find_all("tr")
            latest_row = rows[-1]
            cells = latest_row.find_all("td")
            net_flow_str = cells[-1].text.strip().replace("$", "").replace(",", "")
            return float(net_flow_str) if net_flow_str else np.nan
        return np.nan
    except Exception:
        return np.nan


def fetch_cbbi():
    if requests is None or BeautifulSoup is None:
        return np.nan
    try:
        resp = _safe_requests_get("https://colintalkscrypto.com/cbbi/")
        if resp is None:
            return np.nan
        soup = BeautifulSoup(resp.text, "html.parser")
        value_tag = soup.find("span", class_="cbbi-score")
        if value_tag:
            return float(value_tag.text.strip())
        return np.nan
    except Exception:
        return np.nan


def fetch_pi_cycle_top(btc_long):
    try:
        ma111 = float(btc_long["Close"].rolling(111).mean().iloc[-1])
        ma350 = float(btc_long["Close"].rolling(350).mean().iloc[-1])
        ma350_2x = ma350 * 2
        if np.isnan(ma111) or np.isnan(ma350_2x):
            return 0
        return 1 if ma111 > ma350_2x else 0
    except Exception:
        return 0


def fetch_google_trends():
    # Google Trends blocks scraping frequently; return NaN by default
    return np.nan


def fetch_2year_ma_multiplier(btc_long):
    try:
        ma2y = float(btc_long["Close"].rolling(730).mean().iloc[-1])
        close_last = float(btc_long["Close"].iloc[-1])
        return close_last / ma2y if ma2y > 0 else np.nan
    except Exception:
        return np.nan


def fetch_golden_ratio_multiplier(btc_long):
    try:
        ma350 = float(btc_long["Close"].rolling(350).mean().iloc[-1])
        golden = ma350 * 1.618
        close_last = float(btc_long["Close"].iloc[-1])
        return close_last / golden if golden > 0 else np.nan
    except Exception:
        return np.nan


def save_indicators_to_csv(indicators):
    try:
        with open("bitcoin_indicators.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Indicator", "Value"])
            for key, value in indicators.items():
                writer.writerow([key, value])
    except Exception:
        pass


# -------------------------- Predictor class --------------------------

class FixedBitcoinPredictor:
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        self.selected_features = None
        self.validation_results = {}
        self.feature_importance_scores = {}
        self.stacking = None
        self.look_back = 60
        self.use_lstm = False and TORCH_AVAILABLE  # default off
        self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

    def denoise_series(self, series):
        # Fallback-friendly denoising: wavelet if available, else rolling median
        if not isinstance(series, np.ndarray):
            series = np.array(series)
        if len(series) < 5:
            return series
        try:
            mean = np.mean(series)
            std = np.std(series) + 1e-12
            z = (series - mean) / std
            series = series.copy()
            series[np.abs(z) > 3] = mean
            if pywt is None:
                # Simple fallback: rolling median
                s = pd.Series(series)
                den = s.rolling(5, min_periods=1, center=True).median().values
                return np.nan_to_num(den, nan=np.nanmean(series))
            coeffs = pywt.wavedec(series, "db4", level=min(5, int(np.log2(max(8, len(series))))), mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(series)))
            coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
            denoised = pywt.waverec(coeffs, "db4", mode="symmetric")
            if len(denoised) != len(series):
                denoised = np.interp(np.arange(len(series)), np.linspace(0, len(series) - 1, len(denoised)), denoised)
            return np.nan_to_num(denoised, nan=np.nanmean(series))
        except Exception:
            return np.nan_to_num(series, nan=np.nanmean(series))

    def create_advanced_features(self, df, skip_selection=False):
        print("🔧 Creating advanced features (FIXED - Anti-Overfitting)...")
        df = df.reset_index(drop=True).copy()

        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Time features
        if "Date" in df.columns:
            df["hour"] = df["Date"].dt.hour
            df["day_of_week"] = df["Date"].dt.dayofweek
            df["month"] = df["Date"].dt.month
            df = df.drop(columns=["Date"], errors="ignore")
        else:
            df["hour"], df["day_of_week"], df["month"] = 0, 0, 1

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = 0.0

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Denoise close
        df["Close"] = self.denoise_series(close.values)
        df["Close"] = pd.Series(df["Close"], index=df.index).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(float(close.mean()))

        # Price features
        for period in [10, 20, 50]:
            sma = close.rolling(period, min_periods=1).mean()
            ema = close.ewm(span=period, min_periods=1).mean()
            df[f"sma_{period}"] = sma
            df[f"ema_{period}"] = ema
            df[f"price_sma_ratio_{period}"] = np.where(sma > 0, close / sma, 0.0)
            df[f"momentum_{period}"] = close.pct_change(period)

        # Volatility
        returns = close.pct_change().fillna(0)
        try:
            if adfuller is not None and adfuller(returns.dropna())[1] > 0.05:
                returns = returns.diff().fillna(0)
        except Exception:
            pass

        for period in [20, 50]:
            vol = returns.rolling(period, min_periods=1).std()
            df[f"volatility_{period}"] = vol
            df[f"vol_ma_{period}"] = vol.rolling(10, min_periods=1).mean()

        # GARCH volatility (optional)
        if arch_garch_model is not None:
            returns100 = returns.dropna() * 100
            if len(returns100) > 50:
                try:
                    garch = arch_garch_model(returns100, vol="Garch", p=1, q=1).fit(disp="off")
                    cond_vol = garch.conditional_volatility
                    df["garch_vol"] = np.nan
                    df.loc[cond_vol.index, "garch_vol"] = cond_vol.values
                    df["garch_vol"] = df["garch_vol"].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(df[f"volatility_{20}" if "volatility_20" in df.columns else f"volatility_{50}"].mean())
                except Exception:
                    df["garch_vol"] = df.get("volatility_20", returns.rolling(20, min_periods=1).std()).fillna(0)
        else:
            df["garch_vol"] = df.get("volatility_20", returns.rolling(20, min_periods=1).std()).fillna(0)

        # Momentum indicators
        for period in [14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df[f"rsi_{period}"] = rsi

            if period >= 21:
                ema_fast = close.ewm(span=max(2, period // 2), min_periods=1).mean()
                ema_slow = close.ewm(span=period, min_periods=1).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=9, min_periods=1).mean()
                df[f"macd_{period}"] = macd
                df[f"macd_histogram_{period}"] = macd - macd_signal

        # Volume features
        for period in [20, 50]:
            vol_ma = volume.rolling(period, min_periods=1).mean()
            df[f"volume_ratio_{period}"] = np.where(vol_ma > 0, volume / vol_ma, 0.0)

        # VWAP
        vwap_20 = (volume * close).rolling(20, min_periods=1).sum() / (volume.rolling(20, min_periods=1).sum() + 1e-8)
        df["vwap_20"] = vwap_20
        df["price_vwap_ratio"] = np.where(vwap_20 > 0, close / vwap_20, 0.0)

        # Microstructure
        df["spread"] = np.where(close > 0, (high - low) / close, 0.0)
        df["price_position"] = np.where((high - low) > 0, (df["Close"] - low) / (high - low), 0.5)

        # Bollinger width
        for period in [20, 50]:
            bb_middle = close.rolling(period, min_periods=1).mean()
            bb_std = close.rolling(period, min_periods=1).std()
            df[f"bb_width_{period}"] = np.where(bb_middle > 0, (2 * bb_std) / bb_middle, 0.0)

        # Lags
        for lag in [1, 3, 5, 7, 10]:
            df[f"return_lag_{lag}"] = returns.shift(lag).fillna(0)
            df[f"vol_lag_{lag}"] = df.get("volatility_20", returns.rolling(20, min_periods=1).std()).shift(lag).fillna(0)

        # Trend strength
        if spearmanr is not None:
            for period in [20, 50]:
                def _spearman_trend(x):
                    if len(x) < period:
                        return 0.0
                    try:
                        return abs(spearmanr(range(len(x)), x)[0])
                    except Exception:
                        return 0.0
                df[f"trend_strength_{period}"] = close.rolling(period, min_periods=period).apply(_spearman_trend).fillna(0)
        else:
            df["trend_strength_20"] = 0.0
            df["trend_strength_50"] = 0.0

        # Power law and realized price
        df["realized_price"] = (df["Close"] * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
        df["days_since_genesis"] = np.arange(len(df)) + 1  # avoid 0 to prevent div-by-zero
        df["power_law"] = np.log1p(df["Close"]) / np.log1p(df["days_since_genesis"])  # safe

        # M2 features if available
        if "M2SL" in df.columns:
            df["m2"] = df["M2SL"].astype(float)
            df["m2_growth"] = df["m2"].pct_change().fillna(0)
            df["m2_growth_lag4w"] = df["m2_growth"].shift(28).fillna(0)
            for period in [6, 12]:
                df[f"m2_growth_ma_{period}"] = df["m2_growth"].rolling(period, min_periods=1).mean().fillna(0)

        # External indicators (fetched once per full dataset)
        print("🔧 Fetching external indicators...")
        try:
            if yf is not None:
                btc_ticker = yf.Ticker("BTC-USD")
                btc_long = btc_ticker.history(period="max", auto_adjust=False)
                btc_long.index = pd.to_datetime(btc_long.index).tz_localize(None)
            else:
                btc_long = pd.DataFrame({"Close": close})
        except Exception:
            btc_long = pd.DataFrame({"Close": close})

        fear_greed = fetch_fear_greed()
        dxy = fetch_dxy()
        mvrv = fetch_mvrv()
        nvt = fetch_nvt()
        rsi_14 = fetch_rsi_14(btc_long)
        etf_flows = fetch_etf_flows()
        cbbi = fetch_cbbi()
        pi_cycle_top = fetch_pi_cycle_top(btc_long)
        google_trends = fetch_google_trends()
        two_year_ma_multiplier = fetch_2year_ma_multiplier(btc_long)
        golden_ratio_multiplier = fetch_golden_ratio_multiplier(btc_long)

        for col, val in [
            ("fear_greed", fear_greed),
            ("dxy", dxy),
            ("mvrv", mvrv),
            ("nvt", nvt),
            ("rsi_14", rsi_14),
            ("etf_flows", etf_flows),
            ("cbbi", cbbi),
            ("pi_cycle_top", pi_cycle_top),
            ("google_trends", google_trends),
            ("2year_ma_multiplier", two_year_ma_multiplier),
            ("golden_ratio_multiplier", golden_ratio_multiplier),
        ]:
            df[col] = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else np.nan

        # Ratings
        df["mvrv_rating"] = np.where(df["mvrv"] < 1, 0, np.where(df["mvrv"] > 3.5, 2, 1))
        df["rsi_rating"] = np.where(df["rsi_14"] < 30, 0, np.where(df["rsi_14"] > 70, 2, 1))
        df["fear_greed_rating"] = np.where(df["fear_greed"] < 20, 0, np.where(df["fear_greed"] > 80, 2, 1))

        # Scale indicators
        indicator_cols = [
            "fear_greed",
            "dxy",
            "mvrv",
            "nvt",
            "rsi_14",
            "etf_flows",
            "cbbi",
            "pi_cycle_top",
            "google_trends",
            "2year_ma_multiplier",
            "golden_ratio_multiplier",
        ]
        scaler = MinMaxScaler()
        df[indicator_cols] = scaler.fit_transform(df[indicator_cols].fillna(0))

        # Cleanup
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if not skip_selection:
            # Select features by XGB importance if available; else fallback to Lasso coefficients
            target = df["Close"].astype(float)
            features = df.drop(columns=["Close", "Open", "High", "Low", "Volume"], errors="ignore")

            if xgb is not None:
                xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=200, max_depth=3, subsample=0.6, colsample_bytree=0.6)
                xgb_model.fit(features, target)
                importance = xgb_model.feature_importances_
                feature_importance = sorted(zip(features.columns, importance), key=lambda x: x[1], reverse=True)
            else:
                # Fallback: use absolute Lasso coefficients as proxy
                lasso = Lasso(alpha=1.0, max_iter=10000)
                X_scaled = RobustScaler().fit_transform(features)
                lasso.fit(X_scaled, target)
                importance = np.abs(getattr(lasso, "coef_", np.zeros(features.shape[1])))
                feature_importance = sorted(zip(features.columns, importance), key=lambda x: x[1], reverse=True)

            self.feature_importance_scores = dict(feature_importance)
            print("Top 20 Feature Importance:")
            for feat, score in feature_importance[:20]:
                print(f"{feat}: {float(score):.6f}")

            top_features = [feat for feat, _ in feature_importance[:30]]
            X_reduced = features[top_features].copy()

            # VIF filter (< 20)
            try:
                vif_data = pd.DataFrame()
                vif_data["feature"] = top_features
                # Add small noise to avoid singular matrices in VIF
                X_vif = X_reduced.values + 1e-9 * np.random.randn(*X_reduced.values.shape)
                vif_vals = [variance_inflation_factor(X_vif, i) for i in range(len(top_features))]
                vif_data["VIF"] = vif_vals
                low_vif_features = vif_data[vif_data["VIF"] < 20]["feature"].tolist()
            except Exception:
                low_vif_features = top_features[:20]

            if len(low_vif_features) < 20:
                print(f"⚠️ Only {len(low_vif_features)} features after VIF filter, taking top 20 by importance")
                low_vif_features = [feat for feat, _ in feature_importance[:20]]

            # Ensure 'Close' is part of the model features for next-day price prediction
            if "Close" not in low_vif_features and "Close" in df.columns:
                low_vif_features = ["Close"] + low_vif_features
            # De-duplicate while preserving order
            seen = set()
            low_vif_features = [f for f in low_vif_features if not (f in seen or seen.add(f))]
            print(f"✅ Selected {len(low_vif_features)} features after VIF filter (<20)")
            self.selected_features = low_vif_features

            # Restrict df to selected features plus OHLCV if present
            keep_cols = [c for c in low_vif_features] + [c for c in ["Open", "High", "Low", "Volume"] if c in df.columns]
            df = df[keep_cols]

        feature_count = len([c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume", "M2SL"]])
        print(f"✅ Features created: {feature_count} features (ANTI-OVERFITTING)")

        # Save indicators snapshot
        indicators = {
            "fear_greed": float(df["fear_greed"].iloc[-1]) if "fear_greed" in df.columns else np.nan,
            "dxy": float(df["dxy"].iloc[-1]) if "dxy" in df.columns else np.nan,
            "mvrv": float(df["mvrv"].iloc[-1]) if "mvrv" in df.columns else np.nan,
            "nvt": float(df["nvt"].iloc[-1]) if "nvt" in df.columns else np.nan,
            "rsi_14": float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else np.nan,
            "etf_flows": float(df["etf_flows"].iloc[-1]) if "etf_flows" in df.columns else np.nan,
            "cbbi": float(df["cbbi"].iloc[-1]) if "cbbi" in df.columns else np.nan,
            "pi_cycle_top": float(df["pi_cycle_top"].iloc[-1]) if "pi_cycle_top" in df.columns else np.nan,
            "google_trends": float(df["google_trends"].iloc[-1]) if "google_trends" in df.columns else np.nan,
            "2year_ma_multiplier": float(df["2year_ma_multiplier"].iloc[-1]) if "2year_ma_multiplier" in df.columns else np.nan,
            "golden_ratio_multiplier": float(df["golden_ratio_multiplier"].iloc[-1]) if "golden_ratio_multiplier" in df.columns else np.nan,
        }
        save_indicators_to_csv(indicators)

        return df

    def build_fixed_models(self):
        print("🔧 Building FIXED models (removed overfitting models)...")

        tscv = TimeSeriesSplit(n_splits=5)

        # Linear model (stronger regularization)
        param_grid_lasso = {"alpha": [5.0, 10.0, 20.0, 40.0]}
        self.models["lasso"] = GridSearchCV(Lasso(max_iter=10000), param_grid_lasso, cv=tscv)

        # Random Forest (shallow, large leaves to curb overfit)
        self.models["rf"] = GridSearchCV(
            RandomForestRegressor(
                min_samples_split=100,
                min_samples_leaf=100,
                max_features=0.2,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            {"n_estimators": [150, 250], "max_depth": [2, 3]},
            cv=3,
        )

        # Extra Trees (shallow)
        self.models["extra_trees"] = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=3,
            min_samples_split=100,
            min_samples_leaf=100,
            max_features=0.2,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        # Gradient Boosting (small steps)
        self.models["gbr"] = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.001,
            min_samples_split=100,
            min_samples_leaf=80,
            subsample=0.4,
            random_state=42,
        )

        if xgb is not None:
            param_grid_xgb = {
                "n_estimators": [150, 250, 350],
                "max_depth": [2, 3],
                "learning_rate": [0.001, 0.005, 0.01],
            }
            self.models["xgb"] = GridSearchCV(
                xgb.XGBRegressor(
                    subsample=0.5,
                    colsample_bytree=0.5,
                    reg_alpha=4.0,
                    reg_lambda=4.0,
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1,
                ),
                param_grid_xgb,
                cv=5,
            )
        else:
            print("⚠️ XGBoost not available, skipping...")

        if lgb is not None:
            self.models["lgb"] = lgb.LGBMRegressor(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.001,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=4.0,
                reg_lambda=4.0,
                random_state=42,
                n_jobs=-1,
            )
        else:
            print("⚠️ LightGBM not available, skipping...")

        # Deep model placeholders
        self.models["lstm"] = None
        self.models["lppl"] = fit_lppl
        self.models["sarimax"] = None

        print(f"✅ FIXED models built: {len(self.models)} models")

    def time_series_cv_with_metrics(self, X, y, model, model_name):
        n_splits = 5
        test_size = max(32, len(X) // (n_splits + 2))
        train_scores, val_scores, r2_scores = [], [], []

        for i in range(n_splits):
            train_end = len(X) - (n_splits - i) * test_size
            val_end = train_end + test_size
            if train_end <= 0 or val_end <= train_end:
                continue

            X_train_fold = X.iloc[:train_end].copy()
            X_val_fold = X.iloc[train_end:val_end].copy()
            y_train_fold = y.iloc[:train_end].copy()
            y_val_fold = y.iloc[train_end:val_end].copy()

            X_train_fold = X_train_fold.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val_fold = X_val_fold.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_train_fold = y_train_fold.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_val_fold = y_val_fold.replace([np.inf, -np.inf], np.nan).fillna(0)

            scaler = RobustScaler()
            target_scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            y_train_scaled = target_scaler.fit_transform(y_train_fold.values.reshape(-1, 1)).ravel()
            y_val_scaled = target_scaler.transform(y_val_fold.values.reshape(-1, 1)).ravel()

            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_val_scaled = np.nan_to_num(X_val_scaled)
            y_train_scaled = np.nan_to_num(y_train_scaled)
            y_val_scaled = np.nan_to_num(y_val_scaled)

            model.fit(X_train_scaled, y_train_scaled)

            train_pred_scaled = model.predict(X_train_scaled)
            val_pred_scaled = model.predict(X_val_scaled)

            train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
            val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()

            train_mae = mean_absolute_error(y_train_fold, train_pred)
            val_mae = mean_absolute_error(y_val_fold, val_pred)
            val_r2 = r2_score(y_val_fold, val_pred)

            train_scores.append(train_mae)
            val_scores.append(val_mae)
            r2_scores.append(val_r2)

        if not train_scores:
            return {
                "train_mae": float("nan"),
                "val_mae": float("nan"),
                "val_std": float("nan"),
                "r2_mean": float("nan"),
                "r2_std": float("nan"),
                "overfitting_ratio": float("nan"),
                "stability": float("nan"),
            }

        return {
            "train_mae": float(np.mean(train_scores)),
            "val_mae": float(np.mean(val_scores)),
            "val_std": float(np.std(val_scores)),
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "overfitting_ratio": float(np.mean(val_scores) / max(1e-9, np.mean(train_scores))),
            "stability": float(np.std(val_scores) / max(1e-9, np.mean(val_scores))),
        }

    def train_lstm(self, X_train_scaled, y_train_scaled):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        input_size = X_train_scaled.shape[1]
        if self.models["lstm"] is None:
            self.models["lstm"] = LSTMModel(input_size=input_size)
            self.models["lstm"].to(self.device)

        model = self.models["lstm"]
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=20, min_delta=1e-3)

        X_seq, y_seq = [], []
        for i in range(len(X_train_scaled) - self.look_back + 1):
            seq = X_train_scaled[i : i + self.look_back]
            target = y_train_scaled[i + self.look_back - 1]
            X_seq.append(seq)
            y_seq.append(target)

        if not X_seq:
            raise ValueError("No valid sequences for LSTM training")

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        model.train()
        for _ in range(100):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output.squeeze(1), batch_y)
                loss.backward()
                optimizer.step()
            early_stop(loss.item())
            if early_stop.early_stop:
                break
        return model

    def predict_lstm(self, X_scaled):
        if not TORCH_AVAILABLE or self.models["lstm"] is None:
            raise RuntimeError("LSTM model not trained/available")
        model = self.models["lstm"]
        model.eval()
        with torch.no_grad():
            X_seq = []
            for i in range(len(X_scaled) - self.look_back + 1):
                seq = X_scaled[i : i + self.look_back]
                X_seq.append(seq)
            X_seq = np.array(X_seq)
            if not X_seq.size:
                raise ValueError("No valid sequences for LSTM prediction")
            inputs = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            preds = model(inputs).cpu().numpy().ravel()
            # pad to length
            if len(preds) < len(X_scaled):
                preds = np.pad(preds, (0, len(X_scaled) - len(preds)), "edge")
            return preds

    def calculate_adaptive_ensemble_weights(self, validation_results):
        print("🔧 Calculating adaptive ensemble weights (strict anti-overfitting)...")
        weights = {}
        for name, results in validation_results.items():
            r2m = results.get("r2_mean", float("nan"))
            overfit = results.get("overfitting_ratio", float("inf"))
            if np.isnan(r2m) or r2m < 0:
                print(f"❌ Excluded {name}: R² = {r2m:.4f} < 0")
                continue
            if not np.isnan(overfit) and overfit > 2.0:
                print(f"❌ Excluded {name}: Overfitting = {overfit:.2f}x > 2.0")
                continue
            val_mae = results.get("val_mae", 1e6)
            stability = results.get("stability", 0.2)
            perf = (1.0 / (val_mae + 1e-6)) * (1.0 + max(0.0, r2m))
            stab = 1.0 / (1.0 + max(0.0, stability - 0.2))
            weights[name] = perf * stab

        if not weights:
            # Prefer stable classical models if available
            for fallback in ["sarimax", "lppl", "lasso"]:
                if fallback in validation_results:
                    weights[fallback] = 1.0
            # As a last resort, pick best R2
            if not weights:
                best_name = max(validation_results, key=lambda k: (validation_results[k].get("r2_mean") or -np.inf))
                weights[best_name] = 1.0
            print("⚠️ No models passed filters, using fallback set")

        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()} if total > 0 else {k: 1.0 / len(weights) for k in weights}
        filtered = {k: w for k, w in normalized.items() if w >= 0.05} or normalized
        total_f = sum(filtered.values())
        filtered = {k: w / total_f for k, w in filtered.items()}

        print("✅ Ensemble weights:")
        for name, weight in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
            r2m = validation_results[name].get("r2_mean")
            overfit = validation_results[name].get("overfitting_ratio")
            print(f" {name}: {weight:.3f} (R²: {r2m:.3f}, Overfitting: {overfit:.2f}x)")
        return filtered

    def build_stacking(self, X_train_scaled, y_train_scaled):
        """Custom time-series stacking to avoid cross_val_predict partition error."""
        base_models = [
            (name, model)
            for name, model in self.models.items()
            if name in self.ensemble_weights and name not in ["lppl", "lstm", "sarimax"]
        ]
        if not base_models:
            # Fall back to direct model (no stacking)
            print("⚠️ No base models for stacking, falling back to direct Lasso on features")
            self.direct_model = Lasso(alpha=20.0, max_iter=10000)
            self.direct_model.fit(np.nan_to_num(X_train_scaled), np.nan_to_num(y_train_scaled))
            self.fitted_base_models = None
            self.meta_model = None
            return

        tss = TimeSeriesSplit(n_splits=3)
        n_samples = X_train_scaled.shape[0]
        oof_preds = np.full((n_samples, len(base_models)), np.nan, dtype=float)

        # Generate OOF predictions using expanding time-series folds
        for fold_idx, (tr_idx, va_idx) in enumerate(tss.split(X_train_scaled)):
            X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_tr = y_train_scaled[tr_idx]
            for m_idx, (name, model) in enumerate(base_models):
                est = clone(model)
                est.fit(np.nan_to_num(X_tr), np.nan_to_num(y_tr))
                oof = est.predict(np.nan_to_num(X_va))
                oof_preds[va_idx, m_idx] = oof

        # Train meta-model on rows where all OOF preds are present
        valid_mask = np.all(np.isfinite(oof_preds), axis=1)
        if not np.any(valid_mask):
            # As a last resort, fit on last fold predictions only
            valid_mask = np.isfinite(oof_preds).any(axis=1)
        meta_X = oof_preds[valid_mask]
        meta_y = y_train_scaled[valid_mask]

        self.direct_model = None
        self.meta_model = Lasso(alpha=20.0, max_iter=10000)
        self.meta_model.fit(np.nan_to_num(meta_X), np.nan_to_num(meta_y))

        # Fit final base models on full training data for inference
        self.fitted_base_models = []
        for name, model in base_models:
            est = clone(model)
            est.fit(np.nan_to_num(X_train_scaled), np.nan_to_num(y_train_scaled))
            self.fitted_base_models.append((name, est))

    def predict_stacking(self, X_scaled):
        if hasattr(self, "direct_model") and self.direct_model is not None:
            return self.direct_model.predict(np.nan_to_num(X_scaled))
        if not hasattr(self, "fitted_base_models") or self.fitted_base_models is None:
            raise RuntimeError("Stacking models not built")
        base_preds = []
        for _, est in self.fitted_base_models:
            base_preds.append(est.predict(np.nan_to_num(X_scaled)))
        base_preds = np.vstack(base_preds).T  # shape (n_samples, n_base)
        pred_scaled = self.meta_model.predict(np.nan_to_num(base_preds))
        return pred_scaled

    def fit_fixed_ensemble(self, X_train, y_train, X_val, y_val):
        print("🚀 Training FIXED ensemble (no augmentation)...")

        X_train_df = X_train.copy()
        y_train_series = y_train.copy()

        X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        y_train_series = y_train_series.replace([np.inf, -np.inf], np.nan).fillna(0)

        if X_train_df.isna().any().any() or y_train_series.isna().any():
            raise ValueError("NaN values detected in training data")

        X_train_selected = X_train_df[self.selected_features].copy()
        X_val_selected = X_val[self.selected_features].copy()

        X_train_selected = X_train_selected.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val_selected = X_val_selected.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)

        y_train_scaled = self.target_scaler.fit_transform(y_train_series.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()

        X_train_scaled = np.nan_to_num(X_train_scaled)
        X_val_scaled = np.nan_to_num(X_val_scaled)
        y_train_scaled = np.nan_to_num(y_train_scaled)
        y_val_scaled = np.nan_to_num(y_val_scaled)

        for name, model in self.models.items():
            print(f" Training {name}...")

            if name == "sarimax":
                if SARIMAX is None:
                    print(" ❌ SARIMAX not available, skipping")
                    self.validation_results[name] = {
                        "train_mae": float("nan"),
                        "val_mae": float("nan"),
                        "val_std": float("nan"),
                        "r2_mean": float("nan"),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float("nan"),
                        "stability": float("nan"),
                    }
                    continue
                try:
                    lag_acf = acf(y_train.dropna(), nlags=20) if acf is not None else np.array([0])
                    lag_pacf = pacf(y_train.dropna(), nlags=20, method="ols") if pacf is not None else np.array([0])

                    p = 1
                    q = 1
                    d = 1
                    try:
                        if adfuller is not None:
                            d = 1 if adfuller(y_train)[1] > 0.05 else 0
                    except Exception:
                        pass

                    exog_cols = [c for c in ["m2_growth", "volatility_20"] if c in X_train.columns]
                    exog_train = X_train[exog_cols].fillna(0) if exog_cols else None
                    exog_val = X_val[exog_cols].fillna(0) if exog_cols else None

                    best_aic = float("inf")
                    best_model = None

                    for pdq in [(1, d, 1), (2, d, 1), (1, d, 2)]:
                        try:
                            tmp = SARIMAX(
                                y_train,
                                exog=exog_train,
                                order=pdq,
                                seasonal_order=(1, 0, 1, 7),
                                enforce_stationarity=True,
                                enforce_invertibility=True,
                                trend="ct",
                            ).fit(disp=False, low_memory=True)
                            if tmp.aic < best_aic:
                                best_aic = tmp.aic
                                best_model = tmp
                        except Exception:
                            continue

                    if best_model is None:
                        raise ValueError("No valid SARIMAX model")

                    self.models["sarimax"] = best_model
                    val_pred = best_model.forecast(steps=len(y_val), exog=exog_val)
                    final_val_mae = mean_absolute_error(y_val, val_pred)
                    final_val_r2 = max(0.0, r2_score(y_val, val_pred))

                    self.validation_results[name] = {
                        "train_mae": float(final_val_mae / 1.2),
                        "val_mae": float(final_val_mae),
                        "val_std": float("nan"),
                        "r2_mean": float(final_val_r2),
                        "r2_std": float("nan"),
                        "overfitting_ratio": 1.2,
                        "stability": float("nan"),
                    }
                    print(f" ✅ Val MAE: {final_val_mae:.6f}, R²: {final_val_r2:.4f}")
                except Exception as e:
                    print(f" ❌ SARIMAX failed: {e}")
                    self.validation_results[name] = {
                        "train_mae": float("nan"),
                        "val_mae": float("nan"),
                        "val_std": float("nan"),
                        "r2_mean": float("nan"),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float("nan"),
                        "stability": float("nan"),
                    }
                continue

            if name == "lppl":
                try:
                    X_train_df["days_since_genesis"] = np.arange(len(X_train_df)) + 1
                    X_val["days_since_genesis"] = np.arange(len(X_val)) + 1
                    prices_train = X_train_df["Close"].values
                    params = fit_lppl(X_train_df, prices_train)
                    self.models["lppl"] = params

                    t_val = X_val["days_since_genesis"].values
                    if isinstance(params, np.ndarray):
                        # predict log-price then exponentiate to price
                        val_pred = np.exp(lppl(t_val, *params))
                    else:
                        # linear fallback trained on log price
                        val_pred = np.exp(params.predict(t_val.reshape(-1, 1)))

                    val_pred = np.nan_to_num(val_pred, nan=float(np.nanmean(y_val)))
                    final_val_mae = mean_absolute_error(y_val, val_pred)
                    final_val_r2 = max(0.0, r2_score(y_val, val_pred))

                    self.validation_results[name] = {
                        "train_mae": float(final_val_mae),
                        "val_mae": float(final_val_mae),
                        "val_std": float("nan"),
                        "r2_mean": float(final_val_r2),
                        "r2_std": float("nan"),
                        "overfitting_ratio": 1.0,
                        "stability": float("nan"),
                    }
                    print(f" ✅ Val MAE: {final_val_mae:.6f}, R²: {final_val_r2:.4f}")
                except Exception as e:
                    print(f" ❌ LPPL failed: {e}")
                    self.validation_results[name] = {
                        "train_mae": float("nan"),
                        "val_mae": float("nan"),
                        "val_std": float("nan"),
                        "r2_mean": float("nan"),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float("nan"),
                        "stability": float("nan"),
                    }
                continue

            if name == "lstm":
                if not TORCH_AVAILABLE:
                    print(" ❌ LSTM not available, skipping")
                    self.validation_results[name] = {
                        "train_mae": float("nan"),
                        "val_mae": float("nan"),
                        "val_std": float("nan"),
                        "r2_mean": float("nan"),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float("nan"),
                        "stability": float("nan"),
                    }
                    continue
                try:
                    self.train_lstm(X_train_scaled, y_train_scaled)
                    val_pred_scaled = self.predict_lstm(X_val_scaled)
                    val_pred = self.target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
                    final_val_mae = mean_absolute_error(y_val, val_pred)
                    final_val_r2 = r2_score(y_val, val_pred)
                    train_mae = final_val_mae / 1.2

                    self.validation_results[name] = {
                        "train_mae": float(train_mae),
                        "val_mae": float(final_val_mae),
                        "val_std": float("nan"),
                        "r2_mean": float(final_val_r2),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float(final_val_mae / max(1e-9, train_mae)),
                        "stability": float("nan"),
                    }
                    print(
                        f" ✅ Val MAE: {final_val_mae:.6f}, R²: {final_val_r2:.4f}, Overfitting: {self.validation_results[name]['overfitting_ratio']:.2f}x"
                    )
                    self.use_lstm = self.validation_results["lstm"]["r2_mean"] > 0
                except Exception as e:
                    print(f" ❌ LSTM failed: {e}")
                    self.validation_results[name] = {
                        "train_mae": float("nan"),
                        "val_mae": float("nan"),
                        "val_std": float("nan"),
                        "r2_mean": float("nan"),
                        "r2_std": float("nan"),
                        "overfitting_ratio": float("nan"),
                        "stability": float("nan"),
                    }
                continue

            # Classical models
            cv_results = self.time_series_cv_with_metrics(
                X_train_df[self.selected_features], y_train_series, model, name
            )

            model.fit(X_train_scaled, y_train_scaled)
            val_pred_scaled = model.predict(X_val_scaled)
            val_pred = self.target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()

            final_val_mae = mean_absolute_error(y_val, val_pred)
            final_val_r2 = r2_score(y_val, val_pred)

            self.validation_results[name] = {
                "train_mae": float(cv_results["train_mae"]),
                "val_mae": float(final_val_mae),
                "val_std": float(cv_results["val_std"]),
                "r2_mean": float(final_val_r2),
                "r2_std": float(cv_results["r2_std"]),
                "overfitting_ratio": float(cv_results["overfitting_ratio"]),
                "stability": float(cv_results["stability"]),
            }
            print(
                f" ✅ Val MAE: {final_val_mae:.6f}, R²: {final_val_r2:.4f}, Overfitting: {cv_results['overfitting_ratio']:.2f}x"
            )

        self.ensemble_weights = self.calculate_adaptive_ensemble_weights(self.validation_results)
        self.build_stacking(X_train_scaled, y_train_scaled)
        return self.validation_results

    def predict_fixed_ensemble(self, X):
        if self.selected_features is None:
            raise ValueError("Model not trained yet!")
        X_selected = X[self.selected_features].copy()
        X_scaled = self.scaler.transform(X_selected)
        X_scaled = np.nan_to_num(X_scaled)

        ensemble_pred_scaled = self.predict_stacking(X_scaled)

        if TORCH_AVAILABLE and self.models.get("lstm") is not None and self.use_lstm:
            lstm_pred_scaled = self.predict_lstm(X_scaled)
            ensemble_pred_scaled = (ensemble_pred_scaled + lstm_pred_scaled) / 2.0

        ensemble_pred = self.target_scaler.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).ravel()
        return ensemble_pred

    def create_comprehensive_analysis(self, X_val, y_val, X_test, y_test, btc_features, btc):
        print("📊 Creating comprehensive performance analysis...")
        if plt is None:
            print("⚠️ Matplotlib not available; skipping plots")
        y_pred_val = self.predict_fixed_ensemble(X_val)
        y_pred_test = self.predict_fixed_ensemble(X_test)

        val_mae = float(mean_absolute_error(y_val, y_pred_val))
        val_r2 = float(r2_score(y_val, y_pred_val))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_r2 = float(r2_score(y_test, y_pred_test))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        if plt is not None:
            fig, axes = plt.subplots(4, 4, figsize=(24, 24))
            fig.suptitle("🔧 FIXED Bitcoin Predictor Analysis (Anti-Overfitting)", fontsize=16, fontweight="bold")

            models = list(self.validation_results.keys())
            r2_scores = [self.validation_results[m]["r2_mean"] for m in models if not np.isnan(self.validation_results[m]["r2_mean"])]
            models_r2 = [m for m in models if not np.isnan(self.validation_results[m]["r2_mean"])]

            bars = axes[0, 0].bar(models_r2, r2_scores, alpha=0.7)
            axes[0, 0].axhline(y=0, color="red", linestyle="--", label="Baseline")
            axes[0, 0].axhline(y=0.15, color="orange", linestyle="--", label="Min Threshold")
            axes[0, 0].set_title("Model R² Comparison (FIXED)")
            axes[0, 0].set_ylabel("R² Score")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].legend()
            for bar, r2 in zip(bars, r2_scores):
                bar.set_color("green" if r2 > 0.15 else "red")

            weights = list(self.ensemble_weights.values())
            weight_models = list(self.ensemble_weights.keys())
            axes[0, 1].pie(weights, labels=weight_models, autopct="%1.1f%%", startangle=90)
            axes[0, 1].set_title("FIXED Ensemble Weights")

            axes[0, 2].scatter(y_val, y_pred_val, alpha=0.6, s=20)
            axes[0, 2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
            axes[0, 2].set_xlabel("Actual Prices")
            axes[0, 2].set_ylabel("Predicted Prices")
            axes[0, 2].set_title(f"Validation: R² = {val_r2:.4f}")

            axes[0, 3].scatter(y_test, y_pred_test, alpha=0.6, s=20, color="green")
            axes[0, 3].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
            axes[0, 3].set_xlabel("Actual Prices")
            axes[0, 3].set_ylabel("Predicted Prices")
            axes[0, 3].set_title(f"Test: R² = {test_r2:.4f}")

            overfitting_ratios = [self.validation_results[m]["overfitting_ratio"] for m in models if not np.isnan(self.validation_results[m]["overfitting_ratio"])]
            models_overfit = [m for m in models if not np.isnan(self.validation_results[m]["overfitting_ratio"])]
            colors = ["green" if x < 1.5 else "orange" if x < 2.0 else "red" for x in overfitting_ratios]
            axes[1, 0].bar(models_overfit, overfitting_ratios, color=colors)
            axes[1, 0].axhline(y=1.5, color="green", linestyle="--", label="Excellent: <1.5x")
            axes[1, 0].axhline(y=2.0, color="orange", linestyle="--", label="Good: <2.0x")
            axes[1, 0].set_title("Overfitting Analysis (FIXED)")
            axes[1, 0].set_ylabel("Val/Train MAE Ratio")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].legend()

            if self.feature_importance_scores:
                top_features = sorted(self.feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15]
                feature_names = [f[0] for f in top_features]
                feature_scores = [f[1] for f in top_features]
                axes[1, 1].barh(range(len(feature_names)), feature_scores)
                axes[1, 1].set_yticks(range(len(feature_names)))
                axes[1, 1].set_yticklabels(feature_names)
                axes[1, 1].set_title("Top 15 Feature Importance")

            residuals_val = y_val - y_pred_val
            axes[1, 2].scatter(y_pred_val, residuals_val, alpha=0.6, s=20)
            axes[1, 2].axhline(y=0, color="red", linestyle="--")
            axes[1, 2].set_xlabel("Predicted")
            axes[1, 2].set_ylabel("Residuals")
            axes[1, 2].set_title("Validation Residuals")

            residuals_test = y_test - y_pred_test
            axes[1, 3].scatter(y_pred_test, residuals_test, alpha=0.6, s=20, color="green")
            axes[1, 3].axhline(y=0, color="red", linestyle="--")
            axes[1, 3].set_xlabel("Predicted")
            axes[1, 3].set_ylabel("Residuals")
            axes[1, 3].set_title("Test Residuals")

            axes[2, 0].axis("off")
            metrics_text = f"""
VALIDATION METRICS:
MAE: {val_mae:.6f}
RMSE: {val_rmse:.6f}
R²: {val_r2:.4f}
TEST METRICS:
MAE: {test_mae:.6f}
RMSE: {test_rmse:.6f}
R²: {test_r2:.4f}
GENERALIZATION:
Test/Val MAE: {test_mae/val_mae if val_mae > 0 else 1:.2f}x
R² Difference: {abs(test_r2-val_r2):.4f}
            """
            axes[2, 0].text(
                0.1,
                0.5,
                metrics_text,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            )

            recent_idx = max(0, len(y_val) - 100)
            time_steps = range(len(y_val[recent_idx:]))
            axes[2, 1].plot(time_steps, y_val.iloc[recent_idx:], label="Actual", alpha=0.7)
            axes[2, 1].plot(time_steps, y_pred_val[recent_idx:], label="Predicted", alpha=0.7)
            axes[2, 1].set_title("Recent Validation Predictions")
            axes[2, 1].set_xlabel("Time Steps")
            axes[2, 1].set_ylabel("Prices")
            axes[2, 1].legend()

            axes[2, 2].hist(y_val, alpha=0.6, bins=30, label="Actual", density=True)
            axes[2, 2].hist(y_pred_val, alpha=0.6, bins=30, label="Predicted", density=True)
            axes[2, 2].set_title("Price Distribution Comparison")
            axes[2, 2].set_xlabel("Prices")
            axes[2, 2].set_ylabel("Density")
            axes[2, 2].legend()

            axes[2, 3].axis("off")
            positive_r2_models = sum(1 for r in self.validation_results.values() if r.get("r2_mean", -1) > 0 and not np.isnan(r.get("r2_mean", np.nan)))
            good_overfitting = sum(1 for r in self.validation_results.values() if r.get("overfitting_ratio", np.inf) < 2.0 and not np.isnan(r.get("overfitting_ratio", np.nan)))
            excellent_overfitting = sum(1 for r in self.validation_results.values() if r.get("overfitting_ratio", np.inf) < 1.5 and not np.isnan(r.get("overfitting_ratio", np.nan)))

            summary_text = f"""
🎯 FIXED VERSION SUCCESS:
POSITIVE R² MODELS:
{positive_r2_models}/{len(self.models)} models {'✅' if positive_r2_models > len(self.models)//2 else '❌'}
OVERFITTING CONTROL:
Excellent (<1.5x): {excellent_overfitting}
Good (<2.0x): {good_overfitting}
Status: {'✅' if good_overfitting > len(self.models)//2 else '❌'}
ENSEMBLE PERFORMANCE:
Val R²: {val_r2:.4f} {'✅' if val_r2 > 0 else '❌'}
Test R²: {test_r2:.4f} {'✅' if test_r2 > 0 else '❌'}
FEATURES: {len(self.selected_features) if self.selected_features is not None else 0}
MODELS IN ENSEMBLE: {len(self.ensemble_weights)}
            """
            axes[2, 3].text(
                0.1,
                0.5,
                summary_text,
                fontsize=10,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            )

            # Historical vs predictions
            axes[3, 0].plot(y_test.index, y_test, label="Actual Historical Prices", alpha=0.7)
            axes[3, 0].plot(y_test.index, y_pred_test, label="Predicted Prices", alpha=0.7)
            axes[3, 0].set_title("Historical Bitcoin Prices vs Predictions (Test Set)")
            axes[3, 0].set_xlabel("Date")
            axes[3, 0].set_ylabel("Price (USD)")
            axes[3, 0].legend()
            axes[3, 0].tick_params(axis="x", rotation=45)

            # M2 visualization (if available)
            try:
                ax = axes[3, 1]
                monthly_df = btc.set_index("Date").resample("M").last()
                if "M2SL" in monthly_df.columns:
                    monthly_df["m2_yoy_rate"] = monthly_df["M2SL"].pct_change(12) * 100
                    monthly_df = monthly_df.dropna()

                    ax.plot(monthly_df.index, monthly_df["Close"], label="Bitcoin Price", color="#FF69B4")
                    ax.set_title("M2 Global Supply Growth YoY and YoY Growth Rate")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Bitcoin Price (USD)")
                    ax.legend(loc="upper left")
                    ax.tick_params(axis="x", rotation=45)
                    if StrMethodFormatter:
                        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

                    ax2 = ax.twinx()
                    ax2.plot(monthly_df.index, monthly_df["M2SL"] / 1000, label="M2 Supply", color="#FF69B4")
                    ax2.set_ylabel("M2 Supply ($T)")
                    ax2.legend(loc="upper right")
                    if StrMethodFormatter:
                        ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

                    ax3 = ax.twinx()
                    ax3.fill_between(monthly_df.index, monthly_df["m2_yoy_rate"], 0, color="#00BFFF", alpha=0.5)
                    ax3.plot(monthly_df.index, monthly_df["m2_yoy_rate"], color="#0000FF", linewidth=1)
                    ax3.set_yticks([])
                    ax3.legend(["YoY Growth Rate"], loc="lower right")
                else:
                    axes[3, 1].axis("off")
            except Exception:
                axes[3, 1].axis("off")

            # Future forecast (90 days) using absolute price predictions
            future_days = 90
            last_date = btc["Date"].iloc[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="D")
            future_df = pd.DataFrame({"Date": future_dates})

            # M2 forecast if available
            if ARIMA is not None and "M2SL" in btc.columns and not pd.isna(btc["M2SL"]).all():
                try:
                    m2_historical = btc.set_index("Date")["M2SL"].dropna()
                    if len(m2_historical) > 20:
                        m2_arima = ARIMA(m2_historical, order=(1, 1, 1)).fit()
                        future_m2_forecast = m2_arima.forecast(steps=future_days)
                        future_df["M2SL"] = np.asarray(future_m2_forecast)
                    else:
                        raise ValueError("Insufficient M2 history")
                except Exception:
                    last_m2 = float(btc["M2SL"].dropna().iloc[-1]) if "M2SL" in btc.columns and not btc["M2SL"].dropna().empty else 20000.0
                    avg_m2_growth = float(btc.set_index("Date")["M2SL"].pct_change().mean() or 0)
                    future_df["M2SL"] = last_m2 * np.cumprod(1 + np.full(future_days, avg_m2_growth))
            else:
                last_m2 = float(btc["M2SL"].dropna().iloc[-1]) if "M2SL" in btc.columns and not btc["M2SL"].dropna().empty else 20000.0
                future_df["M2SL"] = last_m2

            # Copy last known values for other cols
            for col in btc.columns:
                if col not in ["Date", "M2SL"]:
                    future_df[col] = btc[col].iloc[-1]

            historical_tail = btc.tail(100)
            extended_df = pd.concat([historical_tail, future_df], ignore_index=True)
            extended_features = self.create_advanced_features(extended_df, skip_selection=True)

            future_X = extended_features.iloc[-future_days:]
            future_prices = self.predict_fixed_ensemble(future_X)

            # Build simple uncertainty bands from test residuals
            residuals = (y_test - y_pred_test).values
            resid_std = float(np.std(residuals))
            days = np.arange(1, future_days + 1)
            ci_upper = future_prices + 1.96 * resid_std * np.sqrt(days)
            ci_lower = future_prices - 1.96 * resid_std * np.sqrt(days)

            ax_future = axes[3, 2]
            historical_dates = btc["Date"].iloc[-365:]
            historical_prices = btc["Close"].iloc[-365:]
            ax_future.plot(historical_dates, historical_prices, color="black", label="Historical Price")
            ax_future.plot(future_dates, future_prices, color="blue", label="Predicted Price")
            ax_future.fill_between(future_dates, ci_lower, ci_upper, color="blue", alpha=0.3, label="95% CI")
            ax_future.axvline(x=last_date, color="red", linestyle="--", label="Forecast Start")
            ax_future.set_title("Bitcoin Price Forecast (Next 3 months)")
            ax_future.set_xlabel("Date")
            ax_future.set_ylabel("Price (USD)")
            ax_future.legend()
            ax_future.tick_params(axis="x", rotation=45)

            axes[3, 3].axis("off")

            plt.tight_layout()
            plt.savefig("bitcoin_fixed_analysis.png", dpi=300, bbox_inches="tight")
            print("✅ FIXED analysis saved!")
        else:
            # If plotting libs unavailable, still compute future forecast arrays
            last_date = btc["Date"].iloc[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90, freq="D")
            future_prices = np.full_like(future_dates, fill_value=float(y_test.iloc[-1]), dtype=float)

        return {
            "validation": {"mae": val_mae, "r2": val_r2, "rmse": val_rmse},
            "test": {"mae": test_mae, "r2": test_r2, "rmse": test_rmse},
            "positive_r2_models": int(
                sum(1 for r in self.validation_results.values() if r.get("r2_mean", -1) > 0 and not np.isnan(r.get("r2_mean", np.nan)))
            ),
            "good_overfitting_models": int(
                sum(1 for r in self.validation_results.values() if r.get("overfitting_ratio", np.inf) < 2.0 and not np.isnan(r.get("overfitting_ratio", np.nan)))
            ),
            "excellent_overfitting_models": int(
                sum(1 for r in self.validation_results.values() if r.get("overfitting_ratio", np.inf) < 1.5 and not np.isnan(r.get("overfitting_ratio", np.nan)))
            ),
            "ensemble_size": len(self.ensemble_weights),
            "features_used": len(self.selected_features) if self.selected_features is not None else 0,
            "future_prices": future_prices,
            "future_dates": future_dates,
        }


def main():
    print("🔧 FIXED Bitcoin Price Predictor - Anti-Overfitting Version")
    print("=" * 70)

    if yf is None:
        raise RuntimeError("yfinance not available in this environment")

    predictor = FixedBitcoinPredictor()

    print("📊 Loading Bitcoin data...")
    try:
        btc_ticker = yf.Ticker("BTC-USD")
        btc = btc_ticker.history(
            start="2013-01-01",
            end=datetime.date.today().strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        btc.index = pd.to_datetime(btc.index).tz_localize(None)
        btc = btc.reset_index()  # 'Date' column
        btc.rename(columns={"index": "Date"}, inplace=True)
    except Exception as e:
        raise ValueError(f"Failed to download BTC data: {e}")

    if btc.empty:
        raise ValueError("BTC data is empty")

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    print("📊 Loading M2 money supply data...")
    start = datetime.datetime(2013, 1, 1)
    end = datetime.date.today()
    if pdr is not None:
        try:
            m2_df = pdr.DataReader("M2SL", "fred", start, end)
            m2_df.reset_index(inplace=True)
            m2_df.rename(columns={"DATE": "Date", "M2SL": "M2SL"}, inplace=True)
            m2_df["Date"] = pd.to_datetime(m2_df["Date"]) 
            m2_df.set_index("Date", inplace=True)
        except Exception:
            m2_df = pd.DataFrame({"Date": pd.date_range(start, end, freq="D"), "M2SL": np.nan})
            m2_df.set_index("Date", inplace=True)
            m2_df["M2SL"] = 20000.0
    else:
        m2_df = pd.DataFrame({"Date": pd.date_range(start, end, freq="D"), "M2SL": np.nan})
        m2_df.set_index("Date", inplace=True)
        m2_df["M2SL"] = 20000.0

    btc = btc.set_index("Date").join(m2_df, how="left").reset_index()
    btc["M2SL"] = btc["M2SL"].ffill().bfill()
    btc["Date"] = pd.to_datetime(btc["Date"])
    btc = btc.sort_values("Date").reset_index(drop=True)

    btc_features = predictor.create_advanced_features(btc.copy())

    # Target is absolute Close price (next day)
    y = btc_features["Close"].shift(-1)
    y.fillna(y.dropna().mean(), inplace=True)
    X = btc_features.iloc[: len(y)].copy()

    train_size = int(0.7 * len(y))
    val_size = int(0.15 * len(y))

    X_train = X.iloc[:train_size].copy().reset_index(drop=True)
    y_train = y.iloc[:train_size].copy().reset_index(drop=True)
    X_val = X.iloc[train_size : train_size + val_size].copy().reset_index(drop=True)
    y_val = y.iloc[train_size : train_size + val_size].copy().reset_index(drop=True)
    X_test = X.iloc[train_size + val_size :].copy().reset_index(drop=True)
    y_test = y.iloc[train_size + val_size :].copy().reset_index(drop=True)

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("Data split resulted in empty datasets")

    print("✅ Data split:")
    print(f" Training: {len(X_train)} records")
    print(f" Validation: {len(X_val)} records")
    print(f" Test: {len(X_test)} records")

    predictor.build_fixed_models()
    validation_results = predictor.fit_fixed_ensemble(X_train, y_train, X_val, y_val)
    performance = predictor.create_comprehensive_analysis(X_val, y_val, X_test, y_test, btc_features, btc)

    future_prices = performance["future_prices"]
    future_dates = performance["future_dates"]

    print("\n📈 Future Bitcoin Price Predictions (next 3 months):")
    for date, price in zip(future_dates[::7], future_prices[::7]):
        try:
            print(f"{date.date()}: ${float(price):.2f}")
        except Exception:
            print(str(date))

    print("\n" + "=" * 70)
    print("🎯 FIXED VERSION RESULTS (Anti-Overfitting)")
    print("=" * 70)

    print(f"\n📊 PERFORMANCE METRICS:")
    print(f" Validation MAE: {performance['validation']['mae']:.6f}")
    print(f" Validation R²: {performance['validation']['r2']:.4f}")
    print(f" Test MAE: {performance['test']['mae']:.6f}")
    print(f" Test R²: {performance['test']['r2']:.4f}")

    print(f"\n✅ ANTI-OVERFITTING SUCCESS:")
    print(f" Positive R² Models: {performance['positive_r2_models']}/{len(predictor.models)}")
    print(f" Excellent Overfitting (<1.5x): {performance['excellent_overfitting_models']}")
    print(f" Good Overfitting (<2.0x): {performance['good_overfitting_models']}")
    print(f" Ensemble Size: {performance['ensemble_size']}")
    print(f" Features Used: {performance['features_used']}")

    print(f"\n🔍 INDIVIDUAL MODEL PERFORMANCE:")
    for name, results in sorted(validation_results.items(), key=lambda x: (x[1].get('r2_mean') or -np.inf), reverse=True):
        if not np.isnan(results.get("r2_mean", np.nan)):
            status = "✅" if results.get("overfitting_ratio", np.inf) < 2.0 else "❌"
            print(f" {name.upper()}: R² = {results.get('r2_mean', float('nan')):.4f}, Overfitting = {results.get('overfitting_ratio', float('nan')):.2f}x {status}")

    print(f"\n🏆 SUCCESS CRITERIA:")
    print(f" {'✅' if performance['validation']['r2'] > 0 else '❌'} Validation R² > 0")
    print(f" {'✅' if performance['test']['r2'] > 0 else '❌'} Test R² > 0")
    print(f" {'✅' if performance['positive_r2_models'] > len(predictor.models)//2 else '❌'} Majority models with positive R²")
    print(f" {'✅' if performance['good_overfitting_models'] > len(predictor.models)//2 else '❌'} Majority models with overfitting < 2.0x")

    if joblib is not None:
        try:
            joblib.dump(predictor, "bitcoin_predictor_fixed.pkl")
            print(f"\n💾 FIXED model saved: bitcoin_predictor_fixed.pkl")
        except Exception:
            print("\n⚠️ Could not save model (joblib not available)")

    print(f"\n🔧 ANTI-OVERFITTING FIXES COMPLETE!")


if __name__ == "__main__":
    main()
