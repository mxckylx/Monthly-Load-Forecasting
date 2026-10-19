import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

st.set_page_config(
    page_title="Load Forecast Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--navy:#142b4a;--teal:#0e9384;--ink:#172b45;--muted:#708096;--line:#e4eaf1;--wash:#f5f8fb;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
[data-testid="stAppViewContainer"]{background:#f7f9fc}
[data-testid="stHeader"]{background:rgba(247,249,252,.9)}
.block-container{max-width:1440px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:.9rem}
.brand-lockup{display:flex;align-items:center;gap:12px;padding:8px 2px 22px;border-bottom:1px solid var(--line);margin-bottom:22px}
.brand-icon{width:42px;height:42px;border-radius:13px;background:#e2f5f2;color:#087d73;display:flex;align-items:center;justify-content:center;font-size:23px;font-weight:800}
.brand-name{font-family:Manrope,sans-serif;font-weight:800;font-size:1.02rem;color:var(--navy);line-height:1.15}
.brand-caption{font-size:.72rem;color:#7d8ca0;margin-top:4px;letter-spacing:.04em}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:.7rem;font-weight:800;color:var(--teal);margin-bottom:8px}
.page-title{font-family:Manrope,sans-serif;font-size:clamp(1.8rem,3vw,2.45rem);font-weight:800;letter-spacing:-.045em;line-height:1.15;color:var(--navy);margin:0}
.page-subtitle{color:var(--muted);font-size:.98rem;margin-top:10px;max-width:780px;line-height:1.65}
.hero{background:linear-gradient(120deg,#142b4a 0%,#1a4564 70%,#176b70 100%);border-radius:22px;padding:30px 32px;color:white;margin:14px 0 24px;position:relative;overflow:hidden}
.hero:after{content:'';position:absolute;width:280px;height:280px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-50px;top:-115px;box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 70px rgba(255,255,255,.025)}
.hero-label{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#a8e7dd;font-weight:800;margin-bottom:12px}
.hero-title{font-family:Manrope,sans-serif;font-size:clamp(1.65rem,3vw,2.5rem);font-weight:800;line-height:1.15;letter-spacing:-.04em;max-width:750px}
.hero-copy{font-size:.95rem;color:#d7e5ef;line-height:1.7;max-width:680px;margin-top:12px}
.section-heading{font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;color:var(--navy);margin:1.4rem 0 .25rem}
.section-copy{color:var(--muted);font-size:.9rem;line-height:1.6;margin-bottom:1rem}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 20px;box-shadow:0 4px 16px rgba(20,43,74,.035)}
[data-testid="stMetricLabel"]{color:#77869a;font-size:.82rem;font-weight:600}
[data-testid="stMetricValue"]{font-family:Manrope,sans-serif;color:var(--navy);font-weight:800;font-size:1.6rem}
[data-testid="stMetricDelta"]{font-size:.78rem}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #c8d6e2;border-radius:14px;padding:10px}
.stButton>button[kind="primary"]{background:var(--teal);border:1px solid var(--teal);color:#fff;border-radius:11px;min-height:3rem;font-weight:700;box-shadow:0 5px 14px rgba(14,147,132,.15)}
.stButton>button[kind="primary"]:hover{background:#08796f;border-color:#08796f}
.stButton>button:not([kind="primary"]),.stDownloadButton>button{border-radius:10px;border:1px solid #d8e1ea;font-weight:650}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff}
[data-testid="stRadio"] label{font-weight:600}
hr{border-color:var(--line)}
.note{color:var(--muted);font-size:.8rem;line-height:1.55}
.status-pill{display:inline-block;background:#e2f5f2;color:#087d73;border-radius:100px;padding:5px 10px;font-size:.72rem;font-weight:800;letter-spacing:.03em}
@media(max-width:700px){.block-container{padding:1rem 1rem 2rem}.hero{padding:23px 20px;border-radius:17px}.page-title{font-size:1.8rem}}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-icon">ϟ</div>'
        '<div><div class="brand-name">Load Forecast<br>Studio</div>'
        '<div class="brand-caption">MONTHLY ENERGY ANALYTICS</div></div></div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "WORKSPACE",
        ["Home", "Forecast", "Model Comparison", "About"],
        key="navigation"
    )

    st.divider()
    st.markdown("**DATA SOURCE**")

    uploaded_file = st.file_uploader(
        "Upload training history (through 2025)",
        type=["xlsx", "xls", "csv"],
        help="Use the historical monthly data only; pandemic years are excluded."
    )

    validation_file = st.file_uploader(
        "Optional: upload 2026 actuals for validation",
        type=["xlsx", "xls", "csv"],
        help="For the AMTES–Meralco workbook, the app reads the '2026' sheet."
    )
    kwh_col_num = st.number_input(
        "2026 Excel sheet: kWh column number (A=1, L=12)",
        min_value=1, max_value=40, value=12, step=1,
        help="Pick the same meter/column you used for training. Only used for Excel validation files."
    )
    st.caption("Training data: Date/Year + Month + Consumption_kWh. Keep 2026 actuals separate.")

    if uploaded_file is not None:
        st.markdown(
            '<span class="status-pill">FILE SELECTED</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div class="note">Research prototype · LSTM + XGBoost<br>'
        'Forecasts are estimates, not actual utility bills.</div>',
        unsafe_allow_html=True
    )

# MODEL LOGIC
LOOKBACK = 12
EVAL_RECURSIVE = "Recursive (multi-step)"
EVAL_ONESTEP = "One-step-ahead"

# First item of each grid is the untuned BASELINE model (as in the thesis flowchart).
XGB_GRID = [
    dict(n_estimators=400, max_depth=4, learning_rate=0.025, subsample=1.0, colsample_bytree=1.0, min_child_weight=1),
    dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
    dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=2),
    dict(n_estimators=100, max_depth=2, learning_rate=0.10, subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
]
LSTM_GRID = [
    dict(units=48, dense=24, epochs=100, dropout=0.10),
    dict(units=16, dense=8, epochs=60, dropout=0.10),
    dict(units=32, dense=16, epochs=60, dropout=0.20),
]

def xgb_cfg_text(cfg):
    return (f"trees={cfg['n_estimators']}, depth={cfg['max_depth']}, "
            f"lr={cfg['learning_rate']}, subsample={cfg['subsample']}")

def lstm_cfg_text(cfg):
    return (f"units={cfg['units']}, dense={cfg['dense']}, "
            f"epochs={cfg['epochs']}, dropout={cfg['dropout']}")

def is_consecutive_window(periods, start_idx, end_idx):
    window = periods[start_idx:end_idx + 1]
    return len(window) <= 1 or all(
        window[j].ordinal - window[j - 1].ordinal == 1
        for j in range(1, len(window))
    )

def latest_contiguous_segment_length(periods):
    if not periods:
        return 0

    count = 1
    for i in range(len(periods) - 1, 0, -1):
        if periods[i].ordinal - periods[i - 1].ordinal != 1:
            break
        count += 1

    return count

def fill_monthly_hours(hours, periods):
    """Fill missing operating hours using the same calendar month's history."""
    hours = np.asarray(hours, dtype=float).copy()
    periods = list(periods)
    valid = np.isfinite(hours)
    overall = float(np.nanmean(hours[valid])) if valid.any() else 0.0
    month_means = {}
    for month in range(1, 13):
        vals = [hours[i] for i, period in enumerate(periods)
                if period.month == month and np.isfinite(hours[i])]
        month_means[month] = float(np.mean(vals)) if vals else overall
    for i, period in enumerate(periods):
        if not np.isfinite(hours[i]):
            hours[i] = month_means[period.month]
    return hours, month_means, overall

def future_hours_for_periods(periods, month_means, overall):
    return np.asarray([month_means.get(p.month, overall) for p in periods], dtype=float)

def xgb_lags(mode):
    """'paper' = lag 1, 2, 3, 12 (as written in the thesis). 'extended' = lag 1-12 + rolling means."""
    return [1, 2, 3, 12] if mode == "paper" else list(range(1, LOOKBACK + 1))

def xgb_features(history, target_period, mode="paper", hours_history=None, target_hours=None):
    h = np.asarray(history, dtype=float)
    if len(h) < LOOKBACK:
        raise ValueError("At least 12 consecutive monthly values are needed.")
    lags = xgb_lags(mode)
    features = [float(h[-lag]) for lag in lags]
    if mode == "extended":
        features += [float(np.mean(h[-3:])), float(np.mean(h[-6:])), float(np.mean(h[-12:]))]
    features += [np.sin(2 * np.pi * target_period.month / 12),
                 np.cos(2 * np.pi * target_period.month / 12)]
    if hours_history is not None:
        hh = np.asarray(hours_history, dtype=float)
        if len(hh) < LOOKBACK:
            raise ValueError("At least 12 monthly operating-hour values are needed.")
        features += [float(hh[-lag]) for lag in lags]
        if mode == "extended":
            features += [float(np.mean(hh[-3:])), float(np.mean(hh[-6:])), float(np.mean(hh[-12:]))]
        features += [float(target_hours if target_hours is not None else np.mean(hh[-12:]))]
    return features

def make_xgb_training_data(values, periods, mode="paper", hours=None):
    X, target = [], []
    for i in range(LOOKBACK, len(values)):
        if is_consecutive_window(periods, i - LOOKBACK, i):
            target_hours = float(hours[i]) if hours is not None else None
            X.append(xgb_features(values[i-LOOKBACK:i], periods[i], mode,
                                  hours[i-LOOKBACK:i] if hours is not None else None,
                                  target_hours))
            target.append(values[i])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)

def fit_xgb(X, target, cfg):
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1, **cfg)
    model.fit(X, target)
    return model

def recursive_xgb_forecast(model, history, mode, start_period, steps, hours_history=None, future_hours=None):
    history = list(np.asarray(history, dtype=float))
    hours_history = list(np.asarray(hours_history, dtype=float)) if hours_history is not None else None
    predictions = []
    for step in range(steps):
        target_period = start_period + step
        target_hours = float(future_hours[step]) if future_hours is not None else None
        prediction = max(0.0, float(model.predict(np.asarray([xgb_features(
            history, target_period, mode, hours_history, target_hours)]))[0]))
        predictions.append(prediction)
        history.append(prediction)
        if hours_history is not None and target_hours is not None:
            hours_history.append(target_hours)
    return np.asarray(predictions)

def onestep_xgb(model, y, periods, hours, mode, idxs, target_hours=None):
    """One-step-ahead: every prediction uses ACTUAL past months, never its own predictions."""
    preds = []
    for k, i in enumerate(idxs):
        hh = hours[i - LOOKBACK:i] if hours is not None else None
        th = float(target_hours[k]) if target_hours is not None else None
        feats = xgb_features(y[i - LOOKBACK:i], periods[i], mode, hh, th)
        preds.append(max(0.0, float(model.predict(np.asarray([feats]))[0])))
    return np.asarray(preds)

def month_features(period):
    angle = 2 * np.pi * period.month / 12
    return [float(np.sin(angle)), float(np.cos(angle))]

def train_lstm(training_values, training_periods, training_hours=None, cfg=None):
    cfg = cfg or LSTM_GRID[0]
    training_values = np.asarray(training_values, dtype=float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(training_values.reshape(-1, 1)).flatten()
    if training_hours is not None:
        hours_scale = max(float(np.nanmax(training_hours)), 1.0)
        scaled_hours = np.asarray(training_hours, dtype=float) / hours_scale
    else:
        hours_scale = 1.0
        scaled_hours = None
    X, target = [], []
    for i in range(LOOKBACK, len(scaled)):
        if is_consecutive_window(training_periods, i-LOOKBACK, i):
            seq = []
            for j in range(i-LOOKBACK, i):
                row = [scaled[j]] + month_features(training_periods[j])
                if scaled_hours is not None:
                    row.append(float(scaled_hours[j]))
                seq.append(row)
            X.append(seq)
            target.append(scaled[i])
    if len(X) < 12:
        raise ValueError("Not enough consecutive monthly sequences to train LSTM. At least 12 valid training sequences are recommended.")
    n_features = 4 if scaled_hours is not None else 3
    X = np.asarray(X, dtype=float).reshape(-1, LOOKBACK, n_features)
    target = np.asarray(target, dtype=float)
    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(cfg["units"], activation="tanh", dropout=cfg["dropout"]),
        Dense(cfg["dense"], activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
    model.fit(X, target, epochs=cfg["epochs"], batch_size=8, verbose=0,
              callbacks=[EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)], shuffle=False)
    return model, scaler, hours_scale if scaled_hours is not None else None

def recursive_lstm_forecast(model, scaler, history, start_period, steps, hours_history=None, future_hours=None, hours_scale=None):
    """start_period = the period of the LAST value in `history`."""
    history = np.asarray(history, dtype=float).tolist()
    if len(history) < LOOKBACK:
        raise ValueError("At least 12 history values are required.")
    scaled_history = scaler.transform(np.asarray(history).reshape(-1, 1)).flatten().tolist()
    history_periods = [start_period - (LOOKBACK - 1 - i) for i in range(LOOKBACK)]
    hh = np.asarray(hours_history, dtype=float).tolist() if hours_history is not None else None
    predictions = []
    for step in range(steps):
        target_period = start_period + step
        seq = []
        for j in range(LOOKBACK):
            row = [scaled_history[-LOOKBACK+j]] + month_features(history_periods[j])
            if hh is not None:
                row.append(float(hh[-LOOKBACK+j] / (hours_scale or 1.0)))
            seq.append(row)
        sequence = np.asarray(seq, dtype=float).reshape(1, LOOKBACK, 4 if hh is not None else 3)
        scaled_prediction = float(model.predict(sequence, verbose=0)[0][0])
        prediction = max(0.0, float(scaler.inverse_transform([[scaled_prediction]])[0][0]))
        predictions.append(prediction)
        scaled_history.append(float(scaler.transform([[prediction]])[0][0]))
        if hh is not None and future_hours is not None:
            hh.append(float(future_hours[step]))
        # Keep month features aligned after each recursive prediction.
        history_periods = history_periods[1:] + [target_period + 1]
    return np.asarray(predictions)

def onestep_lstm(model, scaler, y, periods, hours, hours_scale, idxs):
    preds = []
    for i in idxs:
        p = recursive_lstm_forecast(
            model, scaler, y[i - LOOKBACK:i], periods[i] - 1, 1,
            hours[i - LOOKBACK:i] if hours is not None else None, None, hours_scale
        )
        preds.append(p[0])
    return np.asarray(preds)

def seasonal_naive(target_periods, value_map):
    """Forecast for a month = actual of the same month one year earlier (reference benchmark)."""
    vm = dict(value_map)
    out = []
    for p in target_periods:
        v = vm.get(p - 12, np.nan)
        out.append(v)
        if p not in vm:
            vm[p] = v
    return np.asarray(out, dtype=float)

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if np.any(nonzero)
        else np.nan
    )

    return mae, rmse, mape

def naive_metrics(actual, naive_pred):
    actual = np.asarray(actual, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)
    mask = np.isfinite(naive_pred) & np.isfinite(actual)
    if not mask.any():
        return (np.nan, np.nan, np.nan)
    return calculate_metrics(actual[mask], naive_pred[mask])

def build_eval_df(eval_dates, actual, lstm_pred, xgb_pred, naive_pred):
    df = pd.DataFrame({
        "Month": [str(d) for d in eval_dates],
        "Actual (kWh)": np.asarray(actual, dtype=float),
        "LSTM Prediction (kWh)": np.asarray(lstm_pred, dtype=float),
        "XGBoost Prediction (kWh)": np.asarray(xgb_pred, dtype=float),
        "Seasonal Naive (kWh)": np.asarray(naive_pred, dtype=float),
    })
    df["LSTM AE (kWh)"] = (df["Actual (kWh)"] - df["LSTM Prediction (kWh)"]).abs()
    df["XGBoost AE (kWh)"] = (df["Actual (kWh)"] - df["XGBoost Prediction (kWh)"]).abs()
    df["d = AE_LSTM - AE_XGBoost"] = df["LSTM AE (kWh)"] - df["XGBoost AE (kWh)"]
    safe_actual = df["Actual (kWh)"].replace(0, np.nan)
    df["LSTM APE (%)"] = df["LSTM AE (kWh)"] / safe_actual * 100
    df["XGBoost APE (%)"] = df["XGBoost AE (kWh)"] / safe_actual * 100
    return df

def eval_metric_bundle(df):
    actual = df["Actual (kWh)"].to_numpy()
    lm = calculate_metrics(actual, df["LSTM Prediction (kWh)"].to_numpy())
    xm = calculate_metrics(actual, df["XGBoost Prediction (kWh)"].to_numpy())
    nm = naive_metrics(actual, df["Seasonal Naive (kWh)"].to_numpy())
    metrics = pd.DataFrame({
        "Metric": ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"],
        "LSTM": list(lm),
        "XGBoost": list(xm),
        "Seasonal Naive": list(nm)
    })
    return metrics, lm, xm, nm

def paired_error_tests(ae_lstm, ae_xgb):
    """Shapiro-Wilk on d = AE_LSTM - AE_XGBoost, then paired t-test (normal) or Wilcoxon (not normal)."""
    ae_l = np.asarray(ae_lstm, dtype=float)
    ae_x = np.asarray(ae_xgb, dtype=float)
    d = ae_l - ae_x
    n = len(d)
    out = {"n": n, "d": d, "mean_d": float(np.mean(d)) if n else np.nan}
    if n < 3:
        out["error"] = "At least 3 paired months are needed for the statistical tests."
        return out
    try:
        sw = stats.shapiro(d)
        out["sw_W"], out["sw_p"] = float(sw.statistic), float(sw.pvalue)
    except Exception:
        out["sw_W"], out["sw_p"] = np.nan, np.nan
    try:
        tt = stats.ttest_rel(ae_l, ae_x)
        out["t"], out["t_p"] = float(tt.statistic), float(tt.pvalue)
    except Exception:
        out["t"], out["t_p"] = np.nan, np.nan
    try:
        wx = stats.wilcoxon(d)
        out["w"], out["w_p"] = float(wx.statistic), float(wx.pvalue)
    except Exception:
        out["w"], out["w_p"] = np.nan, np.nan
    out["normal"] = bool(np.isfinite(out["sw_p"]) and out["sw_p"] >= 0.05)
    if out["normal"]:
        out["chosen"], out["chosen_p"] = "Paired t-test", out["t_p"]
    else:
        out["chosen"], out["chosen_p"] = "Wilcoxon signed-rank", out["w_p"]
    return out

def tune_hyperparameters(y, dates, hours, mode, val_start, test_start, progress=None, status=None):
    """Chronological validation: train on [0:val_start], score on [val_start:test_start].
    The first config in each grid is the untuned baseline; the lowest validation MAE wins."""
    use_h = hours is not None
    tr_y, tr_d = y[:val_start], dates[:val_start]
    tr_h = hours[:val_start] if use_h else None
    val_dates = dates[val_start:test_start]
    val_actual = y[val_start:test_start]
    n_val = len(val_actual)
    if use_h:
        _, mm, ov = fill_monthly_hours(tr_h, tr_d)
        val_hours = future_hours_for_periods(val_dates, mm, ov)
    else:
        val_hours = None

    rows = []
    Xtr, ytr = make_xgb_training_data(tr_y, tr_d, mode, tr_h)
    best_x, best_x_mae, best_x_row = XGB_GRID[0], np.inf, None
    for k, cfg in enumerate(XGB_GRID):
        if status:
            status.info(f"Tuning XGBoost ({k + 1}/{len(XGB_GRID)})…")
        m = fit_xgb(Xtr, ytr, cfg)
        p = recursive_xgb_forecast(m, tr_y[-LOOKBACK:], mode, val_dates[0], n_val,
                                   tr_h[-LOOKBACK:] if use_h else None, val_hours)
        mae, rmse, mape = calculate_metrics(val_actual, p)
        rows.append({"Model": "XGBoost", "Setting": "Baseline" if k == 0 else f"Option {k}",
                     "Parameters": xgb_cfg_text(cfg), "Val MAE (kWh)": mae,
                     "Val RMSE (kWh)": rmse, "Val MAPE (%)": mape, "Selected": ""})
        if mae < best_x_mae:
            best_x_mae, best_x, best_x_row = mae, cfg, len(rows) - 1
    rows[best_x_row]["Selected"] = "✓"
    if progress:
        progress.progress(25)

    best_l, best_l_mae, best_l_row = LSTM_GRID[0], np.inf, None
    for k, cfg in enumerate(LSTM_GRID):
        if status:
            status.info(f"Tuning LSTM ({k + 1}/{len(LSTM_GRID)})…")
        m, sc, hs = train_lstm(tr_y, tr_d, tr_h, cfg)
        p = recursive_lstm_forecast(m, sc, tr_y[-LOOKBACK:], val_dates[0] - 1, n_val,
                                    tr_h[-LOOKBACK:] if use_h else None, val_hours, hs)
        mae, rmse, mape = calculate_metrics(val_actual, p)
        rows.append({"Model": "LSTM", "Setting": "Baseline" if k == 0 else f"Option {k}",
                     "Parameters": lstm_cfg_text(cfg), "Val MAE (kWh)": mae,
                     "Val RMSE (kWh)": rmse, "Val MAPE (%)": mape, "Selected": ""})
        if mae < best_l_mae:
            best_l_mae, best_l, best_l_row = mae, cfg, len(rows) - 1
    rows[best_l_row]["Selected"] = "✓"
    if progress:
        progress.progress(45)

    return best_x, best_l, pd.DataFrame(rows)

def read_2026_validation_file(file, kwh_col_index=11):
    """Read separate 2026 actuals without ever adding them to training data."""
    if file is None:
        return pd.DataFrame(columns=["Month", "Actual (kWh)"])

    month_map = {
        "jan": 1, "january": 1, "jan.": 1,
        "feb": 2, "february": 2, "feb.": 2,
        "mar": 3, "march": 3, "mar.": 3,
        "apr": 4, "april": 4, "apr.": 4,
        "may": 5, "jun": 6, "june": 6, "jun.": 6,
        "jul": 7, "july": 7, "jul.": 7,
        "aug": 8, "august": 8, "aug.": 8,
        "sep": 9, "sept": 9, "september": 9, "sep.": 9,
        "oct": 10, "october": 10, "oct.": 10,
        "nov": 11, "november": 11, "nov.": 11,
        "dec": 12, "december": 12, "dec.": 12
    }

    if file.name.lower().endswith(".csv"):
        frame = pd.read_csv(file)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        date_col = next((c for c in frame.columns if c in ["date", "month", "period"]), None)
        value_col = next((c for c in frame.columns if "consumption" in c or "kwh" in c or c == "actual"), None)
        if date_col is None or value_col is None:
            raise ValueError("For a CSV, use columns named Date and Consumption_kWh.")
        parsed_dates = pd.to_datetime(frame[date_col], errors="coerce").dt.to_period("M")
        values = pd.to_numeric(frame[value_col], errors="coerce")
        out = pd.DataFrame({"_period": parsed_dates, "Actual (kWh)": values}).dropna()
        out = out[out["_period"].apply(lambda p: p.year == 2026)]
    else:
        # Sheet '2026', month labels in column A, kWh in the column chosen in the sidebar.
        book = pd.ExcelFile(file)
        sheet = next((x for x in book.sheet_names if x.strip().lower() == "2026"), None)
        if sheet is None:
            raise ValueError("Excel validation file needs a sheet named '2026'.")
        frame = pd.read_excel(file, sheet_name=sheet, header=None)
        rows = []
        for _, row in frame.iterrows():
            label = str(row.iloc[0]).strip().lower().replace("\n", " ") if len(row) else ""
            key = label.split()[0] if label else ""
            month_num = month_map.get(key)
            if month_num and len(row) > kwh_col_index:
                value = pd.to_numeric(pd.Series([row.iloc[kwh_col_index]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    rows.append((pd.Period(year=2026, month=month_num, freq="M"), float(value)))
        out = pd.DataFrame(rows, columns=["_period", "Actual (kWh)"])

    if out.empty:
        raise ValueError("No 2026 actual monthly consumption values were found in the validation file.")
    duplicates = int(len(out) - out["_period"].nunique())
    out = out.sort_values("_period").drop_duplicates("_period", keep="first")
    out["Month"] = out["_period"].astype(str)
    result = out[["Month", "Actual (kWh)"]].reset_index(drop=True)
    result.attrs["duplicates"] = duplicates
    return result

# DATA PREPARATION
raw_df = None
data = None
dates = []
y = np.array([])
invalid_count = 0
missing_months = []
data_error = None
hours_available = False
hours = None
hours_col = None
hours_month_means, hours_overall_mean = {}, 0.0

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        raw_df.columns = [str(c).strip() for c in raw_df.columns]

        if raw_df.empty:
            raise ValueError("The uploaded file is empty.")

        cols = list(raw_df.columns)
        lower = {c: c.lower().strip() for c in cols}

        date_candidates = [
            c for c in cols
            if lower[c] in ["date", "month_date", "billing_date", "period"]
        ]
        year_candidates = [c for c in cols if lower[c] == "year"]
        month_candidates = [c for c in cols if lower[c] == "month"]

        numeric_candidates = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(raw_df[c])
            and c not in year_candidates
        ]

        preferred = [
            c for c in numeric_candidates
            if any(word in c.lower() for word in ["consumption", "kwh", "load"])
        ]

        with st.sidebar:
            st.markdown("**COLUMN MAPPING**")

            if date_candidates:
                date_mode = st.radio(
                    "Date format",
                    ["Date column", "Year + month"],
                    index=0 if not (year_candidates and month_candidates) else 1,
                    key="date_mode"
                )
            else:
                date_mode = st.radio(
                    "Date format",
                    ["Year + month", "Date column"],
                    index=0,
                    key="date_mode"
                )

            if date_mode == "Date column":
                date_col = st.selectbox(
                    "Date column",
                    cols,
                    index=cols.index(date_candidates[0]) if date_candidates else 0,
                    key="date_col"
                )
                year_col = None
                month_col = None
            else:
                year_col = st.selectbox(
                    "Year column",
                    cols,
                    index=cols.index(year_candidates[0]) if year_candidates else 0,
                    key="year_col"
                )
                month_col = st.selectbox(
                    "Month column",
                    cols,
                    index=cols.index(month_candidates[0])
                    if month_candidates else min(1, len(cols) - 1),
                    key="month_col"
                )
                date_col = None

            if not numeric_candidates:
                raise ValueError("No numeric consumption column was detected.")

            consume_default = (
                numeric_candidates.index(preferred[0]) if preferred else 0
            )

            consumption_col = st.selectbox(
                "Consumption (kWh)",
                numeric_candidates,
                index=consume_default,
                key="consumption_col"
            )

        data = raw_df.copy()

        if date_mode == "Date column":
            data["_Date"] = pd.to_datetime(
                data[date_col], errors="coerce"
            ).dt.to_period("M")

        else:
            month_names = {
                "jan": 1, "january": 1,
                "feb": 2, "february": 2,
                "mar": 3, "march": 3,
                "apr": 4, "april": 4,
                "may": 5,
                "jun": 6, "june": 6,
                "jul": 7, "july": 7,
                "aug": 8, "august": 8,
                "sep": 9, "sept": 9, "september": 9,
                "oct": 10, "october": 10,
                "nov": 11, "november": 11,
                "dec": 12, "december": 12
            }

            def convert_month(value):
                if pd.isna(value):
                    return np.nan

                value_string = str(value).strip().lower()

                if value_string in month_names:
                    return month_names[value_string]

                try:
                    number = int(float(value_string))
                    return number if 1 <= number <= 12 else np.nan
                except (ValueError, TypeError):
                    return np.nan

            years = pd.to_numeric(data[year_col], errors="coerce")
            months = data[month_col].apply(convert_month)

            data["_Date"] = [
                pd.Period(year=int(a), month=int(b), freq="M")
                if pd.notna(a) and pd.notna(b) and 1 <= int(b) <= 12
                else pd.NaT
                for a, b in zip(years, months)
            ]

        data["_Consumption"] = pd.to_numeric(
            data[consumption_col], errors="coerce"
        )
        hour_candidates = [c for c in data.columns if c != consumption_col and
                           any(term in str(c).lower().replace("_", " ")
                               for term in ["total hours", "total hrs", "operating hours", "operating hrs", "hours", "hrs"])]
        hours_col = hour_candidates[0] if hour_candidates else None
        data["_Hours"] = pd.to_numeric(data[hours_col], errors="coerce") if hours_col else np.nan

        invalid_count = int(
            data["_Date"].isna().sum()
            + data["_Consumption"].isna().sum()
        )

        data = (
            data.dropna(subset=["_Date", "_Consumption"])
            .sort_values("_Date")
            .groupby("_Date", as_index=False)
            .agg({"_Consumption": "mean", "_Hours": "mean"})
        )

        # Thesis rule: exclude the pandemic period and keep 2026 actuals
        # out of model training so they can be checked separately.
        before_filter_count = len(data)
        data = data[
            ~data["_Date"].apply(lambda p: p.year in (2020, 2021) or p.year >= 2026)
        ].copy()
        excluded_count = before_filter_count - len(data)

        if data.empty:
            raise ValueError("No valid date and consumption rows were found.")

        if (data["_Consumption"] < 0).any():
            raise ValueError(
                "Negative consumption values were found. "
                "Please verify the source data."
            )

        dates = data["_Date"].tolist()
        y = data["_Consumption"].astype(float).to_numpy()

        # Operating hours are an OPTIONAL experiment. Main study = consumption-only.
        hours_in_file = hours_col is not None and data["_Hours"].notna().any()
        if hours_in_file:
            hours_coverage = float(data["_Hours"].notna().mean())
            with st.sidebar:
                use_hours = st.checkbox(
                    "Use operating hours (optional experiment)",
                    value=False,
                    key="use_hours",
                    help="Thesis main study is consumption-only. Turn this on only for the extra hours experiment."
                )
            if use_hours:
                hours, hours_month_means, hours_overall_mean = fill_monthly_hours(
                    data["_Hours"].to_numpy(dtype=float), dates
                )
                hours_available = True
                st.success(f"Operating-hours column '{hours_col}' will be used as an extra input.")
                if hours_coverage < 0.8:
                    st.warning(
                        f"Hours exist for only {hours_coverage:.0%} of the months. Missing months are filled "
                        "with the average of the same calendar month, which can add noise."
                    )
            else:
                st.info(
                    f"Operating-hours column '{hours_col}' detected but not used. "
                    "Models use consumption history only (thesis main setup)."
                )
        else:
            st.info("No operating-hours column detected. Models use consumption history only.")

        if excluded_count > 0:
            st.info(
                f"Excluded {excluded_count} record(s) from training: "
                "pandemic-period months (2020–2021) and records from 2026 onward. "
                "2026 actual consumption is not used to train the models."
            )

        missing_months = [
            f"{a + 1} to {b - 1}"
            for a, b in zip(dates[:-1], dates[1:])
            if b.ordinal - a.ordinal > 1
        ]

        if len(y) < 24:
            raise ValueError(
                "At least 24 valid monthly observations are recommended."
            )

        if not np.isfinite(y).all():
            raise ValueError("Consumption contains invalid numeric values.")

    except Exception as error:
        data_error = str(error)

validation_actuals = pd.DataFrame(columns=["Month", "Actual (kWh)"])
validation_error = None
if validation_file is not None:
    try:
        validation_actuals = read_2026_validation_file(validation_file, int(kwh_col_num) - 1)
    except Exception as error:
        validation_error = str(error)


# PAGE HELPERS
def page_header(kicker, title, subtitle):
    st.markdown(
        f'<div class="eyebrow">{kicker}</div>'
        f'<h1 class="page-title">{title}</h1>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )

def chart_style(fig, height=390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=40, b=12),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color="#52647a"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#edf1f5", zeroline=False)
    )
    return fig

def fmt_p(value):
    return f"{value:.4f}" if np.isfinite(value) else "N/A"


# HOME PAGE
if page == "Home":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-label">Monthly energy intelligence · Research workspace</div>'
        '<div class="hero-title">Understand your consumption.<br>'
        'Forecast what comes next.</div>'
        '<div class="hero-copy">A research-focused workspace for exploring '
        'monthly electricity use and evaluating LSTM against XGBoost. Start '
        'with your historical data, then train both models and compare their results.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    page_header(
        "OVERVIEW / HOME",
        "Your forecasting workspace",
        "Upload your monthly consumption data in the sidebar to start. "
        "Your analysis stays organized across four simple sections."
    )

    if uploaded_file is None:
        left, right = st.columns([1.3, 1], gap="large")

        with left:
            st.markdown(
                '<div class="section-heading">Get started in three steps</div>'
                '<div class="section-copy">No special template required. '
                'Map your columns after uploading a CSV or Excel workbook.</div>',
                unsafe_allow_html=True
            )

            steps = [
                ("01", "Upload your data",
                 "Use monthly readings with a date or separate year and month columns."),
                ("02", "Train the models",
                 "Run LSTM and XGBoost on the same historical dataset, with optional validation tuning."),
                ("03", "Compare and export",
                 "Review MAE, RMSE, MAPE, the seasonal naive benchmark, paired statistical tests, and downloadable results.")
            ]

            for number, title, description in steps:
                with st.container(border=True):
                    c1, c2 = st.columns([0.18, 0.82])
                    c1.markdown(
                        f'<div class="brand-icon">{number}</div>',
                        unsafe_allow_html=True
                    )
                    c2.markdown(
                        f"**{title}**  \n"
                        f"<span style='color:#708096;font-size:.9rem'>"
                        f"{description}</span>",
                        unsafe_allow_html=True
                    )

        with right:
            st.markdown(
                '<div class="section-heading">Dataset checklist</div>'
                '<div class="section-copy">For best results, prepare a clean monthly timeline.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                "- One row per month\n"
                "- Consumption measured consistently in kWh\n"
                "- At least 24 monthly observations\n"
                "- Check for missing months and unusual values"
            )
            st.info("Example fields: `Year`, `Month`, `Consumption_kWh`.")

    elif data_error:
        st.error(f"The file could not be prepared: {data_error}")

    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Monthly records", f"{len(y):,}")
        c2.metric("Average use", f"{np.mean(y):,.0f} kWh")
        c3.metric("Latest reading", f"{y[-1]:,.0f} kWh")
        c4.metric("Coverage", f"{dates[0]} – {dates[-1]}")

        st.markdown(
            '<div class="section-heading">Consumption at a glance</div>'
            '<div class="section-copy">A quick view of the monthly history in the uploaded file.</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure(
            go.Scatter(
                x=[d.to_timestamp() for d in dates],
                y=y,
                mode="lines+markers",
                name="Actual consumption",
                line=dict(color="#0e9384", width=3),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(14,147,132,.08)"
            )
        )

        fig.update_layout(
            title="Historical monthly consumption",
            xaxis_title="Month",
            yaxis_title="Consumption (kWh)"
        )

        st.plotly_chart(chart_style(fig, 400), use_container_width=True)

        # Descriptive analysis (thesis objective 1): seasonality by calendar month.
        monthly_profile = (
            pd.DataFrame({"month": [d.month for d in dates], "kwh": y})
            .groupby("month")["kwh"].agg(["mean", "min", "max"]).reset_index()
        )
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_profile["label"] = monthly_profile["month"].apply(lambda m: month_labels[m - 1])
        st.markdown(
            '<div class="section-heading">Seasonal pattern</div>'
            '<div class="section-copy">Average consumption per calendar month, with the lowest and highest year shown as the range.</div>',
            unsafe_allow_html=True
        )
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=monthly_profile["label"], y=monthly_profile["mean"], name="Average",
            marker_color="#0e9384",
            error_y=dict(
                type="data", symmetric=False,
                array=(monthly_profile["max"] - monthly_profile["mean"]).to_numpy(),
                arrayminus=(monthly_profile["mean"] - monthly_profile["min"]).to_numpy()
            )
        ))
        fig_s.update_layout(xaxis_title="Calendar month", yaxis_title="Consumption (kWh)")
        st.plotly_chart(chart_style(fig_s, 340), use_container_width=True)

        if missing_months:
            st.warning(
                f"Timeline gaps detected: {len(missing_months)}. "
                "Model training excludes sequences that cross missing-month gaps."
            )

        if invalid_count:
            st.info(
                f"{invalid_count} invalid date/value entries were excluded. "
                "Check the source workbook to confirm."
            )

        with st.expander("Preview uploaded records"):
            preview = data.rename(
                columns={
                    "_Date": "Month",
                    "_Consumption": "Consumption (kWh)"
                }
            )[["Month", "Consumption (kWh)"]].head(15)

            preview["Month"] = preview["Month"].astype(str)

            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True
            )

# FORECAST PAGE
elif page == "Forecast":
    page_header(
        "WORKSPACE / FORECAST",
        "Generate a monthly forecast",
        "Choose a forecast horizon, train both models, and generate future "
        "consumption estimates from your uploaded history."
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file from the sidebar first.")
    elif data_error:
        st.error(f"The uploaded data needs attention: {data_error}")

    else:
        st.caption("Main thesis setup is consumption-only. Operating hours are used only if you switch them on in the sidebar; future hours are then estimated from the average of the same calendar month.")
        left, right = st.columns([1.25, 0.75], gap="large")

        with left:
            st.markdown(
                '<div class="section-heading">Forecast setup</div>'
                '<div class="section-copy">The models use the latest 12 '
                'consecutive months as their lookback window.</div>',
                unsafe_allow_html=True
            )

            horizon = st.selectbox(
                "Forecast horizon",
                options=list(range(1, 25)),
                format_func=lambda x: f"{x} month(s)"
            )

            feature_choice = st.radio(
                "XGBoost input features",
                ["Thesis spec: lag 1, 2, 3, 12 + month",
                 "Extended: lag 1–12 + rolling means + month"],
                key="feature_choice",
                help="The thesis table lists lag 1, 2, 3, and 12. Extended has more features and can overfit small data."
            )
            feature_mode = "paper" if feature_choice.startswith("Thesis") else "extended"

            tune_choice = st.checkbox(
                "Tune hyperparameters on a validation period",
                value=True,
                key="tune_choice",
                help="Chronological split: train → validation (12 months) → test (last 12 months). The baseline settings are compared against a few options, and the lowest validation MAE is kept."
            )

            st.caption(
                f"Forecast output: {horizon} future month(s) after {dates[-1]}."
            )

        with right:
            st.markdown(
                '<div class="section-heading">Data readiness</div>',
                unsafe_allow_html=True
            )
            st.metric("Valid monthly records", f"{len(y)}")
            st.metric("Latest available month", str(dates[-1]))
            st.metric("Missing timeline gaps", str(len(missing_months)))

        if len(y) - horizon < 18:
            st.error("There are not enough observations for the selected evaluation setup.")

        if st.button(
            "Train models & generate forecast",
            type="primary",
            use_container_width=True,
            disabled=(len(y) - horizon < 18)
        ):
            progress = st.progress(0)
            status = st.empty()

            try:
                latest_len = latest_contiguous_segment_length(dates)
                eval_h = min(12, max(1, latest_len - LOOKBACK))
                test_start = len(y) - eval_h
                use_h = bool(hours_available)
                h_all = hours if use_h else None

                if latest_len <= LOOKBACK or test_start <= LOOKBACK:
                    raise ValueError(
                        "Not enough consecutive monthly records before the "
                        "evaluation period. At least 13 consecutive training "
                        "months are needed."
                    )

                status.info("Preparing training, validation and test periods…")
                progress.progress(5)

                # Optional validation + tuning (thesis flowchart: validation → tuning)
                tune_ok = False
                tuning_df = pd.DataFrame()
                xgb_cfg, lstm_cfg = XGB_GRID[0], LSTM_GRID[0]
                val_start = test_start - eval_h
                if tune_choice:
                    if val_start > LOOKBACK and is_consecutive_window(dates, val_start - LOOKBACK, test_start - 1):
                        n_seq = len(make_xgb_training_data(
                            y[:val_start], dates[:val_start], feature_mode,
                            h_all[:val_start] if use_h else None)[0])
                        tune_ok = n_seq >= 24
                    if not tune_ok:
                        st.warning(
                            "Not enough consecutive history for a separate validation period. "
                            "Using the baseline settings without tuning."
                        )
                if tune_ok:
                    xgb_cfg, lstm_cfg, tuning_df = tune_hyperparameters(
                        y, dates, h_all, feature_mode, val_start, test_start, progress, status
                    )

                # Train on everything before the test period
                train_y = y[:test_start]
                train_dates = dates[:test_start]
                train_h = h_all[:test_start] if use_h else None
                Xtr, ytr = make_xgb_training_data(train_y, train_dates, feature_mode, train_h)

                if len(Xtr) < 5:
                    raise ValueError(
                        "Not enough valid XGBoost training sequences after "
                        "excluding timeline gaps."
                    )

                status.info("Training XGBoost on the training period…")
                progress.progress(50)
                xgb = fit_xgb(Xtr, ytr, xgb_cfg)

                status.info("Training LSTM on the training period…")
                progress.progress(58)
                lstm, scaler, hours_scale = train_lstm(train_y, train_dates, train_h, lstm_cfg)

                status.info("Evaluating on the held-out test months…")
                progress.progress(68)

                eval_dates = dates[test_start:]
                actual = y[test_start:]
                idxs = list(range(test_start, len(y)))

                # Same assumption as the real forecast: future hours = calendar-month average
                if use_h:
                    _, tr_mm, tr_ov = fill_monthly_hours(train_h, train_dates)
                    eval_hours = future_hours_for_periods(eval_dates, tr_mm, tr_ov)
                else:
                    eval_hours = None

                rec_x = recursive_xgb_forecast(
                    xgb, train_y[-LOOKBACK:], feature_mode, eval_dates[0], eval_h,
                    train_h[-LOOKBACK:] if use_h else None, eval_hours
                )
                rec_l = recursive_lstm_forecast(
                    lstm, scaler, train_y[-LOOKBACK:], eval_dates[0] - 1, eval_h,
                    train_h[-LOOKBACK:] if use_h else None, eval_hours, hours_scale
                )
                one_x = onestep_xgb(xgb, y, dates, h_all, feature_mode, idxs, eval_hours)
                one_l = onestep_lstm(lstm, scaler, y, dates, h_all, hours_scale, idxs)
                naive_eval = seasonal_naive(eval_dates, dict(zip(dates, y)))

                eval_sets = {}
                for name, pl, px in [(EVAL_RECURSIVE, rec_l, rec_x), (EVAL_ONESTEP, one_l, one_x)]:
                    df_eval = build_eval_df(eval_dates, actual, pl, px, naive_eval)
                    metrics_tbl, lm, xm, nm = eval_metric_bundle(df_eval)
                    eval_sets[name] = {"evaluation": df_eval, "metrics": metrics_tbl,
                                       "lm": lm, "xm": xm, "nm": nm}

                status.info("Retraining with all records and forecasting future months…")
                progress.progress(80)

                all_X, all_y = make_xgb_training_data(y, dates, feature_mode, h_all)

                if len(all_X) < 5:
                    raise ValueError(
                        "Not enough valid XGBoost training sequences in the full dataset."
                    )

                final_xgb = fit_xgb(all_X, all_y, xgb_cfg)
                final_lstm, final_scaler, final_hours_scale = train_lstm(y, dates, h_all, lstm_cfg)

                future_periods = [
                    dates[-1] + i for i in range(1, horizon + 1)
                ]

                if latest_contiguous_segment_length(dates) < LOOKBACK:
                    raise ValueError(
                        "The latest 12 months in the training dataset must be "
                        "consecutive. Please check for missing months before forecasting."
                    )

                history = y[-LOOKBACK:]

                future_hours = future_hours_for_periods(
                    future_periods, hours_month_means, hours_overall_mean
                ) if use_h else None
                xf = recursive_xgb_forecast(
                    final_xgb, history, feature_mode, future_periods[0], horizon,
                    h_all[-LOOKBACK:] if use_h else None, future_hours
                )
                lf = recursive_lstm_forecast(
                    final_lstm, final_scaler, history, dates[-1], horizon,
                    h_all[-LOOKBACK:] if use_h else None, future_hours, final_hours_scale
                )
                naive_future = seasonal_naive(future_periods, dict(zip(dates, y)))

                forecast = pd.DataFrame({
                    "Month": [str(p) for p in future_periods],
                    "LSTM Forecast (kWh)": lf,
                    "XGBoost Forecast (kWh)": xf,
                    "Seasonal Naive (kWh)": naive_future
                })

                st.session_state["eval_sets"] = eval_sets
                st.session_state["forecast_df"] = forecast
                st.session_state["tuning_df"] = tuning_df
                st.session_state["run_info"] = {
                    "feature_mode": feature_mode,
                    "use_hours": use_h,
                    "tuned": tune_ok,
                    "xgb_cfg": xgb_cfg_text(xgb_cfg),
                    "lstm_cfg": lstm_cfg_text(lstm_cfg),
                    "eval_months": eval_h,
                    "val_months": eval_h if tune_ok else 0,
                    "train_sequences": int(len(Xtr)),
                    "horizon": horizon
                }

                progress.progress(100)
                status.success(
                    "Training and forecasting completed. Open Model Comparison "
                    "to review evaluation results."
                )

            except Exception as error:
                status.error("The training process could not be completed.")
                st.error(str(error))

        if "forecast_df" in st.session_state:
            st.markdown(
                '<div class="section-heading">Latest forecast</div>'
                '<div class="section-copy">Forecasts are estimates generated '
                'from the uploaded historical records.</div>',
                unsafe_allow_html=True
            )

            forecast_df = st.session_state["forecast_df"]
            run_info = st.session_state.get("run_info", {})
            if run_info:
                st.caption(
                    f"Features: {'thesis spec (lag 1, 2, 3, 12 + month)' if run_info['feature_mode'] == 'paper' else 'extended'} · "
                    f"Hours: {'on' if run_info['use_hours'] else 'off'} · "
                    f"Tuning: {'validated' if run_info['tuned'] else 'baseline settings'} · "
                    f"Training sequences: {run_info['train_sequences']}"
                )

            a, b = st.columns(2)
            a.metric(
                "LSTM next-month estimate",
                f"{forecast_df.iloc[0, 1]:,.0f} kWh"
            )
            b.metric(
                "XGBoost next-month estimate",
                f"{forecast_df.iloc[0, 2]:,.0f} kWh"
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["LSTM Forecast (kWh)"],
                mode="lines+markers",
                name="LSTM",
                line=dict(color="#0e9384", width=3)
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["XGBoost Forecast (kWh)"],
                mode="lines+markers",
                name="XGBoost",
                line=dict(color="#345b88", width=3)
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Seasonal Naive (kWh)"],
                mode="lines+markers",
                name="Seasonal naive",
                line=dict(color="#9aa8b8", width=2, dash="dash")
            ))

            fig.update_layout(
                title="Forecast by model",
                xaxis_title="Future month",
                yaxis_title="Forecast consumption (kWh)"
            )

            st.plotly_chart(chart_style(fig), use_container_width=True)
            st.dataframe(
                forecast_df.round(2),
                use_container_width=True,
                hide_index=True
            )

            tuning_df = st.session_state.get("tuning_df", pd.DataFrame())
            if tuning_df is not None and not tuning_df.empty:
                with st.expander("Tuning log (validation period)"):
                    st.caption(
                        "The baseline is the untuned model. The row marked ✓ had the lowest validation MAE "
                        "and is the one used for testing and forecasting."
                    )
                    st.dataframe(tuning_df.round(3), use_container_width=True, hide_index=True)

            st.download_button(
                "Download forecast CSV",
                forecast_df.to_csv(index=False).encode("utf-8"),
                "monthly_load_forecast.csv",
                "text/csv",
                use_container_width=True
            )


# MODEL COMPARISON PAGE
elif page == "Model Comparison":
    page_header(
        "WORKSPACE / EVALUATION",
        "Compare model performance",
        "Review LSTM and XGBoost on the same held-out historical months. "
        "Lower MAE, RMSE, and MAPE values indicate smaller errors on this evaluation period."
    )

    if "eval_sets" not in st.session_state:
        st.info(
            "No evaluation results yet. Open Forecast, upload your data, "
            "and select **Train models & generate forecast**."
        )

    else:
        eval_sets = st.session_state["eval_sets"]
        forecast = st.session_state["forecast_df"]
        run_info = st.session_state.get("run_info", {})

        mode_name = st.radio(
            "Evaluation mode",
            list(eval_sets.keys()),
            horizontal=True,
            key="eval_mode",
            help="Recursive: the model feeds its own predictions back as history (true multi-month forecasting). "
                 "One-step-ahead: every month is predicted from actual past months."
        )
        res = eval_sets[mode_name]
        evaluation, metrics, lm, xm, nm = res["evaluation"], res["metrics"], res["lm"], res["xm"], res["nm"]

        if run_info:
            st.caption(
                f"XGBoost features: {'thesis spec (lag 1, 2, 3, 12 + month)' if run_info['feature_mode'] == 'paper' else 'extended'} · "
                f"Hours: {'on' if run_info['use_hours'] else 'off'} · "
                f"Tuning: {'validated (' + str(run_info['val_months']) + ' months)' if run_info['tuned'] else 'baseline settings'} · "
                f"Test months: {run_info['eval_months']} · XGBoost: {run_info['xgb_cfg']} · LSTM: {run_info['lstm_cfg']}"
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evaluation months", str(len(evaluation)))
        c2.metric("LSTM MAPE", f"{lm[2]:.2f}%" if np.isfinite(lm[2]) else "N/A")
        c3.metric("XGBoost MAPE", f"{xm[2]:.2f}%" if np.isfinite(xm[2]) else "N/A")
        c4.metric("Seasonal naive MAPE", f"{nm[2]:.2f}%" if np.isfinite(nm[2]) else "N/A")

        st.markdown(
            '<div class="section-heading">Evaluation scorecard</div>'
            '<div class="section-copy">Both models are evaluated on the same '
            'historical test period. Seasonal naive (same month last year) is only a reference benchmark, not a third model.</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            metrics.round(3),
            use_container_width=True,
            hide_index=True
        )

        metric_names = ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="LSTM", x=metric_names, y=lm, marker_color="#0e9384"))
        fig.add_trace(go.Bar(name="XGBoost", x=metric_names, y=xm, marker_color="#345b88"))
        fig.add_trace(go.Bar(name="Seasonal naive", x=metric_names, y=list(nm), marker_color="#b7c3d0"))

        fig.update_layout(
            title="Error metrics by model",
            barmode="group",
            yaxis_title="Metric value"
        )

        st.plotly_chart(chart_style(fig), use_container_width=True)

        wins_l = sum(
            np.isfinite(a) and np.isfinite(b)
            and a < b and not np.isclose(a, b)
            for a, b in zip(lm, xm)
        )

        wins_x = sum(
            np.isfinite(a) and np.isfinite(b)
            and b < a and not np.isclose(a, b)
            for a, b in zip(lm, xm)
        )

        if wins_l > wins_x:
            st.success(
                f"LSTM has lower error in {wins_l} of the 3 comparable "
                "metrics on this test period."
            )
        elif wins_x > wins_l:
            st.success(
                f"XGBoost has lower error in {wins_x} of the 3 comparable "
                "metrics on this test period."
            )
        else:
            st.info(
                "The metric results are mixed or similar. Review the scores "
                "and chart together."
            )

        if np.isfinite(nm[0]):
            beat_l = sum(a < b for a, b in zip(lm, nm) if np.isfinite(a) and np.isfinite(b))
            beat_x = sum(a < b for a, b in zip(xm, nm) if np.isfinite(a) and np.isfinite(b))
            st.caption(
                f"Against the seasonal naive benchmark: LSTM is lower in {beat_l} of 3 metrics, "
                f"XGBoost is lower in {beat_x} of 3. If a model does not beat the benchmark, "
                "the data (or the setup) is the bottleneck, not just the model."
            )

        st.markdown(
            '<div class="section-heading">Actual vs predicted</div>'
            '<div class="section-copy">This chart shows the test months '
            'used to calculate the evaluation metrics. If the model lines look flat while the actual line moves, '
            'try the one-step-ahead mode to see how much recursive error is hurting.</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=evaluation["Month"], y=evaluation["Actual (kWh)"],
            mode="lines+markers", name="Actual",
            line=dict(color="#142b4a", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=evaluation["Month"], y=evaluation["LSTM Prediction (kWh)"],
            mode="lines+markers", name="LSTM",
            line=dict(color="#0e9384", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=evaluation["Month"], y=evaluation["XGBoost Prediction (kWh)"],
            mode="lines+markers", name="XGBoost",
            line=dict(color="#345b88", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=evaluation["Month"], y=evaluation["Seasonal Naive (kWh)"],
            mode="lines+markers", name="Seasonal naive",
            line=dict(color="#9aa8b8", width=2, dash="dash")
        ))

        st.plotly_chart(chart_style(fig, 430), use_container_width=True)

        st.dataframe(
            evaluation.round(2),
            use_container_width=True,
            hide_index=True
        )
        st.caption("APE = absolute percentage error of one month. Check if one or two months with low actual values are pulling MAPE up.")

        # STATISTICAL TEST (thesis section 2.8)
        st.markdown(
            '<div class="section-heading">Statistical test on paired errors</div>'
            '<div class="section-copy">Per month, d = AE of LSTM − AE of XGBoost. Shapiro-Wilk checks if d is normal at α = 0.05. '
            'If normal, the paired t-test is used; if not, the Wilcoxon signed-rank test.</div>',
            unsafe_allow_html=True
        )
        tests = paired_error_tests(evaluation["LSTM AE (kWh)"], evaluation["XGBoost AE (kWh)"])
        if "error" in tests:
            st.info(tests["error"])
        else:
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Paired months (n)", str(tests["n"]))
            t2.metric("Mean d (kWh)", f"{tests['mean_d']:,.1f}")
            t3.metric("Shapiro-Wilk p", fmt_p(tests["sw_p"]))
            t4.metric(f"{tests['chosen']} p", fmt_p(tests["chosen_p"]))

            test_table = pd.DataFrame({
                "Test": ["Shapiro-Wilk (normality of d)", "Paired t-test", "Wilcoxon signed-rank"],
                "Statistic": [tests["sw_W"], tests["t"], tests["w"]],
                "p-value": [tests["sw_p"], tests["t_p"], tests["w_p"]],
                "Used for decision": ["", "✓" if tests["normal"] else "", "" if tests["normal"] else "✓"]
            })
            st.dataframe(test_table.round(4), use_container_width=True, hide_index=True)

            if np.isfinite(tests["chosen_p"]):
                if tests["chosen_p"] < 0.05:
                    lower = "LSTM" if tests["mean_d"] < 0 else "XGBoost"
                    st.success(
                        f"{tests['chosen']}: p = {tests['chosen_p']:.4f} < 0.05. Reject H₀. "
                        f"The difference in forecasting errors is statistically significant, and {lower} has the smaller mean absolute error."
                    )
                else:
                    st.info(
                        f"{tests['chosen']}: p = {tests['chosen_p']:.4f} ≥ 0.05. Fail to reject H₀. "
                        "There is not enough evidence of a significant difference between the two models' errors."
                    )
            else:
                st.warning("The decision test could not be computed (for example, all differences are zero).")

            (osm, osr), (slope, intercept, _r) = stats.probplot(tests["d"], dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="d",
                                        marker=dict(color="#0e9384", size=8)))
            fig_qq.add_trace(go.Scatter(x=[osm.min(), osm.max()],
                                        y=[slope * osm.min() + intercept, slope * osm.max() + intercept],
                                        mode="lines", name="Normal reference",
                                        line=dict(color="#9aa8b8", dash="dash")))
            fig_qq.update_layout(title="Q-Q plot of paired differences (d)",
                                 xaxis_title="Theoretical quantiles", yaxis_title="Ordered d (kWh)")
            st.plotly_chart(chart_style(fig_qq, 360), use_container_width=True)
            st.caption(
                f"Only {tests['n']} monthly pairs are available, so the tests have low power, and monthly errors "
                "can be correlated. Report this as a limitation and read the result together with MAE, RMSE, and MAPE."
            )

        # 2026 OUT-OF-SAMPLE VALIDATION
        st.markdown(
            '<div class="section-heading">2026 out-of-sample validation</div>'
            '<div class="section-copy">These actuals are uploaded separately and are not used to train either model. Only months present in both the forecast and actuals are compared.</div>',
            unsafe_allow_html=True
        )
        if validation_error:
            st.warning(f"Could not read the 2026 validation file: {validation_error}")
        elif validation_file is None:
            st.info("Upload the separate 2026 actuals workbook in the sidebar to calculate out-of-sample metrics.")
        else:
            dup_count = validation_actuals.attrs.get("duplicates", 0)
            if dup_count:
                st.warning(
                    f"{dup_count} duplicate month label(s) were found in the 2026 sheet (maybe a second table). "
                    "The first occurrence was used. Check that it is the right meter."
                )
            if len(y) >= 12 and len(validation_actuals) > 0:
                ratio = float(validation_actuals["Actual (kWh)"].mean()) / max(float(np.mean(y[-12:])), 1e-9)
                if ratio > 1.5 or ratio < 0.67:
                    st.warning(
                        f"The 2026 actuals average is {ratio:.2f}× the average of the last 12 training months. "
                        "This looks like a different meter/column or unit. Check the kWh column number in the sidebar."
                    )

            overlap = forecast.merge(validation_actuals, on="Month", how="inner")
            if overlap.empty:
                st.warning(
                    "No overlapping months. Make sure the forecast starts in January 2026 "
                    "and its horizon reaches the months with actual data."
                )
            else:
                val_lstm = calculate_metrics(overlap["Actual (kWh)"], overlap["LSTM Forecast (kWh)"])
                val_xgb = calculate_metrics(overlap["Actual (kWh)"], overlap["XGBoost Forecast (kWh)"])
                val_naive = naive_metrics(overlap["Actual (kWh)"], overlap["Seasonal Naive (kWh)"])
                val_metrics = pd.DataFrame({
                    "Metric": ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"],
                    "LSTM": list(val_lstm),
                    "XGBoost": list(val_xgb),
                    "Seasonal Naive": list(val_naive)
                })
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("2026 months checked", str(len(overlap)))
                v2.metric("2026 LSTM MAPE", f"{val_lstm[2]:.2f}%")
                v3.metric("2026 XGBoost MAPE", f"{val_xgb[2]:.2f}%")
                v4.metric("2026 naive MAPE", f"{val_naive[2]:.2f}%" if np.isfinite(val_naive[2]) else "N/A")
                st.dataframe(overlap.round(2), use_container_width=True, hide_index=True)
                st.dataframe(val_metrics.round(3), use_container_width=True, hide_index=True)

        with st.expander("Understand the evaluation metrics"):
            st.markdown(
                "**MAE (Mean Absolute Error)** measures the average absolute "
                "prediction error in kWh. **RMSE (Root Mean Squared Error)** "
                "is also in kWh and penalizes larger errors more strongly. "
                "**MAPE (Mean Absolute Percentage Error)** expresses average "
                "absolute error as a percentage of actual consumption and "
                "is not defined for zero actual values. For all three metrics, "
                "lower values generally indicate closer predictions on this "
                "test period. These scores do not guarantee future accuracy."
            )

        d1, d2, d3 = st.columns(3)

        d1.download_button(
            "Download metrics CSV",
            metrics.to_csv(index=False).encode("utf-8"),
            "model_evaluation_metrics.csv",
            "text/csv",
            use_container_width=True
        )

        d2.download_button(
            "Download actual vs predicted (with AE)",
            evaluation.to_csv(index=False).encode("utf-8"),
            "actual_vs_predicted.csv",
            "text/csv",
            use_container_width=True
        )

        d3.download_button(
            "Download forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            "monthly_load_forecast.csv",
            "text/csv",
            use_container_width=True
        )


# ABOUT PAGE
else:
    page_header(
        "PROJECT INFORMATION / ABOUT",
        "A transparent forecasting workflow",
        "This application is designed to support a research comparison of "
        "two machine-learning approaches for monthly electricity consumption."
    )

    st.markdown(
        '<div class="hero">'
        '<div class="hero-label">Research prototype</div>'
        '<div class="hero-title">A clearer view of model performance.</div>'
        '<div class="hero-copy">The app combines a consistent data preparation '
        'workflow, two forecasting models, and a shared historical evaluation '
        'period so their errors can be compared side by side.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### LSTM")
            st.write(
                "Long Short-Term Memory is a recurrent neural network designed "
                "to learn sequential patterns from previous monthly consumption values."
            )
            st.caption("Input: previous 12 consecutive months + month of year")

        with st.container(border=True):
            st.markdown("### Evaluation metrics")
            st.write(
                "MAE and RMSE are expressed in kWh. MAPE is expressed as a "
                "percentage and excludes zero-valued actual observations. "
                "A seasonal naive forecast (same month last year) is shown as a reference benchmark."
            )

    with right:
        with st.container(border=True):
            st.markdown("### XGBoost")
            st.write(
                "Extreme Gradient Boosting learns from lagged consumption "
                "values and calendar month features."
            )
            st.caption("Input: lag 1, 2, 3, 12 + monthly seasonality (extended option available)")

        with st.container(border=True):
            st.markdown("### Statistical comparison")
            st.write(
                "Absolute errors of the two models are paired per month. Shapiro-Wilk checks "
                "normality of the differences, then a paired t-test (or Wilcoxon signed-rank test "
                "if not normal) is applied at α = 0.05."
            )

    with st.container(border=True):
        st.markdown("### Important limitations")
        st.write(
            "Forecasts depend on the uploaded dataset and the assumptions "
            "learned by each model. Historical evaluation scores describe "
            "the selected test period only; they do not guarantee future accuracy. "
            "With about 12 test months, statistical tests have low power."
        )

    st.markdown(
        '<div class="section-heading">Data handling</div>'
        '<div class="section-copy">The app accepts CSV and Excel files, maps '
        'date and consumption columns, sorts records chronologically, excludes '
        'invalid rows and the pandemic years 2020–2021, and avoids training sequences that cross missing calendar '
        'months. Data is split chronologically (training → validation → test) with no shuffling. '
        'Duplicate months are averaged during preparation, so review '
        'your source data before interpreting results.</div>',
        unsafe_allow_html=True
    )


st.divider()

st.markdown(
    '<div class="note">LOAD FORECAST STUDIO · Monthly electrical load forecasting '
    '· Academic research prototype</div>',
    unsafe_allow_html=True
)
