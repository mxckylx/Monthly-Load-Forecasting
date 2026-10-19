import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
st.set_page_config(page_title="Monthly Load Forecasting", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:#f6f8fb;}
.block-container {max-width:1250px;padding-top:2rem;padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e4eaf1;}
[data-testid="stMetric"] {background:white;border:1px solid #e4eaf1;border-radius:14px;padding:16px;}
.hero {padding:26px 28px;border-radius:18px;background:linear-gradient(120deg,#142b4a,#176b70);color:white;margin-bottom:22px;}
.hero h1 {font-family:Manrope,sans-serif;color:white;font-size:2rem;margin:0 0 8px 0;}
.muted {color:#718096;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

LOOKBACK = 12
MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
MONTH_MAP = {name.lower(): num for num, name in MONTH_NAMES.items()}
MONTH_MAP.update({name[:3].lower(): num for num, name in MONTH_NAMES.items()})

with st.sidebar:
    st.markdown("## ⚡ Monthly Load Forecasting")
    page = st.radio("MENU", ["Home", "Forecast", "Model Comparison"])
    st.divider()
    uploaded_file = st.file_uploader("Upload monthly data", type=["csv", "xlsx", "xls"])
    st.caption("Required: Year, Month, Consumption_kWh. Operating_Hours is needed to use operating hours in the models.")


def parse_month(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in MONTH_MAP:
        return MONTH_MAP[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 12 else np.nan
    except (ValueError, TypeError):
        dt = pd.to_datetime(s, errors="coerce")
        return int(dt.month) if pd.notna(dt) else np.nan


def prepare_data(file):
    if file is None:
        return None, None
    file.seek(0)
    frame = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    frame.columns = [str(c).strip() for c in frame.columns]
    lookup = {c.lower().strip(): c for c in frame.columns}
    year_col = lookup.get("year")
    month_col = lookup.get("month")
    cons_col = next((lookup[k] for k in ["consumption_kwh", "consumption (kwh)", "electricity consumption (kwh)", "consumption", "kwh"] if k in lookup), None)
    hours_col = next((lookup[k] for k in ["operating_hours", "operating hours", "total hours", "total_hrs", "hours"] if k in lookup), None)
    date_col = next((lookup[k] for k in ["date", "period", "month_date", "billing_date"] if k in lookup), None)
    if cons_col is None:
        cons_col = next((c for c in frame.columns if "consumption" in c.lower() or "kwh" in c.lower()), None)
    if cons_col is None:
        raise ValueError("Cannot find the consumption column. Name it Consumption_kWh.")
    if year_col and month_col:
        years = pd.to_numeric(frame[year_col], errors="coerce")
        months = frame[month_col].apply(parse_month)
        dates = [pd.Period(year=int(y), month=int(m), freq="M") if pd.notna(y) and pd.notna(m) else pd.NaT for y, m in zip(years, months)]
    elif date_col:
        dates = pd.to_datetime(frame[date_col], errors="coerce").dt.to_period("M")
    else:
        raise ValueError("Include Year and Month columns, or one Date column.")
    out = pd.DataFrame({"Period": dates, "Consumption_kWh": pd.to_numeric(frame[cons_col], errors="coerce")})
    if hours_col:
        out["Operating_Hours"] = pd.to_numeric(frame[hours_col], errors="coerce")
    else:
        out["Operating_Hours"] = np.nan
    out = out.dropna(subset=["Period", "Consumption_kWh"])
    out = out[out["Period"].apply(lambda p: p.year not in (2020, 2021) and p.year < 2026)].copy()
    out = out.groupby("Period", as_index=False).agg({"Consumption_kWh":"mean", "Operating_Hours":"mean"}).sort_values("Period")
    if len(out) < 24:
        raise ValueError("Please upload at least 24 valid monthly records for training.")
    if (out["Consumption_kWh"] < 0).any():
        raise ValueError("Consumption cannot be negative.")
    return out.reset_index(drop=True), hours_col is not None


def period_consecutive(periods, start, end):
    return all(periods[i].ordinal - periods[i-1].ordinal == 1 for i in range(start+1, end+1))


def month_features(period):
    angle = 2 * np.pi * period.month / 12
    return [np.sin(angle), np.cos(angle)]


def xgb_features(values, hours, periods, i, future_hours=None):
    vals = np.asarray(values[max(0, i-LOOKBACK):i], dtype=float)
    if len(vals) != LOOKBACK:
        raise ValueError("Need 12 consecutive months before the prediction period.")
    h = np.asarray(hours[max(0, i-LOOKBACK):i], dtype=float)
    h = np.nan_to_num(h, nan=float(np.nanmean(hours)) if np.isfinite(hours).any() else 0.0)
    target_period = periods[i] if i < len(periods) else periods[-1] + 1
    next_hours = float(future_hours if future_hours is not None else 0.0)
    return list(vals[::-1]) + [float(vals[-3:].mean()), float(vals[-6:].mean()), float(vals.mean()), *month_features(target_period), next_hours, float(h[-1]), float(np.mean(h[-3:])), float(np.mean(h[-6:])), float(np.mean(h))]


def make_xgb_data(values, hours, periods):
    X, y = [], []
    for i in range(LOOKBACK, len(values)):
        if period_consecutive(periods, i-LOOKBACK, i):
            X.append(xgb_features(values, hours, periods, i, hours[i]))
            y.append(values[i])
    if not X:
        raise ValueError("Not enough consecutive months to train XGBoost.")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def train_xgb(values, hours, periods):
    X, target = make_xgb_data(values, hours, periods)
    model = XGBRegressor(n_estimators=250, max_depth=2, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
    model.fit(X, target)
    return model


def train_lstm(values, hours, periods):
    values = np.asarray(values, dtype=float)
    hours = np.asarray(hours, dtype=float)
    hours = np.nan_to_num(hours, nan=float(np.nanmean(hours)) if np.isfinite(hours).any() else 0.0)
    y_scaler = MinMaxScaler()
    h_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(values.reshape(-1,1)).ravel()
    h_scaled = h_scaler.fit_transform(hours.reshape(-1,1)).ravel()
    X, target = [], []
    for i in range(LOOKBACK, len(values)):
        if period_consecutive(periods, i-LOOKBACK, i):
            seq = [[y_scaled[j], *month_features(periods[j]), h_scaled[j]] for j in range(i-LOOKBACK, i)]
            X.append(seq)
            target.append(y_scaled[i])
    if len(X) < 8:
        raise ValueError("Not enough consecutive monthly records to train LSTM. Use at least 24 months without gaps.")
    tf.keras.utils.set_random_seed(42)
    model = Sequential([Input(shape=(LOOKBACK, 4)), LSTM(32, activation="tanh"), Dense(16, activation="relu"), Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.fit(np.asarray(X, dtype=float), np.asarray(target), epochs=80, batch_size=8, verbose=0, shuffle=False, callbacks=[EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)])
    return model, y_scaler, h_scaler


def predict_lstm(model, y_scaler, h_scaler, values, hours, periods, future_hours):
    values = list(map(float, values))
    hours = list(map(float, hours))
    preds = []
    for step, fh in enumerate(future_hours):
        target_period = periods[-1] + (step + 1)
        y_scaled = y_scaler.transform(np.asarray(values[-LOOKBACK:]).reshape(-1,1)).ravel()
        hs = np.asarray(hours[-LOOKBACK:], dtype=float)
        hs = np.nan_to_num(hs, nan=float(np.nanmean(hours)) if np.isfinite(hours).any() else 0.0)
        h_scaled = h_scaler.transform(hs.reshape(-1,1)).ravel()
        recent_periods = [target_period - (LOOKBACK-j) for j in range(LOOKBACK)]
        seq = [[y_scaled[j], *month_features(recent_periods[j]), h_scaled[j]] for j in range(LOOKBACK)]
        scaled_pred = float(model.predict(np.asarray(seq).reshape(1, LOOKBACK, 4), verbose=0)[0][0])
        pred = max(0.0, float(y_scaler.inverse_transform([[scaled_pred]])[0][0]))
        preds.append(pred)
        values.append(pred)
        hours.append(float(fh))
    return np.asarray(preds)


def predict_xgb(model, values, hours, periods, future_hours):
    vals = list(map(float, values))
    hrs = list(map(float, hours))
    preds = []
    for step, fh in enumerate(future_hours):
        target_period = periods[-1] + (step + 1)
        hist_vals = np.asarray(vals[-LOOKBACK:], dtype=float)
        hist_hours = np.asarray(hrs[-LOOKBACK:], dtype=float)
        hist_hours = np.nan_to_num(hist_hours, nan=float(np.nanmean(hrs)) if np.isfinite(hrs).any() else 0.0)
        f = list(hist_vals[::-1]) + [float(hist_vals[-3:].mean()), float(hist_vals[-6:].mean()), float(hist_vals.mean()), *month_features(target_period), float(fh), float(hist_hours[-1]), float(hist_hours[-3:].mean()), float(hist_hours[-6:].mean()), float(hist_hours.mean())]
        pred = max(0.0, float(model.predict(np.asarray([f], dtype=float))[0]))
        preds.append(pred)
        vals.append(pred)
        hrs.append(float(fh))
    return np.asarray(preds)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nz = actual != 0
    mape = float(np.mean(np.abs((actual[nz]-predicted[nz])/actual[nz]))*100) if nz.any() else np.nan
    return mae, rmse, mape


def get_history(df):
    return df["Consumption_kWh"].to_numpy(dtype=float), df["Operating_Hours"].to_numpy(dtype=float), df["Period"].tolist()


def future_hours_inputs(last_period, horizon, default_hours):
    st.markdown("### Expected operating hours")
    st.caption("Enter the planned operating hours for each forecast month. These values are used by both LSTM and XGBoost.")
    cols = st.columns(3)
    hours = []
    for i in range(horizon):
        p = last_period + (i+1)
        with cols[i % 3]:
            value = st.number_input(f"{MONTH_NAMES[p.month]} {p.year}", min_value=0.0, max_value=744.0, value=float(default_hours), step=12.0, key=f"future_hours_{p.year}_{p.month}")
            hours.append(value)
    return hours


def model_comparison(df):
    values, hours, periods = get_history(df)
    # Use last 12 months as holdout when available; models only train on earlier data.
    test_size = 12
    split = len(df) - test_size
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    if len(train_df) < 24 or len(test_df) < 1:
        st.error("Not enough data for a holdout comparison. Please upload at least 25 monthly records.")
        return
    tr_y, tr_h, tr_p = get_history(train_df)
    te_y, te_h, te_p = get_history(test_df)
    if not all(te_p[i].ordinal-te_p[i-1].ordinal == 1 for i in range(1,len(te_p))):
        st.warning("The last test period has missing months. Metrics may not represent a complete year.")
    try:
        with st.spinner("Training and comparing LSTM and XGBoost..."):
            xgb = train_xgb(tr_y, tr_h, tr_p)
            lstm, ys, hs = train_lstm(tr_y, tr_h, tr_p)
            # Forecast test period recursively; use known actual operating hours for those test months.
            pred_xgb = predict_xgb(xgb, tr_y, tr_h, tr_p, te_h)
            pred_lstm = predict_lstm(lstm, ys, hs, tr_y, tr_h, tr_p, te_h)
        rows=[]
        for name, pred in [("LSTM", pred_lstm), ("XGBoost", pred_xgb)]:
            mae, rmse, mape = metrics(te_y, pred)
            rows.append({"Model":name, "MAE (kWh)":mae, "RMSE (kWh)":rmse, "MAPE (%)":mape})
        result = pd.DataFrame(rows)
        st.caption(f"Evaluation: {te_p[0]} to {te_p[-1]}. The models were trained only on earlier months; test-period operating hours are used as known inputs.")
        st.dataframe(result.round(2), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[str(p) for p in te_p], y=te_y, mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=[str(p) for p in te_p], y=pred_lstm, mode="lines+markers", name="LSTM"))
        fig.add_trace(go.Scatter(x=[str(p) for p in te_p], y=pred_xgb, mode="lines+markers", name="XGBoost"))
        fig.update_layout(title="Actual vs Predicted Consumption", xaxis_title="Month", yaxis_title="Consumption (kWh)", height=420, template="plotly_white", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download metrics (CSV)", result.to_csv(index=False).encode("utf-8"), "model_metrics.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not compare the models: {e}")


# Main UI
if page == "Home":
    st.markdown('<div class="hero"><h1>Monthly Electricity Load Forecasting</h1><div>Compare LSTM and XGBoost using monthly consumption and operating hours.</div></div>', unsafe_allow_html=True)
    st.write("Upload your monthly dataset, forecast future electricity consumption, and compare model accuracy using only MAE, RMSE, and MAPE.")
    st.markdown("**Required columns**")
    st.code("Year, Month, Consumption_kWh, Operating_Hours", language="text")
    st.caption("Use one row per month. Pandemic years 2020–2021 and records from 2026 onward are excluded from training.")
    if uploaded_file is not None:
        try:
            df, has_hours = prepare_data(uploaded_file)
            st.success(f"Loaded {len(df)} monthly records.")
            a,b,c = st.columns(3)
            a.metric("Monthly records", len(df))
            a2 = int(df["Period"].iloc[0].year); b2 = int(df["Period"].iloc[-1].year)
            b.metric("Data range", f"{a2}–{b2}")
            c.metric("Operating hours", "Available" if has_hours else "Missing")
            st.dataframe(pd.DataFrame({"Year":[p.year for p in df.Period],"Month":[MONTH_NAMES[p.month] for p in df.Period],"Consumption_kWh":df.Consumption_kWh,"Operating_Hours":df.Operating_Hours}), use_container_width=True, hide_index=True)
            if not has_hours:
                st.warning("No Operating_Hours column was found. Add it to your file to use operating hours as model inputs.")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload your CSV or Excel file in the sidebar to begin.")

elif page == "Forecast":
    st.title("Forecast Future Consumption")
    if uploaded_file is None:
        st.info("Upload your historical monthly dataset in the sidebar first.")
    else:
        try:
            df, has_hours = prepare_data(uploaded_file)
            if not has_hours or df["Operating_Hours"].isna().all():
                st.warning("Operating hours are missing. Add an Operating_Hours column before forecasting with this version of the app.")
            else:
                st.caption(f"Training data: {df.Period.iloc[0]} to {df.Period.iloc[-1]}. Both models use historical consumption and operating hours.")
                horizon = st.selectbox("Forecast horizon", [1,3,6,12], index=1, format_func=lambda n:f"{n} month(s)")
                known_hours = df["Operating_Hours"].dropna()
                default_hours = float(known_hours.iloc[-1]) if len(known_hours) else 480.0
                future_hours = future_hours_inputs(df["Period"].iloc[-1], horizon, default_hours)
                if st.button("Generate Forecast", type="primary", use_container_width=True):
                    with st.spinner("Training LSTM and XGBoost and generating forecast..."):
                        values, hours, periods = get_history(df)
                        xgb = train_xgb(values, hours, periods)
                        lstm, ys, hs = train_lstm(values, hours, periods)
                        pred_xgb = predict_xgb(xgb, values, hours, periods, future_hours)
                        pred_lstm = predict_lstm(lstm, ys, hs, values, hours, periods, future_hours)
                    future_periods = [periods[-1]+(i+1) for i in range(horizon)]
                    output = pd.DataFrame({"Year":[p.year for p in future_periods],"Month":[MONTH_NAMES[p.month] for p in future_periods],"Operating_Hours":future_hours,"LSTM_Forecast_kWh":np.round(pred_lstm,2),"XGBoost_Forecast_kWh":np.round(pred_xgb,2)})
                    st.session_state["forecast_output"] = output
                    st.session_state["forecast_periods"] = [str(p) for p in future_periods]
                    st.session_state["forecast_lstm"] = pred_lstm
                    st.session_state["forecast_xgb"] = pred_xgb
                if "forecast_output" in st.session_state:
                    output = st.session_state["forecast_output"]
                    st.subheader("Forecast results")
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=output["Month"]+" "+output["Year"].astype(str), y=output["LSTM_Forecast_kWh"], mode="lines+markers", name="LSTM"))
                    fig.add_trace(go.Scatter(x=output["Month"]+" "+output["Year"].astype(str), y=output["XGBoost_Forecast_kWh"], mode="lines+markers", name="XGBoost"))
                    fig.update_layout(title="Forecasted Monthly Electricity Consumption", xaxis_title="Month", yaxis_title="Consumption (kWh)", height=400, template="plotly_white", legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button("Download forecast (CSV)", output.to_csv(index=False).encode("utf-8"), "monthly_load_forecast.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not prepare the data: {e}")

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.write("This page evaluates LSTM and XGBoost on the latest available historical months. Lower error values indicate better performance.")
    if uploaded_file is None:
        st.info("Upload your historical monthly dataset in the sidebar first.")
    else:
        try:
            df, has_hours = prepare_data(uploaded_file)
            if not has_hours or df["Operating_Hours"].isna().all():
                st.warning("Operating hours are missing. Add an Operating_Hours column to evaluate both models with this input.")
            elif st.button("Compare Models", type="primary"):
                model_comparison(df)
        except Exception as e:
            st.error(f"Could not prepare the data: {e}")
