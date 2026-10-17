import io
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
        "Upload monthly data",
        type=["xlsx", "xls", "csv"],
        help="CSV or Excel with Year, Month, Consumption_kWh, or a date column."
    )

    st.caption("Recommended columns: Year · Month · Consumption_kWh")

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

def xgb_features(history, target_period):
    h = np.asarray(history, dtype=float)

    if len(h) < LOOKBACK:
        raise ValueError("At least 12 consecutive monthly values are needed.")

    features = [h[-lag] for lag in range(1, LOOKBACK + 1)]
    features += [
        float(np.mean(h[-3:])),
        float(np.mean(h[-6:])),
        float(np.mean(h[-12:]))
    ]
    features += [
        np.sin(2 * np.pi * target_period.month / 12),
        np.cos(2 * np.pi * target_period.month / 12)
    ]

    return features

def make_xgb_training_data(values, periods):
    X, target = [], []

    for i in range(LOOKBACK, len(values)):
        if is_consecutive_window(periods, i - LOOKBACK, i):
            X.append(xgb_features(values[i - LOOKBACK:i], periods[i]))
            target.append(values[i])

    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)

def recursive_xgb_forecast(model, history, start_period, steps):
    history = list(np.asarray(history, dtype=float))
    predictions = []

    for step in range(steps):
        prediction = max(
            0.0,
            float(model.predict(np.asarray([
                xgb_features(history, start_period + step)
            ]))[0])
        )
        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions)

def train_lstm(training_values, training_periods):
    training_values = np.asarray(training_values, dtype=float)
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        training_values.reshape(-1, 1)
    ).flatten()

    X, target = [], []

    for i in range(LOOKBACK, len(scaled)):
        if is_consecutive_window(training_periods, i - LOOKBACK, i):
            X.append(scaled[i - LOOKBACK:i])
            target.append(scaled[i])

    if len(X) < 5:
        raise ValueError(
            "Not enough consecutive monthly sequences to train LSTM. "
            "Check the date range and missing months."
        )

    X = np.asarray(X).reshape(-1, LOOKBACK, 1)
    target = np.asarray(target)

    tf.keras.utils.set_random_seed(42)

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(
        X,
        target,
        epochs=40,
        batch_size=8,
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="loss",
                patience=5,
                restore_best_weights=True
            )
        ],
        shuffle=False
    )

    return model, scaler

def recursive_lstm_forecast(model, scaler, history, steps):
    history = np.asarray(history, dtype=float)

    if len(history) < LOOKBACK:
        raise ValueError("At least 12 history values are required.")

    scaled_history = scaler.transform(
        history.reshape(-1, 1)
    ).flatten().tolist()

    predictions = []

    for _ in range(steps):
        sequence = np.asarray(
            scaled_history[-LOOKBACK:]
        ).reshape(1, LOOKBACK, 1)

        scaled_prediction = float(
            model.predict(sequence, verbose=0)[0][0]
        )

        prediction = max(
            0.0,
            float(
                scaler.inverse_transform(
                    [[scaled_prediction]]
                )[0][0]
            )
        )

        predictions.append(prediction)
        scaled_history.append(
            float(scaler.transform([[prediction]])[0][0])
        )

    return np.asarray(predictions)

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
    # DATA PREPARATION
raw_df = None
data = None
dates = []
y = np.array([])
invalid_count = 0
missing_months = []
data_error = None

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

        invalid_count = int(
            data["_Date"].isna().sum()
            + data["_Consumption"].isna().sum()
        )

        data = (
            data.dropna(subset=["_Date", "_Consumption"])
            .sort_values("_Date")
            .groupby("_Date", as_index=False)
            .agg({"_Consumption": "mean"})
        )

        if data.empty:
            raise ValueError("No valid date and consumption rows were found.")

        if (data["_Consumption"] < 0).any():
            raise ValueError(
                "Negative consumption values were found. "
                "Please verify the source data."
            )

        dates = data["_Date"].tolist()
        y = data["_Consumption"].astype(float).to_numpy()

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
                 "Run LSTM and XGBoost on the same historical dataset."),
                ("03", "Compare and export",
                 "Review MAE, RMSE, MAPE, charts, and downloadable results.")
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
    options=list(range(1, 13)),
    format_func=lambda x: f"{x} month(s)"
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
                split = len(y) - eval_h

                train_y = y[:split]
                train_dates = dates[:split]

                if latest_len <= LOOKBACK or len(train_y) <= LOOKBACK:
                    raise ValueError(
                        "Not enough consecutive monthly records before the "
                        "evaluation period. At least 13 consecutive training "
                        "months are needed."
                    )

                status.info("Preparing training and evaluation data…")
                progress.progress(10)

                Xtr, ytr = make_xgb_training_data(train_y, train_dates)

                if len(Xtr) < 5:
                    raise ValueError(
                        "Not enough valid XGBoost training sequences after "
                        "excluding timeline gaps."
                    )

                status.info("Training XGBoost…")
                progress.progress(25)

                xgb = XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1
                )
                xgb.fit(Xtr, ytr)

                status.info("Training LSTM…")
                progress.progress(45)

                lstm, scaler = train_lstm(train_y, train_dates)

                status.info("Evaluating both models on held-out months…")
                progress.progress(65)

                eval_dates = dates[split:]
                actual = y[split:]

                xgb_test = recursive_xgb_forecast(
                    xgb, train_y[-LOOKBACK:], eval_dates[0], eval_h
                )

                lstm_test = recursive_lstm_forecast(
                    lstm, scaler, train_y[-LOOKBACK:], eval_h
                )

                xm = calculate_metrics(actual, xgb_test)
                lm = calculate_metrics(actual, lstm_test)

                metrics = pd.DataFrame({
                    "Metric": ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"],
                    "LSTM": list(lm),
                    "XGBoost": list(xm)
                })

                status.info(
                    "Retraining with all records and forecasting future months…"
                )
                progress.progress(80)

                all_X, all_y = make_xgb_training_data(y, dates)

                if len(all_X) < 5:
                    raise ValueError(
                        "Not enough valid XGBoost training sequences in the full dataset."
                    )

                final_xgb = XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1
                )
                final_xgb.fit(all_X, all_y)

                final_lstm, final_scaler = train_lstm(y, dates)

                future_periods = [
                    dates[-1] + i for i in range(1, horizon + 1)
                ]
                history = y[-LOOKBACK:]

                xf = recursive_xgb_forecast(
                    final_xgb, history, future_periods[0], horizon
                )

                lf = recursive_lstm_forecast(
                    final_lstm, final_scaler, history, horizon
                )

                forecast = pd.DataFrame({
                    "Month": [str(p) for p in future_periods],
                    "LSTM Forecast (kWh)": lf,
                    "XGBoost Forecast (kWh)": xf
                })

                evaluation = pd.DataFrame({
                    "Month": [str(d) for d in eval_dates],
                    "Actual (kWh)": actual,
                    "LSTM Prediction (kWh)": lstm_test,
                    "XGBoost Prediction (kWh)": xgb_test
                })

                st.session_state["metrics_df"] = metrics
                st.session_state["evaluation_df"] = evaluation
                st.session_state["forecast_df"] = forecast
                st.session_state["lstm_metrics"] = lm
                st.session_state["xgb_metrics"] = xm
                st.session_state["horizon"] = horizon
                st.session_state["evaluation_horizon"] = eval_h

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

    if "metrics_df" not in st.session_state:
        st.info(
            "No evaluation results yet. Open Forecast, upload your data, "
            "and select **Train models & generate forecast**."
        )

    else:
        metrics = st.session_state["metrics_df"]
        evaluation = st.session_state["evaluation_df"]
        forecast = st.session_state["forecast_df"]

        lm = st.session_state["lstm_metrics"]
        xm = st.session_state["xgb_metrics"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluation months", str(len(evaluation)))
        c2.metric(
            "LSTM MAPE",
            f"{lm[2]:.2f}%" if np.isfinite(lm[2]) else "N/A"
        )
        c3.metric(
            "XGBoost MAPE",
            f"{xm[2]:.2f}%" if np.isfinite(xm[2]) else "N/A"
        )

        st.markdown(
            '<div class="section-heading">Evaluation scorecard</div>'
            '<div class="section-copy">Both models are evaluated on the same '
            'historical holdout period.</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            metrics.round(3),
            use_container_width=True,
            hide_index=True
        )

        metric_names = ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="LSTM",
            x=metric_names,
            y=lm,
            marker_color="#0e9384"
        ))
        fig.add_trace(go.Bar(
            name="XGBoost",
            x=metric_names,
            y=xm,
            marker_color="#345b88"
        ))

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

        st.markdown(
            '<div class="section-heading">Actual vs predicted</div>'
            '<div class="section-copy">This chart shows the holdout months '
            'used to calculate the evaluation metrics.</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=evaluation["Month"],
            y=evaluation["Actual (kWh)"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#142b4a", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=evaluation["Month"],
            y=evaluation["LSTM Prediction (kWh)"],
            mode="lines+markers",
            name="LSTM",
            line=dict(color="#0e9384", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=evaluation["Month"],
            y=evaluation["XGBoost Prediction (kWh)"],
            mode="lines+markers",
            name="XGBoost",
            line=dict(color="#345b88", width=2)
        ))

        st.plotly_chart(chart_style(fig, 430), use_container_width=True)

        st.dataframe(
            evaluation.round(2),
            use_container_width=True,
            hide_index=True
        )

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
            "Download actual vs predicted",
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
            st.caption("Input: previous 12 consecutive months")

        with st.container(border=True):
            st.markdown("### Evaluation metrics")
            st.write(
                "MAE and RMSE are expressed in kWh. MAPE is expressed as a "
                "percentage and excludes zero-valued actual observations."
            )

    with right:
        with st.container(border=True):
            st.markdown("### XGBoost")
            st.write(
                "Extreme Gradient Boosting learns from lagged consumption "
                "values, rolling averages, and calendar month features."
            )
            st.caption("Input: lag features + monthly seasonality")

        with st.container(border=True):
            st.markdown("### Important limitations")
            st.write(
                "Forecasts depend on the uploaded dataset and the assumptions "
                "learned by each model. Historical evaluation scores describe "
                "the selected holdout period only; they do not guarantee future accuracy."
            )

    st.markdown(
        '<div class="section-heading">Data handling</div>'
        '<div class="section-copy">The app accepts CSV and Excel files, maps '
        'date and consumption columns, sorts records chronologically, excludes '
        'invalid rows, and avoids training sequences that cross missing calendar '
        'months. Duplicate months are averaged during preparation, so review '
        'your source data before interpreting results.</div>',
        unsafe_allow_html=True
    )


st.divider()

st.markdown(
    '<div class="note">LOAD FORECAST STUDIO · Monthly electrical load forecasting '
    '· Academic research prototype</div>',
    unsafe_allow_html=True
)
