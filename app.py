import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

st.set_page_config(
    page_title="Monthly Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# -------------------- DESIGN --------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2.1rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #777777;
    margin-bottom: 1.5rem;
}
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">⚡ Monthly Electrical Load Forecasting</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Compare LSTM and XGBoost using historical monthly electricity consumption.</div>',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About this forecasting system"):
    st.write("""
    This application uses historical monthly electricity consumption to
    forecast future consumption using two machine learning models:

    - **LSTM:** A neural network designed to learn patterns from sequential data.
    - **XGBoost:** A machine learning model that combines decision trees.

    The models are evaluated using past months that are set aside from training.
    Their evaluation results help users compare prediction errors.

    Forecast results are estimates and should be interpreted together with
    the quality and completeness of the uploaded data.
    """)

# -------------------- DATA UPLOAD --------------------

st.header("1. Upload Historical Data")

st.write(
    "Upload an Excel or CSV file containing monthly electricity consumption. "
    "The file should include a month/date column and a consumption column."
)

uploaded_file = st.file_uploader(
    "Choose an Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None:
    st.info(
        "Upload your dataset to begin. Example columns: "
        "**Year**, **Month**, and **Consumption_kWh**."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]

except Exception as e:
    st.error(f"Unable to read the file: {e}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

st.subheader("Preview of Uploaded Data")
st.dataframe(raw_df.head(10), use_container_width=True)

# -------------------- COLUMN SELECTION --------------------

st.subheader("Identify Your Data Columns")

columns = list(raw_df.columns)

date_candidates = [
    c for c in columns
    if c.lower() in ["date", "month_date", "billing_date", "period"]
]

year_candidates = [c for c in columns if c.lower() == "year"]
month_candidates = [c for c in columns if c.lower() == "month"]

numeric_candidates = [
    c for c in columns
    if pd.api.types.is_numeric_dtype(raw_df[c])
    and c not in year_candidates
]

with st.container(border=True):
    date_mode = st.radio(
        "How is the month recorded in your file?",
        ["One date column", "Separate year and month columns"],
        horizontal=True
    )

    if date_mode == "One date column":
        default_date_index = (
            columns.index(date_candidates[0])
            if date_candidates else 0
        )
        date_col = st.selectbox(
            "Date column",
            columns,
            index=default_date_index
        )
        year_col = None
        month_col = None

    else:
        if len(year_candidates) == 0 or len(month_candidates) == 0:
            st.info(
                "Select the columns containing the year and month. "
                "If your file has one complete date column, choose "
                "'One date column' instead."
            )

        year_col = st.selectbox(
            "Year column",
            columns,
            index=columns.index(year_candidates[0])
            if year_candidates else 0
        )

        month_col = st.selectbox(
            "Month column",
            columns,
            index=columns.index(month_candidates[0])
            if month_candidates else min(1, len(columns) - 1)
        )

        date_col = None

    if not numeric_candidates:
        st.error("No numeric consumption column was detected.")
        st.stop()

    preferred_consumption = [
        c for c in numeric_candidates
        if any(word in c.lower() for word in ["consumption", "kwh", "load"])
    ]

    default_consumption_index = (
        numeric_candidates.index(preferred_consumption[0])
        if preferred_consumption else 0
    )

    consumption_col = st.selectbox(
        "Electricity consumption column (kWh)",
        numeric_candidates,
        index=default_consumption_index
    )

# -------------------- PREPARE DATA --------------------

try:
    data = raw_df.copy()

    if date_mode == "One date column":
        data["_Date"] = pd.to_datetime(
            data[date_col],
            errors="coerce"
        ).dt.to_period("M")
    else:
        year_values = pd.to_numeric(data[year_col], errors="coerce")
        month_values = data[month_col]

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

            text_value = str(value).strip().lower()

            if text_value in month_names:
                return month_names[text_value]

            try:
                number = int(float(text_value))
                if 1 <= number <= 12:
                    return number
            except (ValueError, TypeError):
                pass

            return np.nan

        month_numbers = month_values.apply(convert_month)

        data["_Date"] = [
            pd.Period(year=int(y), month=int(m), freq="M")
            if pd.notna(y) and pd.notna(m)
            and 1 <= int(m) <= 12
            else pd.NaT
            for y, m in zip(year_values, month_numbers)
        ]

    data["_Consumption"] = pd.to_numeric(
        data[consumption_col],
        errors="coerce"
    )

    invalid_rows = data["_Date"].isna() | data["_Consumption"].isna()
    invalid_count = int(invalid_rows.sum())

    data = data.loc[~invalid_rows, ["_Date", "_Consumption"]].copy()

    if data.empty:
        st.error("No valid date and consumption records were found.")
        st.stop()

    if (data["_Consumption"] < 0).any():
        st.error(
            "Negative consumption values were found. "
            "Please check the dataset before training."
        )
        st.stop()

    duplicate_count = int(data["_Date"].duplicated().sum())

    if duplicate_count > 0:
        st.error(
            "Multiple records were found for the same month. "
            "Please combine them into one monthly consumption value "
            "before running the models."
        )
        st.stop()

    data = data.sort_values("_Date").reset_index(drop=True)

except Exception as e:
    st.error(f"Error preparing the data: {e}")
    st.stop()

if invalid_count:
    st.warning(
        f"{invalid_count} row(s) with invalid dates or consumption "
        "values were excluded. Please verify these rows in your source file."
    )

# -------------------- CHECK MONTHLY CONTINUITY --------------------

periods = data["_Date"].tolist()
missing_months = []

for previous, current in zip(periods[:-1], periods[1:]):
    if current.ordinal - previous.ordinal > 1:
        gap = current.ordinal - previous.ordinal - 1
        missing_months.append(
            f"{previous + 1} to {current - 1} ({gap} missing month(s))"
        )

if missing_months:
    st.info(
        "Gaps in the monthly timeline were detected: "
        + "; ".join(missing_months[:5])
        + (" ..." if len(missing_months) > 5 else "")
        + ". Training sequences that cross a missing-month gap are excluded. "
        "The models will learn only from windows whose dates are consecutive."
    )

y = data["_Consumption"].astype(float).to_numpy()
dates = data["_Date"].tolist()

if not np.isfinite(y).all():
    st.error("The consumption data contains invalid numeric values.")
    st.stop()

if len(y) < 24:
    st.error(
        "At least 24 valid monthly observations are recommended for this "
        "version of the application. Upload more historical data."
    )
    st.stop()

# -------------------- DATA SUMMARY --------------------

st.header("2. Historical Consumption")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Monthly Records", f"{len(data):,}")

with col2:
    st.metric("Average Consumption", f"{np.mean(y):,.2f} kWh")

with col3:
    st.metric("Latest Consumption", f"{y[-1]:,.2f} kWh")

history_df = pd.DataFrame({
    "Month": [d.to_timestamp() for d in dates],
    "Consumption (kWh)": y
})

history_fig = go.Figure()
history_fig.add_trace(go.Scatter(
    x=history_df["Month"],
    y=history_df["Consumption (kWh)"],
    mode="lines+markers",
    name="Actual Consumption"
))
history_fig.update_layout(
    title="Historical Monthly Electricity Consumption",
    xaxis_title="Month",
    yaxis_title="Consumption (kWh)",
    xaxis=dict(type="date"),
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(history_fig, use_container_width=True)

# -------------------- FORECAST SETTINGS --------------------

st.header("3. Forecast Settings")

st.write(
    "Choose the number of future months to forecast. Model performance is "
    "evaluated separately on up to the latest 12 consecutive historical months "
    "so the comparison is not based on only one to three test observations."
)

horizon = st.selectbox(
    "Forecast horizon",
    options=[1, 2, 3],
    format_func=lambda x: f"{x} month(s)"
)

st.info(
    f"Future forecast: **{horizon} month(s)**. The evaluation period is selected "
    "separately from the latest consecutive historical segment (up to 12 months)."
)

if len(y) - horizon < 18:
    st.error(
        "There are not enough training observations for this evaluation setup. "
        "Please upload more monthly data or select a shorter forecast horizon."
    )
    st.stop()

# -------------------- MODEL FUNCTIONS --------------------

LOOKBACK = 12


def is_consecutive_window(periods, start_idx, end_idx):
    """True when periods[start_idx:end_idx+1] are consecutive calendar months."""
    window = periods[start_idx:end_idx + 1]
    if len(window) <= 1:
        return True
    return all(
        window[j].ordinal - window[j - 1].ordinal == 1
        for j in range(1, len(window))
    )


def latest_contiguous_segment_length(periods):
    """Count consecutive monthly records at the end of the dataset."""
    if not periods:
        return 0
    count = 1
    for i in range(len(periods) - 1, 0, -1):
        if periods[i].ordinal - periods[i - 1].ordinal != 1:
            break
        count += 1
    return count


def xgb_features(history, target_period):
    """Create lag and calendar features for one target month."""
    h = np.asarray(history, dtype=float)
    if len(h) < LOOKBACK:
        raise ValueError(f"At least {LOOKBACK} consecutive months are needed.")

    features = [h[-lag] for lag in range(1, LOOKBACK + 1)]
    features += [
        float(np.mean(h[-3:])),
        float(np.mean(h[-6:])),
        float(np.mean(h[-12:]))
    ]

    month = target_period.month
    features += [
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12)
    ]
    return features


def make_xgb_training_data(values, periods):
    """Create examples only when all 12 lag months and the target are consecutive."""
    features, targets = [], []
    for i in range(LOOKBACK, len(values)):
        if not is_consecutive_window(periods, i - LOOKBACK, i):
            continue
        features.append(xgb_features(values[i - LOOKBACK:i], periods[i]))
        targets.append(values[i])
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def recursive_xgb_forecast(model, history, start_period, steps):
    history = list(np.asarray(history, dtype=float))
    predictions = []
    for step in range(steps):
        target_period = start_period + step
        features = xgb_features(history, target_period)
        prediction = float(model.predict(np.asarray([features]))[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)


def train_lstm(training_values, training_periods):
    """Train LSTM on 12-month sequences that do not cross calendar gaps."""
    training_values = np.asarray(training_values, dtype=float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(training_values.reshape(-1, 1)).flatten()

    X, target = [], []
    for i in range(LOOKBACK, len(scaled)):
        if not is_consecutive_window(training_periods, i - LOOKBACK, i):
            continue
        X.append(scaled[i - LOOKBACK:i])
        target.append(scaled[i])

    if len(X) < 5:
        raise ValueError(
            "Not enough valid consecutive 12-month sequences to train LSTM. "
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

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )
    model.fit(
        X,
        target,
        epochs=40,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
        shuffle=False
    )
    return model, scaler


def recursive_lstm_forecast(model, scaler, history, steps):
    """Forecast recursively from the latest 12 consecutive observations."""
    history = np.asarray(history, dtype=float)
    if len(history) < LOOKBACK:
        raise ValueError(f"At least {LOOKBACK} history values are required.")
    history_scaled = scaler.transform(history.reshape(-1, 1)).flatten().tolist()
    predictions = []

    for _ in range(steps):
        seq = np.asarray(history_scaled[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred_scaled = float(model.predict(seq, verbose=0)[0][0])
        pred_original = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        pred_original = max(0.0, pred_original)
        predictions.append(pred_original)
        history_scaled.append(float(scaler.transform([[pred_original]])[0][0]))

    return np.asarray(predictions)


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


# -------------------- TRAINING AND EVALUATION --------------------

st.header("4. Model Training and Evaluation")

if st.button(
    "🚀 Train Models and Generate Forecast",
    type="primary",
    use_container_width=True
):

    progress = st.progress(0)
    status = st.empty()

    try:
        # Use up to 12 months from the latest uninterrupted segment for evaluation.
        latest_segment_length = latest_contiguous_segment_length(dates)
        evaluation_horizon = min(12, max(1, latest_segment_length - LOOKBACK))
        split = len(y) - evaluation_horizon
        train_values = y[:split]
        train_dates = dates[:split]

        if latest_segment_length <= LOOKBACK or len(train_values) <= LOOKBACK:
            raise ValueError(
                "There are not enough consecutive monthly records before the "
                "evaluation period. At least 13 consecutive training months "
                "are needed."
            )

        status.info("Step 1 of 5: Preparing gap-safe training and evaluation data...")
        progress.progress(10)

        X_train, y_train = make_xgb_training_data(train_values, train_dates)
        if len(X_train) < 5:
            raise ValueError(
                "Not enough valid XGBoost training sequences after excluding "
                "windows that cross missing-month gaps."
            )

        status.info("Step 2 of 5: Training the XGBoost model...")
        progress.progress(25)

        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
        xgb_model.fit(X_train, y_train)

        status.info("Step 3 of 5: Training the LSTM model...")
        progress.progress(45)

        lstm_model, scaler = train_lstm(train_values, train_dates)

        status.info("Step 4 of 5: Evaluating both models on the same past months...")
        progress.progress(70)

        evaluation_dates = dates[split:]
        actual = y[split:]

        # The holdout begins immediately after the training data.
        # Recursive forecasts use the previous 12 months.
        xgb_test = recursive_xgb_forecast(
            xgb_model,
            train_values[-LOOKBACK:],
            evaluation_dates[0],
            evaluation_horizon
        )
        lstm_test = recursive_lstm_forecast(
            lstm_model,
            scaler,
            train_values[-LOOKBACK:],
            evaluation_horizon
        )

        xgb_metrics = calculate_metrics(actual, xgb_test)
        lstm_metrics = calculate_metrics(actual, lstm_test)

        metrics_df = pd.DataFrame({
            "Metric": ["MAE (kWh)", "RMSE (kWh)", "MAPE (%)"],
            "LSTM": list(lstm_metrics),
            "XGBoost": list(xgb_metrics)
        })

        status.info(
            "Step 5 of 5: Training final models and generating future forecasts..."
        )
        progress.progress(85)

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

        last_period = dates[-1] + 1
        latest_history = y[-LOOKBACK:]

        future_xgb = recursive_xgb_forecast(
            final_xgb, latest_history, last_period, horizon
        )
        future_lstm = recursive_lstm_forecast(
            final_lstm, final_scaler, latest_history, horizon
        )

        future_periods = [
            dates[-1] + i for i in range(1, horizon + 1)
        ]

        forecast_df = pd.DataFrame({
            "Month": [str(p) for p in future_periods],
            "LSTM Forecast (kWh)": future_lstm,
            "XGBoost Forecast (kWh)": future_xgb
        })

        progress.progress(100)
        status.success("Training, evaluation, and forecasting completed!")

        st.session_state["metrics_df"] = metrics_df
        st.session_state["evaluation_df"] = pd.DataFrame({
            "Month": [str(d) for d in evaluation_dates],
            "Actual (kWh)": actual,
            "LSTM Prediction (kWh)": lstm_test,
            "XGBoost Prediction (kWh)": xgb_test
        })
        st.session_state["forecast_df"] = forecast_df
        st.session_state["lstm_metrics"] = lstm_metrics
        st.session_state["xgb_metrics"] = xgb_metrics
        st.session_state["horizon"] = horizon
        st.session_state["evaluation_horizon"] = evaluation_horizon

    except Exception as e:
        status.error("The process could not be completed.")
        st.error(
            "Please check the uploaded dates, consumption values, and app logs. "
            f"Details: {e}"
        )

# -------------------- DISPLAY RESULTS --------------------

if "metrics_df" in st.session_state:

    metrics_df = st.session_state["metrics_df"]
    evaluation_df = st.session_state["evaluation_df"]
    forecast_df = st.session_state["forecast_df"]
    lstm_metrics = st.session_state["lstm_metrics"]
    xgb_metrics = st.session_state["xgb_metrics"]

    st.header("5. Model Evaluation Results")

    st.caption(
        f"Evaluation window: "
        f"{st.session_state.get('evaluation_horizon', len(evaluation_df))} "
        "latest consecutive month(s). The future forecast horizon is selected separately."
    )

    st.write(
        "The table compares both models on the same held-out historical months. "
        "The evaluation uses up to 12 consecutive months from the latest "
        "continuous segment. Lower errors mean closer predictions on this test period."
    )

    st.dataframe(
        metrics_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    # Dynamic model comparison
    metric_names = ["MAE", "RMSE", "MAPE"]
    metric_results = [
        (metric_names[0], lstm_metrics[0], xgb_metrics[0]),
        (metric_names[1], lstm_metrics[1], xgb_metrics[1]),
        (metric_names[2], lstm_metrics[2], xgb_metrics[2])
    ]

    lstm_wins = 0
    xgb_wins = 0
    ties = 0
    comparable_metrics = 0

    for name, lstm_value, xgb_value in metric_results:
        if not np.isfinite(lstm_value) or not np.isfinite(xgb_value):
            continue

        comparable_metrics += 1

        if np.isclose(lstm_value, xgb_value):
            ties += 1
        elif lstm_value < xgb_value:
            lstm_wins += 1
        else:
            xgb_wins += 1

    st.subheader("What do the results mean?")

    if comparable_metrics == 0:
        st.warning(
            "The models could not be compared using these metrics. "
            "Check the evaluation data, especially whether actual values are zero."
        )
    elif lstm_wins > xgb_wins:
        st.success(
            f"LSTM had lower errors in {lstm_wins} of the {comparable_metrics} "
            "comparable metrics. It performed better overall on these "
            "evaluation months, but this does not guarantee better future results."
        )
    elif xgb_wins > lstm_wins:
        st.success(
            f"XGBoost had lower errors in {xgb_wins} of the {comparable_metrics} "
            "comparable metrics. It performed better overall on these "
            "evaluation months, but this does not guarantee better future results."
        )
    else:
        st.info(
            "The models had similar results across the comparable metrics. "
            "Review each metric and the prediction chart before deciding which "
            "model performed better."
        )

    with st.expander("📘 Understand MAE, RMSE, and MAPE"):
        st.markdown("""
        **1. MAE — Mean Absolute Error**

        MAE measures the average absolute difference between actual consumption
        and predicted consumption. Its unit is **kWh**.

        For example, an MAE of 500 kWh means that the model's predictions were
        off by an average of 500 kWh during the evaluation period.

        **Lower MAE is better** because predictions are closer to the actual values
        on average.

        **2. RMSE — Root Mean Squared Error**

        RMSE measures prediction error in kWh too, but it penalizes larger errors
        more strongly than MAE.

        For example, an RMSE of 700 kWh means the root-mean-square error was
        700 kWh. It does not mean every prediction was wrong by exactly 700 kWh.

        **Lower RMSE is better.** If RMSE is noticeably higher than MAE, some
        larger prediction errors may be affecting the result.

        **3. MAPE — Mean Absolute Percentage Error**

        MAPE expresses the average absolute prediction error as a percentage of
        actual consumption.

        For example, a MAPE of 5% means the average absolute percentage error
        was 5% for the nonzero actual values included in the calculation.

        **Lower MAPE is better.** MAPE is not calculated for actual values equal
        to zero because division by zero is undefined.

        **Important:** None of these metrics alone proves that a model is
        universally accurate. Interpret them with the evaluation period, the
        amount and quality of available data, and the forecasting objective.
        """)

    st.subheader("Evaluation: Actual vs Predicted")

    evaluation_fig = go.Figure()

    evaluation_fig.add_trace(go.Scatter(
        x=evaluation_df["Month"],
        y=evaluation_df["Actual (kWh)"],
        mode="lines+markers",
        name="Actual"
    ))

    evaluation_fig.add_trace(go.Scatter(
        x=evaluation_df["Month"],
        y=evaluation_df["LSTM Prediction (kWh)"],
        mode="lines+markers",
        name="LSTM Prediction"
    ))

    evaluation_fig.add_trace(go.Scatter(
        x=evaluation_df["Month"],
        y=evaluation_df["XGBoost Prediction (kWh)"],
        mode="lines+markers",
        name="XGBoost Prediction"
    ))

    evaluation_fig.update_layout(
        xaxis_title="Evaluation Month",
        yaxis_title="Consumption (kWh)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(evaluation_fig, use_container_width=True)

    st.dataframe(
        evaluation_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.header("6. Future Monthly Forecast")

    st.write(
        "These are estimated future consumption values from models retrained "
        "using all valid records in the uploaded dataset. Forecasts are not actual "
        "billed consumption."
    )

    forecast_fig = go.Figure()

    forecast_fig.add_trace(go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["LSTM Forecast (kWh)"],
        mode="lines+markers",
        name="LSTM Forecast"
    ))

    forecast_fig.add_trace(go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["XGBoost Forecast (kWh)"],
        mode="lines+markers",
        name="XGBoost Forecast"
    ))

    forecast_fig.update_layout(
        title="Forecasted Monthly Electricity Consumption",
        xaxis_title="Future Month",
        yaxis_title="Forecast Consumption (kWh)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(forecast_fig, use_container_width=True)

    st.dataframe(
        forecast_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    # -------------------- DOWNLOADS --------------------

    st.header("7. Download Results")

    csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Future Forecast (CSV)",
        data=csv_forecast,
        file_name="monthly_load_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )

    csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Evaluation Metrics (CSV)",
        data=csv_metrics,
        file_name="model_evaluation_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

    csv_evaluation = evaluation_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Actual vs Predicted (CSV)",
        data=csv_evaluation,
        file_name="actual_vs_predicted.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()
st.caption(
    "Academic prototype for monthly electrical consumption forecasting. "
    "Results depend on the uploaded data and model settings."
)
