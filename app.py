
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="LSTM Load Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ LSTM Electrical Load Forecasting")
st.write("Upload your monthly records and configure your own forecasting model.")

# ---------------- FILE READER ----------------

def read_file(file):
    if file is None:
        return None

    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)


def numeric_columns(data):
    return data.select_dtypes(include=np.number).columns.tolist()


# ---------------- 1. MONTH / DATE UPLOAD ----------------

st.header("1. Upload Month / Date File")

month_file = st.file_uploader(
    "Upload your month or date records",
    type=["csv", "xlsx", "xls"],
    key="month_file"
)

# ---------------- 2. CONSUMPTION UPLOAD ----------------

st.header("2. Upload Electricity Consumption File")

consumption_file = st.file_uploader(
    "Upload your electricity consumption records",
    type=["csv", "xlsx", "xls"],
    key="consumption_file"
)

# ---------------- 3. OTHER VARIABLES UPLOAD ----------------

st.header("3. Upload Other Variables")

other_file = st.file_uploader(
    "Upload one file containing your optional variables",
    type=["csv", "xlsx", "xls"],
    key="other_file"
)

st.caption(
    "You may combine manpower, operating hours, production volume, "
    "peak demand, diesel consumption, and other variables in one file."
)

if month_file is None or consumption_file is None:
    st.info("Please upload both the Month/Date file and Consumption file.")
    st.stop()

try:
    month_df = read_file(month_file)
    consumption_df = read_file(consumption_file)
    other_df = read_file(other_file) if other_file else None

except Exception as e:
    st.error(f"File reading error: {e}")
    st.stop()


# ---------------- COLUMN CONFIGURATION ----------------

st.header("4. Configure Your Dataset")

st.subheader("Month / Date Column")

date_col = st.selectbox(
    "Select the column containing your dates or months",
    month_df.columns
)

st.subheader("Consumption Column")

consumption_numeric = numeric_columns(consumption_df)

if not consumption_numeric:
    st.error("The consumption file must contain a numeric column.")
    st.stop()

consumption_col = st.selectbox(
    "Select electricity consumption column",
    consumption_numeric
)

# ---------------- BUILD DATASET ----------------

dates = pd.to_datetime(month_df[date_col], errors="coerce")

if dates.isna().any():
    st.error(
        "Some month/date values could not be recognized. "
        "Please check your Month/Date file."
    )
    st.stop()

data = pd.DataFrame({
    "Date": dates,
    "Consumption": pd.to_numeric(
        consumption_df[consumption_col], errors="coerce"
    )
})

# Align uploaded files by row order
if len(month_df) != len(consumption_df):
    st.error("Month/Date and Consumption files must have the same number of rows.")
    st.stop()

if other_df is not None:
    if len(other_df) != len(data):
        st.error(
            "The Other Variables file must have the same number of rows "
            "as the Month/Date and Consumption files."
        )
        st.stop()

    other_numeric = numeric_columns(other_df)

    if not other_numeric:
        st.warning("No numeric optional variables were found.")

    else:
        st.subheader("Select Optional Variables")

        selected_features = []

        for col in other_numeric:
            if st.checkbox(col, key=f"feature_{col}"):
                selected_features.append(col)

        for col in selected_features:
            data[col] = pd.to_numeric(other_df[col], errors="coerce")

else:
    selected_features = []
    st.info("No optional variables uploaded. The model will use consumption only.")

# ---------------- DATA PREVIEW ----------------

data = data.sort_values("Date").reset_index(drop=True)

st.header("5. Dataset Preview")

st.write("Number of observations:", len(data))
st.write("Selected optional variables:", selected_features)

st.dataframe(data, use_container_width=True)

if data.isnull().any().any():
    st.warning("Missing or non-numeric values were detected.")
    data = data.dropna().reset_index(drop=True)
    st.write("Complete observations remaining:", len(data))

if len(data) < 15:
    st.error(
        "Insufficient observations for this initial LSTM version. "
        "Please provide more historical records."
    )
    st.stop()

# ---------------- MODEL SETTINGS ----------------

st.header("6. Forecasting Configuration")

frequency = st.selectbox(
    "Data frequency",
    ["Monthly", "Quarterly", "Weekly", "Daily"]
)

lookback = st.number_input(
    "Historical observations used by the model",
    min_value=2,
    max_value=120,
    value=6
)

forecast_horizon = st.number_input(
    "Number of future periods to forecast",
    min_value=1,
    max_value=24,
    value=3
)

test_size = st.number_input(
    "Number of observations for testing",
    min_value=1,
    max_value=24,
    value=3
)

epochs = st.slider("Training epochs", 20, 300, 100, 10)
units = st.slider("LSTM units", 16, 128, 32, 16)

# ---------------- TRAINING ----------------

st.header("7. Train LSTM Model")

if st.button("Train Model and Generate Forecast", type="primary"):

    features = ["Consumption"] + selected_features

    model_data = data[features].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_data) <= lookback + test_size + 3:
        st.error(
            "Not enough complete observations for the selected "
            "lookback and test settings."
        )
        st.stop()

    values = model_data.to_numpy(dtype=float)

    target = model_data["Consumption"].to_numpy().reshape(-1, 1)

    split = len(model_data) - int(test_size)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(values[:split])
    y_scaler.fit(target[:split])

    x_scaled = x_scaler.transform(values)
    y_scaled = y_scaler.transform(target).ravel()

    X, Y = [], []

    for i in range(lookback, split):
        X.append(x_scaled[i-lookback:i])
        Y.append(y_scaled[i])

    X = np.array(X)
    Y = np.array(Y)

    if len(X) < 3:
        st.error("Not enough training sequences. Reduce the lookback period.")
        st.stop()

    X_test = x_scaled[split-lookback:split].reshape(
        1, lookback, len(features)
    )

    tf.keras.backend.clear_session()

    model = Sequential([
        tf.keras.Input(shape=(lookback, len(features))),
        LSTM(units),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=12,
        restore_best_weights=True
    )

    with st.spinner("Training LSTM model..."):
        model.fit(
            X,
            Y,
            epochs=epochs,
            batch_size=min(8, len(X)),
            callbacks=[early_stop],
            verbose=0
        )

    # ---------------- TESTING ----------------

    test_predictions = []
    actual_values = []

    history = x_scaled[:split].copy()

    for i in range(split, len(model_data)):
        input_data = history[-lookback:].reshape(
            1, lookback, len(features)
        )

        prediction = model.predict(input_data, verbose=0)[0, 0]

        test_predictions.append(prediction)
        actual_values.append(y_scaled[i])

        # Append actual observed inputs for the next test step
        history = np.vstack([history, x_scaled[i]])

    test_predictions = y_scaler.inverse_transform(
        np.array(test_predictions).reshape(-1, 1)
    ).ravel()

    actual_values = y_scaler.inverse_transform(
        np.array(actual_values).reshape(-1, 1)
    ).ravel()

    mae = mean_absolute_error(actual_values, test_predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, test_predictions))

    nonzero = np.abs(actual_values) > 1e-8

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual_values[nonzero] - test_predictions[nonzero])
                / actual_values[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    # ---------------- FUTURE FORECAST ----------------

    last_input = x_scaled[-lookback:].reshape(
        1, lookback, len(features)
    )

    future_predictions = []

    for _ in range(int(forecast_horizon)):
        pred = model.predict(last_input, verbose=0)[0, 0]

        future_predictions.append(pred)

        # Future optional feature values are unavailable here.
        # Use the latest scaled feature values as a persistence assumption.
        next_row = last_input[0, -1, :].copy()
        next_row[0] = pred

        last_input = np.concatenate(
            [last_input[:, 1:, :], next_row.reshape(1, 1, -1)],
            axis=1
        )

    future_predictions = y_scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).ravel()

    # ---------------- RESULTS ----------------

    st.success("LSTM model training completed!")

    st.subheader("Model Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:,.3f}")
    c2.metric("RMSE", f"{rmse:,.3f}")
    c3.metric("MAPE", "N/A" if np.isnan(mape) else f"{mape:.2f}%")

    evaluation = pd.DataFrame({
        "Actual": actual_values,
        "Predicted": test_predictions
    })

    st.write("Actual vs Predicted")
    st.dataframe(evaluation, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_values, marker="o", label="Actual")
    ax.plot(test_predictions, marker="o", label="Predicted")
    ax.set_title("LSTM Testing Results")
    ax.set_xlabel("Test Observation")
    ax.set_ylabel("Electricity Consumption")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Forecast dates
    freq_map = {
        "Monthly": "MS",
        "Quarterly": "QS",
        "Weekly": "W",
        "Daily": "D"
    }

    future_dates = pd.date_range(
        start=data["Date"].iloc[-1],
        periods=int(forecast_horizon) + 1,
        freq=freq_map[frequency]
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Consumption": future_predictions
    })

    st.subheader("Future Electricity Consumption Forecast")
    st.dataframe(forecast_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        data["Date"],
        data["Consumption"],
        marker="o",
        label="Historical Consumption"
    )
    ax2.plot(
        future_dates,
        future_predictions,
        marker="o",
        linestyle="--",
        label="Forecast"
    )
    ax2.set_title("Historical Consumption and LSTM Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Electricity Consumption")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name="lstm_forecast.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Evaluation CSV",
        evaluation.to_csv(index=False),
        file_name="lstm_evaluation.csv",
        mime="text/csv"
    )
