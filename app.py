import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Monthly Electrical Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Monthly Electrical Load Forecasting")
st.write(
    "Compare LSTM and XGBoost models for monthly "
    "electricity consumption forecasting."
)

uploaded_file = st.file_uploader(
    "Upload monthly consumption data",
    type=["xlsx", "csv"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required = ["Year", "Month", "Consumption_kWh"]

    if not all(c in df.columns for c in required):
        st.error(
            "Required columns: Year, Month, Consumption_kWh"
        )
        st.stop()

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-"
        + df["Month"].astype(str) + "-01",
        errors="coerce"
    )
    df["Consumption_kWh"] = pd.to_numeric(
        df["Consumption_kWh"], errors="coerce"
    )
    df = df.dropna(
        subset=["Date", "Consumption_kWh"]
    ).sort_values("Date")

    if df["Date"].duplicated().any():
        st.error("Please make sure each month appears only once.")
        st.stop()

    if len(df) < 18:
        st.error(
            "Please upload at least 18 valid monthly records "
            "for this prototype."
        )
        st.stop()

    st.subheader("Historical Consumption")
    st.dataframe(df[required], use_container_width=True)

    fig = px.line(
        df, x="Date", y="Consumption_kWh",
        markers=True,
        title="Monthly Electricity Consumption"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Settings")
    horizon = st.selectbox(
        "How many months do you want to forecast?",
        [1, 3, 6, 12, 18, 24],
        index=3
    )

    test_size = st.slider(
        "Holdout months for model evaluation",
        min_value=3,
        max_value=min(12, max(3, len(df) // 4)),
        value=min(3, max(3, len(df) // 4))
    )

    if st.button("Train Models and Forecast", type="primary"):
        y = df["Consumption_kWh"].to_numpy(dtype=float)
        dates = df["Date"].reset_index(drop=True)

        if not np.isfinite(y).all() or (y < 0).any():
            st.error("Consumption must be valid non-negative numbers.")
            st.stop()

        split = len(y) - test_size
        if split < 12:
            st.error("Not enough training months for evaluation.")
            st.stop()

        def make_features(values, index):
            row = {"month": (index % 12) + 1}
            for lag in [1, 2, 3, 6, 12]:
                row[f"lag_{lag}"] = (
                    values[index - lag] if index >= lag else np.nan
                )
            return row

        # XGBoost: sequential lag features
        feature_rows = [
            make_features(y, i) for i in range(12, len(y))
        ]
        X = pd.DataFrame(feature_rows)
        targets = y[12:]
        train_end = split - 12

        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            objective="reg:squarederror",
            random_state=42
        )
        xgb_model.fit(X.iloc[:train_end], targets[:train_end])
        xgb_test = xgb_model.predict(X.iloc[train_end:])
        xgb_actual = targets[train_end:]

        # LSTM: use the previous 12 months to predict the next month.
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(y[:split].reshape(-1, 1))

        def sequences(values, window=12):
            xs, ys = [], []
            for i in range(window, len(values)):
                xs.append(values[i-window:i, 0])
                ys.append(values[i, 0])
            return np.array(xs), np.array(ys)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.callbacks import EarlyStopping

        X_train, y_train = sequences(scaled_train)
        model = Sequential([
            Input(shape=(12, 1)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_train.reshape(-1, 12, 1),
            y_train,
            epochs=100,
            batch_size=8,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="loss", patience=10,
                    restore_best_weights=True
                )
            ]
        )

        lstm_test_scaled = []
        history = list(scaled_train.flatten())
        for _ in range(test_size):
            seq = np.array(history[-12:]).reshape(1, 12, 1)
            pred = float(model.predict(seq, verbose=0)[0, 0])
            lstm_test_scaled.append(pred)
            history.append(pred)

        lstm_test = scaler.inverse_transform(
            np.array(lstm_test_scaled).reshape(-1, 1)
        ).flatten()
        lstm_actual = y[split:]

        # Evaluate only on the same holdout months.
        # Refit XGBoost features using the correct holdout alignment.
        xgb_model_eval = XGBRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.03,
            objective="reg:squarederror",
            random_state=42
        )
        all_X = pd.DataFrame(
            [make_features(y, i) for i in range(12, len(y))]
        )
        train_count = split - 12
        xgb_model_eval.fit(all_X.iloc[:train_count], y[12:split])
        xgb_test = xgb_model_eval.predict(
            pd.DataFrame([
                make_features(y, i) for i in range(split, len(y))
            ])
        )

        def metrics(actual, predicted):
            actual = np.asarray(actual)
            predicted = np.asarray(predicted)
            nonzero = actual != 0
            mape = (
                np.mean(np.abs(
                    (actual[nonzero] - predicted[nonzero])
                    / actual[nonzero]
                )) * 100 if nonzero.any() else np.nan
            )
            return {
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "MAPE (%)": mape
            }

        lstm_metrics = metrics(lstm_actual, lstm_test)
        xgb_metrics = metrics(lstm_actual, xgb_test)

        st.subheader("Model Evaluation")
        results = pd.DataFrame(
            [lstm_metrics, xgb_metrics],
            index=["LSTM", "XGBoost"]
        )
        st.dataframe(results.style.format("{:.3f}"))

        chart_df = pd.DataFrame({
            "Date": dates.iloc[split:].to_numpy(),
            "Actual": lstm_actual,
            "LSTM": lstm_test,
            "XGBoost": xgb_test
        })
        chart_long = chart_df.melt(
            id_vars="Date", var_name="Series", value_name="kWh"
        )
        st.subheader("Actual vs Predicted Consumption")
        st.plotly_chart(
            px.line(
                chart_long, x="Date", y="kWh",
                color="Series", markers=True
            ),
            use_container_width=True
        )

        # Refit models using all available observations for future forecasts.
        final_scaler = MinMaxScaler()
        all_scaled = final_scaler.fit_transform(y.reshape(-1, 1))
        final_seq_x, final_seq_y = sequences(all_scaled)

        final_lstm = Sequential([
            Input(shape=(12, 1)),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        final_lstm.compile(optimizer="adam", loss="mse")
        final_lstm.fit(
            final_seq_x.reshape(-1, 12, 1),
            final_seq_y,
            epochs=50,
            batch_size=8,
            verbose=0
        )

        final_xgb = XGBRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.03,
            objective="reg:squarederror",
            random_state=42
        )
        final_X = pd.DataFrame([
            make_features(y, i) for i in range(12, len(y))
        ])
        final_xgb.fit(final_X, y[12:])

        future_lstm = []
        future_history = list(all_scaled.flatten())
        for _ in range(horizon):
            seq = np.array(future_history[-12:]).reshape(1, 12, 1)
            pred = float(final_lstm.predict(seq, verbose=0)[0, 0])
            future_lstm.append(pred)
            future_history.append(pred)

        future_lstm = final_scaler.inverse_transform(
            np.array(future_lstm).reshape(-1, 1)
        ).flatten()

        future_xgb = []
        history_y = list(y)
        for _ in range(horizon):
            idx = len(history_y)
            features = pd.DataFrame([make_features(history_y, idx)])
            pred = float(final_xgb.predict(features)[0])
            future_xgb.append(pred)
            history_y.append(pred)

        future_dates = pd.date_range(
            dates.iloc[-1] + pd.offsets.MonthBegin(1),
            periods=horizon, freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "LSTM_Forecast_kWh": future_lstm,
            "XGBoost_Forecast_kWh": future_xgb
        })

        st.subheader(f"Next {horizon} Months Forecast")
        st.dataframe(forecast_df, use_container_width=True)
        st.plotly_chart(
            px.line(
                forecast_df, x="Date",
                y=["LSTM_Forecast_kWh", "XGBoost_Forecast_kWh"],
                markers=True,
                title="Future Monthly Forecast"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download Forecast CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="monthly_load_forecast.csv",
            mime="text/csv"
        )

        st.info(
            "Research prototype: results depend on data quality and "
            "the available historical records. Review model settings "
            "and evaluation methodology with your adviser before "
            "using results in the thesis."
        )

else:
    st.info("Upload a dataset to start.")
