
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Monthly Electrical Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Monthly Electrical Load Forecasting")
st.write(
    "A forecasting application for monthly electricity "
    "consumption in a plastic manufacturing plant."
)

st.subheader("Upload Historical Consumption Data")

st.write(
    "Upload an Excel or CSV file containing the columns: "
    "Year, Month, and Consumption_kWh."
)

uploaded_file = st.file_uploader(
    "Upload your electricity consumption data",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(data, use_container_width=True)

        required_columns = [
            "Year",
            "Month",
            "Consumption_kWh"
        ]

        if all(col in data.columns for col in required_columns):
            data["Date"] = pd.to_datetime(
                data["Year"].astype(str)
                + "-"
                + data["Month"].astype(str)
                + "-01",
                errors="coerce"
            )

            data["Consumption_kWh"] = pd.to_numeric(
                data["Consumption_kWh"],
                errors="coerce"
            )

            data = data.dropna(
                subset=["Date", "Consumption_kWh"]
            ).sort_values("Date")

            if not data.empty:
                st.subheader("Historical Monthly Consumption")

                fig = px.line(
                    data,
                    x="Date",
                    y="Consumption_kWh",
                    markers=True,
                    title="Monthly Electricity Consumption"
                )

                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Consumption (kWh)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.success(
                    "Your data was uploaded successfully!"
                )
            else:
                st.error(
                    "No valid consumption records were found."
                )
        else:
            st.error(
                "Please use these exact column names: "
                "Year, Month, Consumption_kWh."
            )

    except Exception as error:
        st.error(f"Unable to read the file: {error}")

else:
    st.info(
        "Upload a sample dataset to begin exploring "
        "monthly electricity consumption."
    )

st.divider()

st.caption(
    "Research prototype | Monthly Electrical Load Forecasting"
)
