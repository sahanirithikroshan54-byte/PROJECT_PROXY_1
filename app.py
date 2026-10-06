import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PJM Power Supply Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DASHBOARD THEME
# ============================================================
import base64

BACKGROUND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pjm_dashboard_background.png",
)

if os.path.exists(BACKGROUND_PATH):
    with open(BACKGROUND_PATH, "rb") as f:
        bg64 = base64.b64encode(f.read()).decode("utf-8")
else:
    bg64 = ""

st.markdown(
    f"""
    <style>
    /* ---------- App background ---------- */
    .stApp {{
        background:
            linear-gradient(
                90deg,
                rgba(2, 12, 28, 0.92) 0%,
                rgba(2, 18, 40, 0.80) 34%,
                rgba(2, 18, 40, 0.72) 100%
            ),
            url("data:image/png;base64,{bg64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ---------- Main content ---------- */
    .block-container {{
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: rgba(2, 12, 28, 0.96);
        border-right: 1px solid rgba(90, 190, 255, 0.18);
    }}

    [data-testid="stSidebar"] * {{
        color: #eef7ff;
    }}

    /* ---------- Titles ---------- */
    .main-title {{
        color: #f5fbff;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 12px rgba(0,0,0,0.45);
    }}

    .subtitle {{
        color: #c6d8e8;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }}

    .section-title {{
        color: #f5fbff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.1rem;
        margin-bottom: 0.6rem;
    }}

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {{
        background: rgba(5, 29, 57, 0.88);
        border: 1px solid rgba(76, 180, 255, 0.24);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.28);
        min-height: 115px;
    }}

    [data-testid="stMetricLabel"] {{
        color: #a9c4da !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #f7fbff !important;
        font-weight: 800 !important;
    }}

    /* ---------- Content panels ---------- */
    .dashboard-card {{
        background: rgba(5, 27, 52, 0.88);
        border: 1px solid rgba(76, 180, 255, 0.22);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }}

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: rgba(7, 35, 67, 0.94) !important;
        border-color: rgba(76, 180, 255, 0.3) !important;
        color: white !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(100, 200, 255, 0.35);
        background: linear-gradient(135deg, #087eea, #05a7d9);
        color: white;
        font-weight: 700;
        min-height: 45px;
        box-shadow: 0 7px 20px rgba(0, 130, 230, 0.25);
    }}

    .stButton > button:hover {{
        border-color: #79d9ff;
        transform: translateY(-1px);
    }}

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(76, 180, 255, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ---------- Info / alerts ---------- */
    .stAlert {{
        background: rgba(5, 29, 57, 0.88);
        border-radius: 12px;
    }}

    /* ---------- Download buttons ---------- */
    .stDownloadButton > button {{
        width: 100%;
        border-radius: 10px;
        background: rgba(7, 47, 88, 0.95);
        color: #eaf7ff;
        border: 1px solid rgba(76, 180, 255, 0.3);
    }}

    /* ---------- Sidebar radio ---------- */
    [data-testid="stSidebar"] .stRadio label {{
        padding: 7px 5px;
        border-radius: 8px;
    }}

    /* ---------- Mobile ---------- */
    @media (max-width: 900px) {{
        .main-title {{
            font-size: 1.65rem;
        }}
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pjm_xgboost_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "pjm_feature_columns.pkl")

# Support either CSV name used during development.
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "PJMW_hourly.csv"),
    os.path.join(BASE_DIR, "PJMW_MW_Hourly.csv"),
    os.path.join(BASE_DIR, "PJMW_MW_Hourly(1).csv"),
]


# ============================================================
# LOAD MODEL / FEATURES / DATA
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n\n"
            "Put pjm_xgboost_model.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_columns():
    if os.path.exists(FEATURES_PATH):
        return joblib.load(FEATURES_PATH)

    # Exact feature list used by the XGBoost section of PJM(1).ipynb
    return [
        "Hour",
        "DayOfWeek",
        "Month",
        "DayOfYear",
        "IsWeekend",
        "Lag1",
        "Lag24",
        "Lag168",
        "Rolling_24",
        "Rolling_168",
    ]


@st.cache_data
def load_data():
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)

    if csv_path is None:
        raise FileNotFoundError(
            "PJM CSV file not found. Put PJMW_hourly.csv in the same folder as app.py."
        )

    df = pd.read_csv(csv_path)

    # Accept a few common variations.
    datetime_col = next(
        (c for c in ["Datetime", "datetime", "DateTime", "date"] if c in df.columns),
        None,
    )
    target_col = next(
        (c for c in ["PJMW_MW", "PJMW MW", "MW", "Demand"] if c in df.columns),
        None,
    )

    if datetime_col is None or target_col is None:
        raise ValueError(
            "CSV must contain Datetime and PJMW_MW columns."
        )

    df = df[[datetime_col, target_col]].copy()
    df.columns = ["Datetime", "PJMW_MW"]

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["PJMW_MW"] = pd.to_numeric(df["PJMW_MW"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    # Match the notebook's data cleaning approach:
    # aggregate duplicate timestamps and create a complete hourly timeline.
    df = df.groupby("Datetime", as_index=False)["PJMW_MW"].mean()
    df = df.set_index("Datetime").sort_index()

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
    )
    df = df.reindex(full_index)
    df.index.name = "Datetime"

    df["PJMW_MW"] = df["PJMW_MW"].interpolate(method="time")
    df["PJMW_MW"] = df["PJMW_MW"].ffill().bfill()

    return df.reset_index()


model = load_model()
features = load_feature_columns()
raw_df = load_data()


# ============================================================
# FEATURE ENGINEERING
# Exact feature logic from PJM(1).ipynb XGBoost section
# ============================================================
def create_model_features(history_df):
    x = history_df.copy()
    x["Datetime"] = pd.to_datetime(x["Datetime"])
    x = x.sort_values("Datetime").reset_index(drop=True)

    x["Hour"] = x["Datetime"].dt.hour
    x["DayOfWeek"] = x["Datetime"].dt.dayofweek
    x["Month"] = x["Datetime"].dt.month
    x["DayOfYear"] = x["Datetime"].dt.dayofyear
    x["IsWeekend"] = (x["DayOfWeek"] >= 5).astype(int)

    x["Lag1"] = x["PJMW_MW"].shift(1)
    x["Lag24"] = x["PJMW_MW"].shift(24)
    x["Lag168"] = x["PJMW_MW"].shift(168)

    x["Rolling_24"] = (
        x["PJMW_MW"].shift(1).rolling(24).mean()
    )

    x["Rolling_168"] = (
        x["PJMW_MW"].shift(1).rolling(168).mean()
    )

    return x


def recursive_forecast(history_df, horizon_hours):
    """
    Recursive multi-step forecast using the trained XGBoost model.
    The model predicts one hour at a time and each prediction becomes
    available as historical demand for later forecast steps.
    """
    history = history_df[["Datetime", "PJMW_MW"]].copy()
    history["Datetime"] = pd.to_datetime(history["Datetime"])
    history = history.sort_values("Datetime").reset_index(drop=True)

    predictions = []

    for _ in range(horizon_hours):
        next_dt = history["Datetime"].iloc[-1] + pd.Timedelta(hours=1)

        temp = pd.concat(
            [
                history,
                pd.DataFrame(
                    {"Datetime": [next_dt], "PJMW_MW": [np.nan]}
                ),
            ],
            ignore_index=True,
        )

        # Build features using known history.
        temp["Hour"] = temp["Datetime"].dt.hour
        temp["DayOfWeek"] = temp["Datetime"].dt.dayofweek
        temp["Month"] = temp["Datetime"].dt.month
        temp["DayOfYear"] = temp["Datetime"].dt.dayofyear
        temp["IsWeekend"] = (temp["DayOfWeek"] >= 5).astype(int)

        temp["Lag1"] = temp["PJMW_MW"].shift(1)
        temp["Lag24"] = temp["PJMW_MW"].shift(24)
        temp["Lag168"] = temp["PJMW_MW"].shift(168)

        temp["Rolling_24"] = (
            temp["PJMW_MW"].shift(1).rolling(24).mean()
        )
        temp["Rolling_168"] = (
            temp["PJMW_MW"].shift(1).rolling(168).mean()
        )

        row = temp.iloc[[-1]][features]

        if row.isna().any().any():
            raise ValueError(
                "Not enough historical observations to create forecast features."
            )

        pred = float(model.predict(row)[0])

        predictions.append(
            {
                "Datetime": next_dt,
                "Forecast_MW": pred,
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {"Datetime": [next_dt], "PJMW_MW": [pred]}
                ),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


# ============================================================
# PLOTLY DARK THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(4,22,42,0.62)",
    font=dict(color="#e8f5ff"),
    title=dict(font=dict(color="#f5fbff", size=20)),
    xaxis=dict(
        color="#bcd2e5",
        gridcolor="rgba(130,190,230,0.14)",
        zerolinecolor="rgba(130,190,230,0.18)",
    ),
    yaxis=dict(
        color="#bcd2e5",
        gridcolor="rgba(130,190,230,0.14)",
        zerolinecolor="rgba(130,190,230,0.18)",
    ),
    margin=dict(l=55, r=25, t=60, b=50),
    hovermode="x unified",
)

# ============================================================
# EDA HELPERS
# ============================================================
def add_calendar_features(df):
    x = df.copy()
    x["Hour"] = x["Datetime"].dt.hour
    x["DayOfWeek"] = x["Datetime"].dt.dayofweek
    x["DayName"] = x["Datetime"].dt.day_name()
    x["Month"] = x["Datetime"].dt.month
    x["MonthName"] = x["Datetime"].dt.month_name()
    x["DayOfYear"] = x["Datetime"].dt.dayofyear
    x["Week"] = x["Datetime"].dt.isocalendar().week.astype(int)
    x["IsWeekend"] = (x["DayOfWeek"] >= 5).astype(int)

    def season(month):
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    x["Season"] = x["Month"].apply(season)
    return x


eda_df = add_calendar_features(raw_df)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚡ PJM Forecasting")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Historical Analysis",
        "🔍 Demand Patterns",
        "🤖 Model Performance",
        "🔮 Forecast",
        "⚡ Peak Demand",
        "📥 Downloads",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("PJM Hourly Energy Consumption Forecast")
st.sidebar.caption("XGBoost time-series deployment")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">⚡ PJM Power Supply Forecasting</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Hourly electricity demand analysis and XGBoost forecasting dashboard</div>',
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "🏠 Overview":

    latest = raw_df.iloc[-1]["PJMW_MW"]
    avg_demand = raw_df["PJMW_MW"].mean()
    max_demand = raw_df["PJMW_MW"].max()
    min_demand = raw_df["PJMW_MW"].min()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest Demand", f"{latest:,.0f} MW")
    c2.metric("Average Demand", f"{avg_demand:,.0f} MW")
    c3.metric("Maximum Demand", f"{max_demand:,.0f} MW")
    c4.metric("Minimum Demand", f"{min_demand:,.0f} MW")

    st.markdown('<div class="section-title">Historical Demand</div>', unsafe_allow_html=True)

    fig = px.line(
        raw_df,
        x="Datetime",
        y="PJMW_MW",
        title="PJM Hourly Electricity Demand",
        labels={"PJMW_MW": "Demand (MW)", "Datetime": "Date"},
    )
    fig.update_layout(hovermode="x unified")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Dataset period: {raw_df['Datetime'].min():%d %b %Y} "
        f"to {raw_df['Datetime'].max():%d %b %Y}"
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================
elif page == "📊 Historical Analysis":

    st.subheader("📊 Historical Analysis")

    min_date = raw_df["Datetime"].min().date()
    max_date = raw_df["Datetime"].max().date()

    selected_range = st.date_input(
        "Select date range",
        value=(max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=30)).date()), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    mask = (
        (raw_df["Datetime"].dt.date >= start_date)
        & (raw_df["Datetime"].dt.date <= end_date)
    )
    selected = raw_df.loc[mask].copy()

    view = st.selectbox(
        "View",
        ["Hourly", "Daily", "Weekly", "Monthly"],
    )

    if view == "Hourly":
        plot_df = selected.copy()
        y_col = "PJMW_MW"
        title = "Hourly Demand"

    elif view == "Daily":
        plot_df = (
            selected.set_index("Datetime")["PJMW_MW"]
            .resample("D")
            .mean()
            .reset_index()
        )
        y_col = "PJMW_MW"
        title = "Daily Average Demand"

    elif view == "Weekly":
        plot_df = (
            selected.set_index("Datetime")["PJMW_MW"]
            .resample("W")
            .mean()
            .reset_index()
        )
        y_col = "PJMW_MW"
        title = "Weekly Average Demand"

    else:
        plot_df = (
            selected.set_index("Datetime")["PJMW_MW"]
            .resample("MS")
            .mean()
            .reset_index()
        )
        y_col = "PJMW_MW"
        title = "Monthly Average Demand"

    fig = px.line(
        plot_df,
        x="Datetime",
        y=y_col,
        title=title,
        labels={"PJMW_MW": "Demand (MW)", "Datetime": "Date"},
    )
    fig.update_layout(hovermode="x unified")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(selected.tail(500), use_container_width=True)


# ============================================================
# DEMAND PATTERNS
# ============================================================
elif page == "🔍 Demand Patterns":

    st.subheader("🔍 Demand Patterns")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Hourly", "Day of Week", "Monthly", "Summer vs Winter"]
    )

    with tab1:
        hourly = (
            eda_df.groupby("Hour")["PJMW_MW"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            hourly,
            x="Hour",
            y="PJMW_MW",
            title="Average Demand by Hour",
            labels={"PJMW_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dow = (
            eda_df.groupby(["DayOfWeek", "DayName"])["PJMW_MW"]
            .mean()
            .reset_index()
            .sort_values("DayOfWeek")
        )
        fig = px.bar(
            dow,
            x="DayName",
            y="PJMW_MW",
            title="Average Demand by Day of Week",
            labels={"PJMW_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        monthly = (
            eda_df.groupby(["Month", "MonthName"])["PJMW_MW"]
            .mean()
            .reset_index()
            .sort_values("Month")
        )
        fig = px.bar(
            monthly,
            x="MonthName",
            y="PJMW_MW",
            title="Average Demand by Month",
            labels={"PJMW_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        season_hourly = (
            eda_df[eda_df["Season"].isin(["Summer", "Winter"])]
            .groupby(["Season", "Hour"])["PJMW_MW"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            season_hourly,
            x="Hour",
            y="PJMW_MW",
            color="Season",
            markers=True,
            title="Summer vs Winter Hourly Demand",
            labels={"PJMW_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":

    st.subheader("🤖 XGBoost Model Performance")

    # The notebook reported these values for the XGBoost model.
    # Recalculate on the last-year test set when possible.
    test_start = raw_df["Datetime"].max() - pd.Timedelta(days=365)

    model_df = create_model_features(raw_df).dropna().reset_index(drop=True)
    test_ml = model_df[model_df["Datetime"] >= test_start].copy()

    if len(test_ml) > 0:
        X_test = test_ml[features]
        y_test = test_ml["PJMW_MW"]
        pred = model.predict(X_test)

        mae = float(np.mean(np.abs(y_test - pred)))
        rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
        mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)

        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:,.2f} MW")
        c2.metric("RMSE", f"{rmse:,.2f} MW")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("R²", f"{r2:.4f}")

        comparison = pd.DataFrame(
            {
                "Datetime": test_ml["Datetime"].values,
                "Actual": y_test.values,
                "XGBoost": pred,
            }
        )

        st.markdown("### Actual vs Predicted — First Week of Test Set")

        first_week = comparison.head(168)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=first_week["Datetime"],
                y=first_week["Actual"],
                mode="lines",
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=first_week["Datetime"],
                y=first_week["XGBoost"],
                mode="lines",
                name="XGBoost",
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Date",
            yaxis_title="Demand (MW)",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Importance")

        importance = pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="XGBoost Feature Importance",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Could not construct the last-year test set.")


# ============================================================
# FORECAST
# ============================================================
elif page == "🔮 Forecast":

    st.subheader("🔮 Future Demand Forecast")

    horizon_days = st.selectbox(
        "Forecast horizon",
        [7, 15, 30],
        index=2,
        format_func=lambda x: f"{x} days",
    )

    horizon_hours = horizon_days * 24

    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating recursive XGBoost forecast..."):
            forecast_df = recursive_forecast(raw_df, horizon_hours)

        st.session_state["forecast_df"] = forecast_df
        st.success(
            f"Generated {len(forecast_df):,} hourly predictions "
            f"for the next {horizon_days} days."
        )

    if "forecast_df" in st.session_state:
        forecast_df = st.session_state["forecast_df"].copy()

        # If the user changes the horizon after generating, regenerate.
        if len(forecast_df) != horizon_hours:
            with st.spinner("Updating forecast..."):
                forecast_df = recursive_forecast(raw_df, horizon_hours)
            st.session_state["forecast_df"] = forecast_df

        peak_row = forecast_df.loc[forecast_df["Forecast_MW"].idxmax()]
        min_row = forecast_df.loc[forecast_df["Forecast_MW"].idxmin()]
        avg_forecast = forecast_df["Forecast_MW"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Average", f"{avg_forecast:,.0f} MW")
        c2.metric("Forecast Peak", f"{peak_row['Forecast_MW']:,.0f} MW")
        c3.metric("Forecast Minimum", f"{min_row['Forecast_MW']:,.0f} MW")

        st.markdown("### Hourly Forecast")

        fig = px.line(
            forecast_df,
            x="Datetime",
            y="Forecast_MW",
            title=f"Next {horizon_days} Days — Hourly Forecast",
            labels={
                "Forecast_MW": "Forecast Demand (MW)",
                "Datetime": "Date",
            },
        )
        fig.update_layout(hovermode="x unified")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Daily aggregation
        daily = (
            forecast_df.assign(Date=forecast_df["Datetime"].dt.date)
            .groupby("Date")
            .agg(
                Average_Demand_MW=("Forecast_MW", "mean"),
                Peak_Demand_MW=("Forecast_MW", "max"),
                Minimum_Demand_MW=("Forecast_MW", "min"),
                Total_MWh=("Forecast_MW", "sum"),
            )
            .reset_index()
        )

        st.markdown("### Daily Forecast")

        fig = px.line(
            daily,
            x="Date",
            y="Average_Demand_MW",
            markers=True,
            title="Daily Average Forecast",
            labels={"Average_Demand_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Weekly aggregation
        weekly = (
            forecast_df.set_index("Datetime")["Forecast_MW"]
            .resample("W-MON", label="left", closed="left")
            .agg(
                Average_Demand_MW="mean",
                Peak_Demand_MW="max",
                Minimum_Demand_MW="min",
                Total_MWh="sum",
                Hours="count",
            )
            .reset_index()
        )

        weekly["Week"] = [
            f"Week {i + 1}" for i in range(len(weekly))
        ]

        st.markdown("### 📅 Weekly Predictions")

        w1, w2, w3 = st.columns(3)
        w1.metric(
            "Weekly Avg (overall)",
            f"{weekly['Average_Demand_MW'].mean():,.0f} MW",
        )
        w2.metric(
            "Highest Weekly Peak",
            f"{weekly['Peak_Demand_MW'].max():,.0f} MW",
        )
        w3.metric(
            "Forecast Weeks",
            f"{len(weekly)}",
        )

        fig = px.bar(
            weekly,
            x="Week",
            y="Average_Demand_MW",
            title="Weekly Average Demand Forecast",
            labels={"Average_Demand_MW": "Average Demand (MW)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            weekly[
                [
                    "Week",
                    "Datetime",
                    "Average_Demand_MW",
                    "Peak_Demand_MW",
                    "Minimum_Demand_MW",
                    "Total_MWh",
                    "Hours",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("### Full Hourly Forecast")
        st.dataframe(forecast_df, use_container_width=True)

    else:
        st.info("Click **Generate Forecast** to create hourly, daily, and weekly predictions.")


# ============================================================
# PEAK DEMAND
# ============================================================
elif page == "⚡ Peak Demand":

    st.subheader("⚡ Peak Demand Analysis")

    if "forecast_df" not in st.session_state:
        with st.spinner("Generating the 30-day forecast for peak analysis..."):
            st.session_state["forecast_df"] = recursive_forecast(raw_df, 30 * 24)

    forecast_df = st.session_state["forecast_df"].copy()

    top_peaks = forecast_df.nlargest(10, "Forecast_MW").copy()
    top_peaks["Rank"] = range(1, len(top_peaks) + 1)

    peak = top_peaks.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Peak", f"{peak['Forecast_MW']:,.0f} MW")
    c2.metric("Peak Date", peak["Datetime"].strftime("%d %b %Y"))
    c3.metric("Peak Time", peak["Datetime"].strftime("%I:%M %p"))

    st.markdown("### Top 10 Forecast Peak Periods")

    st.dataframe(
        top_peaks[
            ["Rank", "Datetime", "Forecast_MW"]
        ].rename(columns={"Forecast_MW": "Forecast Demand (MW)"}),
        use_container_width=True,
    )

    fig = px.bar(
        top_peaks.sort_values("Forecast_MW"),
        x="Forecast_MW",
        y="Datetime",
        orientation="h",
        title="Top 10 Predicted Peak Hours",
        labels={"Forecast_MW": "Demand (MW)"},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DOWNLOADS
# ============================================================
elif page == "📥 Downloads":

    st.subheader("📥 Download Forecast Data")

    if "forecast_df" not in st.session_state:
        with st.spinner("Generating the 30-day forecast..."):
            st.session_state["forecast_df"] = recursive_forecast(raw_df, 30 * 24)

    forecast_df = st.session_state["forecast_df"].copy()

    daily = (
        forecast_df.assign(Date=forecast_df["Datetime"].dt.date)
        .groupby("Date")
        .agg(
            Average_Demand_MW=("Forecast_MW", "mean"),
            Peak_Demand_MW=("Forecast_MW", "max"),
            Minimum_Demand_MW=("Forecast_MW", "min"),
            Total_MWh=("Forecast_MW", "sum"),
        )
        .reset_index()
    )

    weekly = (
        forecast_df.set_index("Datetime")["Forecast_MW"]
        .resample("W-MON", label="left", closed="left")
        .agg(
            Average_Demand_MW="mean",
            Peak_Demand_MW="max",
            Minimum_Demand_MW="min",
            Total_MWh="sum",
            Hours="count",
        )
        .reset_index()
    )

    st.download_button(
        "⬇️ Download Hourly Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="pjm_hourly_forecast.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Daily Forecast CSV",
        data=daily.to_csv(index=False).encode("utf-8"),
        file_name="pjm_daily_forecast.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Weekly Forecast CSV",
        data=weekly.to_csv(index=False).encode("utf-8"),
        file_name="pjm_weekly_forecast.csv",
        mime="text/csv",
    )

    st.markdown("### Weekly Forecast Preview")
    st.dataframe(weekly, use_container_width=True)

