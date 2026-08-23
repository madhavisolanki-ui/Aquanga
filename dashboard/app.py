"""
Streamlit Web Dashboard for Aquanga
Predictive Water Monitoring & Early Warning System for Ganga River
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Ensure project root is prioritized in sys.path and remove dashboard dir to prevent package shadowing
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

dashboard_dir = os.path.abspath(os.path.dirname(__file__))
while dashboard_dir in sys.path:
    sys.path.remove(dashboard_dir)

# If 'app' in sys.modules is this script file instead of the package, reset it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.services.prediction_service import predictor
from app.services.risk_service import calculate_risk
from dashboard.components.map_view import render_ganga_map
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.charts import render_station_forecast_charts, render_basin_analytics
from dashboard.components.model_comparison import render_model_comparison_section

# Page Configuration
st.set_page_config(
    page_title="Aquanga | Ganga Water Monitoring & Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .risk-high { background-color: #feb2b2; color: #9b2c2c; }
    .risk-med { background-color: #feebc8; color: #9c4221; }
    .risk-low { background-color: #c6f6d5; color: #22543d; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_base_data():
    """Loads cleaned dataset and coordinates."""
    clean_csv_path = "data/processed/ganga_water_quality_clean.csv"
    coords_json_path = "data/external/station_coordinates.json"

    if not os.path.exists(clean_csv_path):
        from ml.preprocessing import run_preprocessing
        clean_df = run_preprocessing(output_path=clean_csv_path)
    else:
        clean_df = pd.read_csv(clean_csv_path)

    coords_data = {}
    if os.path.exists(coords_json_path):
        with open(coords_json_path, "r") as f:
            coords_data = json.load(f)

    return clean_df, coords_data


def generate_all_station_forecasts(clean_df: pd.DataFrame, coords_data: dict, model_name: str, forecast_year: int):
    """Generates forecasts and risk assessments for all available stations."""
    stations_list = sorted(clean_df["location"].unique())
    results = []

    for idx, st_name in enumerate(stations_list):
        st_df = clean_df[clean_df["location"] == st_name].sort_values(by="year")
        if len(st_df) == 0:
            continue

        latest = st_df.iloc[-1]
        second_latest = st_df.iloc[-2] if len(st_df) >= 2 else latest

        do_lag1 = float(latest["do"])
        bod_lag1 = float(latest["bod"]) if pd.notna(latest["bod"]) else 3.0
        fc_lag1 = float(latest["fecal_coliform"]) if pd.notna(latest["fecal_coliform"]) else 2500.0

        do_change = do_lag1 - float(second_latest["do"])
        bod_change = bod_lag1 - (float(second_latest["bod"]) if pd.notna(second_latest["bod"]) else 3.0)
        fc_change = fc_lag1 - (float(second_latest["fecal_coliform"]) if pd.notna(second_latest["fecal_coliform"]) else 2500.0)

        # Run prediction
        pred_res = predictor.predict(
            station_name=st_name,
            do_lag1=do_lag1,
            bod_lag1=bod_lag1,
            fecal_coliform_lag1=fc_lag1,
            do_change=do_change,
            bod_change=bod_change,
            fecal_coliform_change=fc_change,
            year=forecast_year,
            station_id=idx,
            model_name=model_name
        )

        meta = coords_data.get(st_name, {})
        results.append({
            "name": st_name,
            "station_id": idx,
            "latitude": meta.get("latitude", 25.5),
            "longitude": meta.get("longitude", 81.5),
            "state": meta.get("state", "Uttar Pradesh"),
            "district": meta.get("district", "Ganga Basin"),
            "predicted_do": pred_res["predicted_do"],
            "bod": bod_lag1,
            "fecal_coliform": fc_lag1,
            "risk_score": pred_res["risk_score"],
            "risk_level": pred_res["risk_level"],
            "warning": pred_res["warning"]
        })

    return results


def main():
    clean_df, coords_data = load_base_data()

    # Title & Branding
    st.markdown('<div class="main-header">🌊 Aquanga River Monitoring System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predictive Water Quality Forecasting & Central Pollution Control Board (CPCB) Early Warning Platform</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.image("https://img.icons8.com/color/96/water.png", width=64)
    st.sidebar.title("🎛️ Control Panel")

    all_stations = sorted(clean_df["location"].unique())
    selected_station = st.sidebar.selectbox(
        "📍 Select Monitoring Station",
        options=all_stations,
        index=0,
        help="Choose any station along the Ganga River basin to inspect historical trends and predictions."
    )

    selected_model = st.sidebar.selectbox(
        "🧠 Forecasting Model",
        options=["XGBoost", "Random Forest", "Linear Regression", "1D CNN", "LSTM", "CNN + LSTM", "best"],
        index=0,
        help="Select which trained ML or Deep Learning model architecture to use for predictions."
    )

    forecast_year = st.sidebar.number_input(
        "📅 Forecast Horizon (Year)",
        min_value=2016,
        max_value=2030,
        value=2016,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 CPCB Quality Standards")
    st.sidebar.markdown(r"""
    - **DO (Dissolved Oxygen)**: $\ge 5.0\text{ mg/L}$
    - **BOD (Organic Load)**: $\le 3.0\text{ mg/L}$
    - **Fecal Coliform**: $\le 2500\text{ MPN/100ml}$
    """)

    # Generate forecasts for all stations
    stations_data = generate_all_station_forecasts(
        clean_df=clean_df,
        coords_data=coords_data,
        model_name=selected_model,
        forecast_year=forecast_year
    )
    stations_df = pd.DataFrame(stations_data)

    # Top KPI Metrics Cards
    render_kpi_cards(stations_data)
    st.markdown("---")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Ganga Basin Monitoring Map",
        "📈 Station Forecasts & Analytics",
        "🔬 ML Model Benchmark Comparison",
        "🚨 Active Alerts & Warning Feed"
    ])

    with tab1:
        st.markdown(f"#### 🛰️ Real-Time Spatial Risk Map ({forecast_year} Forecast)")
        st.markdown("🟢 **Green**: Low Risk | 🟠 **Orange**: Medium Risk / Moderate Stress | 🔴 **Red**: High Risk / Critical Hypoxia")
        render_ganga_map(stations_data, selected_station=selected_station, height=520)

        st.markdown("---")
        # High Risk Stations Summary Table
        st.markdown("#### ⚠️ High & Medium Risk Monitoring Stations")
        elevated_df = stations_df[stations_df["risk_score"] >= 2][["name", "state", "predicted_do", "bod", "fecal_coliform", "risk_level", "warning"]]
        if not elevated_df.empty:
            st.dataframe(elevated_df, use_container_width=True)
        else:
            st.success("All stations currently meet safe CPCB water quality thresholds.")

    with tab2:
        # Station specific history and charts
        station_history = clean_df[clean_df["location"] == selected_station].sort_values(by="year")
        current_station_info = next((s for s in stations_data if s["name"] == selected_station), stations_data[0])

        st.markdown(f"### 📍 Station: **{selected_station}**")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.markdown(f"**State / Region:** {current_station_info.get('state', 'N/A')}")
        with col_meta2:
            st.markdown(f"**Forecasted DO ({forecast_year}):** `{current_station_info['predicted_do']:.2f} mg/L`")
        with col_meta3:
            st.markdown(f"**Risk Level:** `{current_station_info['risk_level']}` (Score: {current_station_info['risk_score']}/3)")

        render_station_forecast_charts(
            station_history=station_history,
            station_name=selected_station,
            forecast_do=current_station_info["predicted_do"],
            forecast_year=forecast_year
        )

        st.markdown("---")
        render_basin_analytics(stations_df, clean_df)

    with tab3:
        render_model_comparison_section()

    with tab4:
        st.markdown("### 🚨 Ganga Environmental Alert Feed & Warning System")
        st.markdown("Active early warnings generated using dynamic CPCB compliance penalty calculations:")

        for st_item in stations_data:
            score = st_item["risk_score"]
            if score >= 2:
                alert_type = "CRITICAL CONTAMINATION ALERT" if score == 3 else "MONITORING ADVISORY"
                alert_color = "error" if score == 3 else "warning"
                
                with st.expander(f"⚠️ {st_item['name']} — [{st_item['risk_level'].upper()} RISK]", expanded=(score == 3)):
                    st.write(f"**Status:** {st_item['warning']}")
                    st.write(f"**Predicted Dissolved Oxygen:** {st_item['predicted_do']:.2f} mg/L (Limit > 5.0 mg/L)")
                    st.write(f"**Observed BOD:** {st_item['bod']} mg/L (Limit < 3.0 mg/L)")
                    st.write(f"**Observed Fecal Coliform:** {st_item['fecal_coliform']} MPN/100ml (Limit < 2500 MPN/100ml)")
                    st.info("Recommended Action: Issue notice to local pollution control board and initiate biological/aeration treatment.")


if __name__ == "__main__":
    main()
