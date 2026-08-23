"""
KPI Overview Metrics Component for Streamlit Dashboard
"""

from typing import List, Dict, Any
import streamlit as st


def render_kpi_cards(stations_data: List[Dict[str, Any]]):
    """Renders prominent metric cards showing overall basin water quality status."""
    total_stations = len(stations_data)
    if total_stations == 0:
        st.warning("No station data available.")
        return

    do_values = [s["predicted_do"] for s in stations_data if "predicted_do" in s and s["predicted_do"] is not None]
    bod_values = [s["bod"] for s in stations_data if "bod" in s and s["bod"] is not None and isinstance(s["bod"], (int, float))]
    high_risk_count = sum(1 for s in stations_data if s.get("risk_level", "").lower() in ["high", "critical"])
    med_risk_count = sum(1 for s in stations_data if s.get("risk_level", "").lower() in ["medium", "moderate"])

    avg_do = sum(do_values) / len(do_values) if do_values else 0.0
    avg_bod = sum(bod_values) / len(bod_values) if bod_values else 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Monitored Stations",
            value=f"{total_stations}",
            help="Total active Ganga River monitoring stations in network."
        )

    with col2:
        do_delta = f"{avg_do - 5.0:+.2f} vs CPCB limit (>5 mg/L)"
        st.metric(
            label="Mean Forecasted DO",
            value=f"{avg_do:.2f} mg/L",
            delta=do_delta,
            delta_color="normal" if avg_do >= 5.0 else "inverse",
            help="Dissolved Oxygen (CPCB threshold: minimum 5.0 mg/L)"
        )

    with col3:
        bod_delta = f"{avg_bod - 3.0:+.2f} vs CPCB limit (<3 mg/L)"
        st.metric(
            label="Mean BOD Level",
            value=f"{avg_bod:.2f} mg/L",
            delta=bod_delta,
            delta_color="inverse" if avg_bod > 3.0 else "normal",
            help="Biochemical Oxygen Demand (CPCB threshold: maximum 3.0 mg/L)"
        )

    with col4:
        st.metric(
            label="Elevated Risk Stations",
            value=f"{high_risk_count + med_risk_count} / {total_stations}",
            delta=f"{high_risk_count} Critical" if high_risk_count > 0 else "0 Critical",
            delta_color="inverse" if high_risk_count > 0 else "off",
            help="Stations currently triggering Medium or High risk warning notices."
        )
