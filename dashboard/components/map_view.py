"""
Interactive Ganga Basin Map Component for Streamlit Dashboard
Uses Folium for high-performance geospatial visualization with CPCB risk coloring and rich popups.
"""

from typing import List, Dict, Any
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import streamlit as st


def get_risk_color(risk_level: str) -> str:
    """Maps risk level to map marker color."""
    level = str(risk_level).lower()
    if level == "high" or level == "critical":
        return "red"
    elif level == "medium" or level == "moderate":
        return "orange"
    else:
        return "green"


def render_ganga_map(stations_data: List[Dict[str, Any]], selected_station: str = None, height: int = 500):
    """
    Renders an interactive Folium map centered on the Ganga River Basin with color-coded risk markers.
    """
    # Center map on Uttar Pradesh / Ganga River stretch
    ganga_center = [26.2, 82.0]
    m = folium.Map(
        location=ganga_center,
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True
    )

    for st_info in stations_data:
        lat = st_info.get("latitude")
        lon = st_info.get("longitude")
        name = st_info.get("name", "Unknown Station")
        pred_do = st_info.get("predicted_do", 0.0)
        bod = st_info.get("bod", "N/A")
        fc = st_info.get("fecal_coliform", "N/A")
        risk_score = st_info.get("risk_score", 0)
        risk_level = st_info.get("risk_level", "Low")
        warning = st_info.get("warning", "Normal parameters.")

        if lat is None or lon is None:
            continue

        marker_color = get_risk_color(risk_level)
        is_selected = (selected_station and selected_station.strip() == name.strip())

        badge_bg = "#28a745" if marker_color == "green" else ("#fd7e14" if marker_color == "orange" else "#dc3545")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 240px; font-size: 13px;">
            <h4 style="margin: 0 0 8px 0; color: #1a365d; font-size: 14px; border-bottom: 2px solid #3182ce; padding-bottom: 4px;">
                {name}
            </h4>
            <div style="margin-bottom: 8px;">
                <span style="background-color: {badge_bg}; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 11px;">
                    {risk_level.upper()} RISK (Score: {risk_score}/3)
                </span>
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 8px;">
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 3px 0; color: #4a5568;"><b>Predicted DO:</b></td>
                    <td style="padding: 3px 0; text-align: right; color: #2b6cb0; font-weight: bold;">{pred_do:.2f} mg/L</td>
                </tr>
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 3px 0; color: #4a5568;"><b>BOD Level:</b></td>
                    <td style="padding: 3px 0; text-align: right;">{bod} mg/L</td>
                </tr>
                <tr style="border-bottom: 1px solid #edf2f7;">
                    <td style="padding: 3px 0; color: #4a5568;"><b>Fecal Coliform:</b></td>
                    <td style="padding: 3px 0; text-align: right;">{fc} MPN/100ml</td>
                </tr>
            </table>
            <div style="background-color: #f7fafc; padding: 6px; border-left: 3px solid {badge_bg}; font-size: 11px; color: #4a5568;">
                <b>Notice:</b> {warning}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=10 if is_selected else 7,
            color="#1a202c" if is_selected else marker_color,
            weight=3 if is_selected else 1.5,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85 if is_selected else 0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{name} ({risk_level} Risk)"
        ).add_to(m)

    st_folium(m, width="100%", height=height, returned_objects=[])
