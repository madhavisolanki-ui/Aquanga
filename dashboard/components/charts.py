"""
Visualizations & Charts Component for Streamlit Dashboard
Renders DO forecasts, BOD & Fecal Coliform trends, Risk Score distributions, and correlation heatmaps.
"""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def render_station_forecast_charts(station_history: pd.DataFrame, station_name: str, forecast_do: float, forecast_year: int = 2016):
    """Renders DO, BOD, and Fecal Coliform time-series charts for selected station."""
    st.subheader(f"📈 Temporal Parameter Trends: {station_name}")

    if station_history.empty:
        st.info("No historical observations available for this station.")
        return

    # Prepare historical + forecast series
    df_plot = station_history.sort_values(by="year").copy()

    col1, col2 = st.columns(2)

    with col1:
        # DO Chart with CPCB Threshold line
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(df_plot["year"], df_plot["do"], marker="o", color="#2b6cb0", linewidth=2.2, label="Historical DO")
        
        # Add forecast point
        last_year = df_plot["year"].max()
        last_do = df_plot.loc[df_plot["year"] == last_year, "do"].values[0]
        ax.plot([last_year, forecast_year], [last_do, forecast_do], linestyle="--", marker="s", color="#e53e3e", linewidth=2, label=f"Forecast DO ({forecast_year})")
        
        # CPCB Standard line
        ax.axhline(5.0, color="#38a169", linestyle=":", linewidth=1.8, label="CPCB Good Limit (≥5 mg/L)")
        ax.axhline(4.0, color="#dd6b20", linestyle=":", linewidth=1.5, label="CPCB Hypoxic Limit (<4 mg/L)")

        ax.set_title("Dissolved Oxygen (DO) & Forecast (mg/L)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("DO (mg/L)")
        ax.set_xticks(sorted(list(df_plot["year"].unique()) + [forecast_year]))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # BOD Chart with CPCB Threshold line
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(df_plot["year"], df_plot["bod"], marker="o", color="#d69e2e", linewidth=2.2, label="Historical BOD")
        ax.axhline(3.0, color="#38a169", linestyle=":", linewidth=1.8, label="CPCB Clean Limit (≤3 mg/L)")
        ax.axhline(6.0, color="#e53e3e", linestyle=":", linewidth=1.5, label="CPCB High Pollution (>6 mg/L)")

        ax.set_title("Biochemical Oxygen Demand (BOD) (mg/L)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("BOD (mg/L)")
        ax.set_xticks(sorted(list(df_plot["year"].unique())))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Fecal Coliform Trend Chart
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(df_plot["year"], df_plot["fecal_coliform"], marker="o", color="#805ad5", linewidth=2.2, label="Fecal Coliform")
        ax.axhline(2500, color="#38a169", linestyle=":", linewidth=1.8, label="CPCB Standard (≤2500 MPN/100ml)")

        ax.set_title("Fecal Coliform Microbial Contamination (MPN/100ml)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("MPN/100ml")
        ax.set_yscale("log")
        ax.set_xticks(sorted(list(df_plot["year"].unique())))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        # Parameter Status Summary Bar
        fig, ax = plt.subplots(figsize=(7, 3.8))
        params = ["Forecast DO", "Latest BOD", "Latest FC / 1000"]
        latest_bod = df_plot.iloc[-1]["bod"]
        latest_fc_scaled = df_plot.iloc[-1]["fecal_coliform"] / 1000.0
        values = [forecast_do, latest_bod, latest_fc_scaled]
        colors = ["#2b6cb0", "#d69e2e", "#805ad5"]

        bars = ax.bar(params, values, color=colors, width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title("Current Water Quality Profile", fontsize=11, fontweight="bold")
        ax.set_ylabel("Magnitude (mg/L or FC/1000)")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_basin_analytics(all_stations_df: pd.DataFrame, clean_df: pd.DataFrame):
    """Renders basin-wide risk distribution and parameter correlation heatmap."""
    st.subheader("🌐 Basin-Wide Risk Distribution & Parameter Analytics")
    col1, col2 = st.columns(2)

    with col1:
        # Risk level distribution pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_counts = all_stations_df["risk_level"].value_counts()
        colors = {"Low": "#48bb78", "Medium": "#ed8936", "High": "#f56565"}
        plot_colors = [colors.get(k, "#a0aec0") for k in risk_counts.index]

        ax.pie(
            risk_counts.values,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=plot_colors,
            textprops={"fontsize": 10, "fontweight": "bold"}
        )
        ax.set_title("Basin Risk Level Distribution", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Correlation Heatmap across water quality parameters
        fig, ax = plt.subplots(figsize=(6, 4))
        corr_cols = ["do", "bod", "fecal_coliform"]
        corr_matrix = clean_df[corr_cols].corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            ax=ax,
            annot_kws={"fontsize": 10, "fontweight": "bold"}
        )
        ax.set_title("Water Quality Parameter Correlation Matrix", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
