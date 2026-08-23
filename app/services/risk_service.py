"""
Water Quality Risk Calculation Service for Aquanga
Implements Central Pollution Control Board (CPCB) standards for Dissolved Oxygen,
Biochemical Oxygen Demand, and Fecal Coliform.
"""

from typing import Dict, Any, Optional, Tuple


# Standard CPCB Water Quality Thresholds (Class B - Outdoor Bathing / Aquatic Life)
THRESHOLDS = {
    "DO": {
        "GOOD_MIN": 5.0,     # mg/L
        "MODERATE_MIN": 4.0   # mg/L
    },
    "BOD": {
        "GOOD_MAX": 3.0,     # mg/L
        "MODERATE_MAX": 6.0  # mg/L
    },
    "FECAL_COLIFORM": {
        "GOOD_MAX": 2500.0,      # MPN/100ml
        "MODERATE_MAX": 10000.0  # MPN/100ml
    }
}


def calculate_risk(
    do: float,
    bod: Optional[float] = None,
    fecal_coliform: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes environmental risk score (0-3), categorical risk level (Low, Medium, High),
    and contextual warning message based on CPCB water quality parameters.

    Args:
        do: Dissolved Oxygen in mg/L (target metric)
        bod: Biochemical Oxygen Demand in mg/L (optional)
        fecal_coliform: Fecal Coliform in MPN/100ml (optional)

    Returns:
        Dict with 'risk_score', 'risk_level', 'warning', and 'parameter_status'.
    """
    points = 0
    warning_reasons = []
    param_status = {}

    # 1. Evaluate Dissolved Oxygen (DO)
    if do < THRESHOLDS["DO"]["MODERATE_MIN"]:
        points += 2
        warning_reasons.append(f"Critical hypoxia (DO: {do:.2f} mg/L < 4.0)")
        param_status["do_status"] = "Critical"
    elif do < THRESHOLDS["DO"]["GOOD_MIN"]:
        points += 1
        warning_reasons.append(f"Oxygen stress (DO: {do:.2f} mg/L < 5.0)")
        param_status["do_status"] = "Moderate"
    else:
        param_status["do_status"] = "Good"

    # 2. Evaluate BOD (if available)
    if bod is not None:
        if bod > THRESHOLDS["BOD"]["MODERATE_MAX"]:
            points += 2
            warning_reasons.append(f"Heavy organic pollution (BOD: {bod:.2f} mg/L > 6.0)")
            param_status["bod_status"] = "Critical"
        elif bod > THRESHOLDS["BOD"]["GOOD_MAX"]:
            points += 1
            warning_reasons.append(f"Elevated organic load (BOD: {bod:.2f} mg/L > 3.0)")
            param_status["bod_status"] = "Moderate"
        else:
            param_status["bod_status"] = "Good"

    # 3. Evaluate Fecal Coliform (if available)
    if fecal_coliform is not None:
        if fecal_coliform > THRESHOLDS["FECAL_COLIFORM"]["MODERATE_MAX"]:
            points += 2
            warning_reasons.append(f"Severe microbial contamination (FC: {fecal_coliform:.0f} MPN/100ml > 10,000)")
            param_status["fc_status"] = "Critical"
        elif fecal_coliform > THRESHOLDS["FECAL_COLIFORM"]["GOOD_MAX"]:
            points += 1
            warning_reasons.append(f"Permissible coliform limit exceeded (FC: {fecal_coliform:.0f} MPN/100ml > 2,500)")
            param_status["fc_status"] = "Moderate"
        else:
            param_status["fc_status"] = "Good"

    # Derive Risk Score (0 - 3) and Level
    if points <= 1:
        risk_score = 1 if points == 1 else 0
        risk_level = "Low"
        warning = "Water quality within permissible CPCB standards. Regular monitoring advised."
    elif points in [2, 3]:
        risk_score = 2
        risk_level = "Medium"
        warning = f"Water quality requires monitoring: {'; '.join(warning_reasons)}."
    else:
        risk_score = 3
        risk_level = "High"
        warning = f"CRITICAL WARNING: {'; '.join(warning_reasons)}. Immediate intervention recommended."

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "warning": warning,
        "points": points,
        "parameter_status": param_status
    }
