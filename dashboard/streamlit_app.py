"""
Streamlit Web Dashboard for Aquanga (Alternative Entrypoint)
Predictive Water Monitoring & Early Warning System for Ganga River
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dashboard.app import main

if __name__ == "__main__":
    main()
