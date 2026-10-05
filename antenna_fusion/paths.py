"""
Project paths
=============
Every asset and output location is resolved from the repository root, so the
apps behave the same no matter which directory they are launched from.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
CAD_DIR    = os.path.join(ASSETS_DIR, 'cad')
IMG_DIR    = os.path.join(ASSETS_DIR, 'img')
FONTS_DIR  = os.path.join(ASSETS_DIR, 'fonts')   # optional: drop Geologica / JetBrains Mono here
DATA_DIR   = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR   = os.path.join(PROJECT_ROOT, 'logs')  # runtime CSV recordings (gitignored)

APP_ICON_PATH = os.path.join(IMG_DIR, 'app_icon.png')

# Filename patterns of the CSVs written by core.py and csv_export.py
SIM_LOG_PATTERN    = 'sim_antenna_*.csv'
TIMESERIES_PATTERN = 'sim_timeseries_*.csv'


def find_latest_log(*patterns):
    """Newest file in LOGS_DIR matching the first pattern that has any hits."""
    import glob
    for pat in patterns:
        candidates = glob.glob(os.path.join(LOGS_DIR, pat))
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None
