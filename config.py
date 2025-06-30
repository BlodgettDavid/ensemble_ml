# config.py
# Authors: David Blodgett and Microsoft Copilot
# Description: Central configuration module for setting paths, logging, and shared constants.

import os
import logging
from datetime import datetime

# === Global Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure required directories exist
for path in [RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(path, exist_ok=True)

# === Logging Configuration ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOGS_DIR, f"experiment_log_{timestamp}.log")

# Create a named logger
logger = logging.getLogger("ensemble_logger")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    "%Y-%m-%d %H:%M:%S"
))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    "%Y-%m-%d %H:%M:%S"
))

# Attach handlers if they haven’t already been added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# === Shared Constants (expandable as needed) ===
RANDOM_SEED = 42