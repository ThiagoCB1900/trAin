"""
trAIn Health - Configuration Module
====================================
Centralized configuration for the Clinical ML Studio application.
"""

from pathlib import Path
from typing import Final

# Application Metadata
APP_NAME: Final[str] = "trAIn Health"
APP_VERSION: Final[str] = "1.0.0"
APP_SUBTITLE: Final[str] = "Clinical ML Studio"
APP_DESCRIPTION: Final[str] = (
    "Plataforma de experimentação em ML para saúde com foco em "
    "reprodutibilidade e governança."
)

# Paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent
LITERATURE_DIR: Final[Path] = PROJECT_ROOT / "literature"
HISTORY_FILE: Final[Path] = PROJECT_ROOT / "history.json"
DATA_DIR: Final[Path] = PROJECT_ROOT / "sample_data"

# Default Experiment Parameters
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_SCALER: Final[str] = "None"
DEFAULT_SAMPLER: Final[str] = "None"
MAX_MODELS: Final[int] = 5

# UI Settings
WINDOW_TITLE: Final[str] = f"{APP_NAME} • {APP_SUBTITLE}"
WINDOW_WIDTH: Final[int] = 1200
WINDOW_HEIGHT: Final[int] = 850
SIDEBAR_WIDTH: Final[int] = 320

# Theme Colors - Dark Mode
DARK_THEME: Final[dict] = {
    "window": "#0f1720",
    "surface": "#14212b",
    "surface_alt": "#1a2b38",
    "panel": "#12202a",
    "border": "#2d4152",
    "text": "#e7f2ef",
    "muted": "#a9c1bc",
    "accent": "#49b9a6",
    "accent_hover": "#5ecab7",
    "accent_pressed": "#3ba793",
    "accent_soft": "#1f4b4d",
    "accent_light": "#6dd4c0",
    "danger": "#d16b6b",
    "shadow": "rgba(0, 0, 0, 0.3)",
}

# Theme Colors - Light Mode
LIGHT_THEME: Final[dict] = {
    "window": "#f4f8f7",
    "surface": "#ffffff",
    "surface_alt": "#edf5f2",
    "panel": "#eef6f3",
    "border": "#c8dbd5",
    "text": "#1f2d2a",
    "muted": "#5f7370",
    "accent": "#2f8f83",
    "accent_hover": "#3ca094",
    "accent_pressed": "#287b71",
    "accent_soft": "#d8efea",
    "accent_light": "#4db3a3",
    "danger": "#b95858",
    "shadow": "rgba(0, 0, 0, 0.08)",
}

# Model Configuration
SCALER_OPTIONS: Final[list] = ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
SAMPLER_OPTIONS: Final[list] = ["None", "RandomOverSampler", "RandomUnderSampler", "SMOTE"]

# Logging
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
