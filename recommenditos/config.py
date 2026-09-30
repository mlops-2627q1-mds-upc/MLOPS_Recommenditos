from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# This repository's own `.env`, named rather than searched for. A bare
# `load_dotenv()` walks up the directory tree until it finds a file, so a clone
# nested under another checkout silently inherits that parent's credentials: the
# clone then looks configured when it is not, and a stale token two directories
# up would train against the wrong server. Naming the path also keeps "no
# tracking URI means tracking is off" falsifiable, which is what
# `recommenditos/tracking.py` relies on. A missing file is not an error.
load_dotenv(PROJ_ROOT / ".env")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

# The pipeline's configuration (see params.yaml). Kept here rather than in
# params.yaml itself so that no path in the project is machine-specific.
PARAMS_FILE = PROJ_ROOT / "params.yaml"
METRICS_FILE = PROJ_ROOT / "metrics.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
