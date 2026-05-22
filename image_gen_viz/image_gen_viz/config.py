from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
