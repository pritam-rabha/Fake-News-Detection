import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-this-key")
    MAX_NEWS_CHARS = int(os.environ.get("MAX_NEWS_CHARS", "5000"))
    DATASET_PATH = os.environ.get("DATASET_PATH", str(ROOT / "data" / "news_dataset.csv"))
    MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "artifacts"))
    MODEL_NAME = os.environ.get("MODEL_NAME", "Linear SVM")
