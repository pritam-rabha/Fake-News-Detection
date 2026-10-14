import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app import create_app
@pytest.fixture
def app(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return create_app({"TESTING": True, "SECRET_KEY": "test-secret",
        "DATASET_PATH": str(root / "data" / "news_dataset.csv"),
        "MODEL_DIR": str(tmp_path / "artifacts")})
