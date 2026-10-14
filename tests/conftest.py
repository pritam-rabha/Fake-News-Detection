import sys
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import joblib

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def app(tmp_path):
    """Create a Flask test app using a lightweight test-only model."""
    from app import create_app

    dataset_path = ROOT / "data" / "news_dataset.csv"
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    # A tiny test-only classifier; never modifies production artifacts.
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", DummyClassifier(strategy="prior")),
    ])

    model.fit(
        [
            "government announces official news",
            "fake shocking viral claim",
            "official election information",
            "false misleading article",
        ],
        ["real", "fake", "real", "fake"],
    )

    joblib.dump(
        model,
        model_dir / "linear_svm.joblib",
    )

    return create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "DATASET_PATH": str(dataset_path),
        "MODEL_DIR": str(model_dir),
        "MODEL_NAME": "Linear SVM",
    })