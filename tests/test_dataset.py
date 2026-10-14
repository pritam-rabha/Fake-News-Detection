from pathlib import Path
import pandas as pd
import pytest
from detector.data_loader import load_dataset
def test_bundled_dataset_has_both_labels():
    p = Path(__file__).resolve().parents[1] / "data" / "news_dataset.csv"
    df = load_dataset(p)
    assert set(df.label) == {"real", "fake"} and len(df) >= 40
def test_bad_dataset_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"headline": ["x"], "category": ["fake"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_dataset(p)
