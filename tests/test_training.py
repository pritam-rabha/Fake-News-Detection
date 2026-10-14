from pathlib import Path
from detector.training import train_all
def test_training_without_tuning_creates_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = train_all(root / "data" / "news_dataset.csv", tmp_path, tune=False)
    assert result["selected_model"] in {"Logistic Regression", "Multinomial Naive Bayes", "Linear SVM", "Random Forest"}
    assert (tmp_path / "best_model.joblib").exists()
    assert (tmp_path / "evaluation.json").exists()
