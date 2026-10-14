from pathlib import Path
import joblib
from .preprocessing import preprocess_text
from .training import train_all

class NewsDetector:
    def __init__(self, dataset_path, model_dir, model_name="Logistic Regression"):
        self.dataset_path, self.model_dir = Path(dataset_path), Path(model_dir)
        self.model_name = model_name
        self.model_path = self.model_dir / "best_model.joblib"
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                return
            except Exception:
                self.model_path.unlink(missing_ok=True)
        train_all(self.dataset_path, self.model_dir, tune=False)
        self.model = joblib.load(self.model_path)

    def predict(self, text):
        cleaned = preprocess_text(text)
        if not cleaned.strip():
            raise ValueError("The text contains no usable words after preprocessing.")
        label = str(self.model.predict([cleaned])[0])
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([cleaned])[0]
            confidence = float(probabilities[list(self.model.classes_).index(label)])
        return {"label": label, "label_display": "Likely Real" if label == "real" else "Likely Fake",
                "confidence": round(confidence * 100, 1) if confidence is not None else None,
                "model": self.model_name,
                "caveat": "This is a model prediction, not a verified fact-check. Verify important claims with reliable independent sources."}
