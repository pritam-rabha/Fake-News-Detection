from flask import Flask, render_template, request, redirect
from pathlib import Path
import json
import os

from detector.config import Config
from detector.predictor import NewsDetector
from detector.data_loader import load_dataset
import hashlib

def file_signature(path):
    """Return a content hash so dataset changes are detected even if row count is unchanged."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_evaluation(model_dir):
    """Read measured model metrics; never fabricate dashboard values."""
    path = Path(model_dir) / "evaluation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            return None
        return payload
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    detector = NewsDetector(
    Path(app.config["DATASET_PATH"]),
    Path(app.config["MODEL_DIR"]),
    app.config.get("MODEL_NAME", "Linear SVM"),
    )
    app.extensions["news_detector"] = detector

    def render_home(result=None, error=None, status=200):
        return render_template(
            "index.html", result=result, error=error,
            model_name=detector.model_name, active_page="detector",
        ), status

    @app.get("/")
    def index():
        return render_home()

    @app.post("/predict")
    def predict():
        text = request.form.get("news_text", "").strip()
        if not text:
            return render_home(error="Please enter a news article.", status=400)
        if len(text) > app.config["MAX_NEWS_CHARS"]:
            return render_home(error=f"Input exceeds the {app.config['MAX_NEWS_CHARS']:,} character limit.", status=400)
        try:
            result = detector.predict(text)
            return render_home(result=result)
        except ValueError as exc:
            return render_home(error=str(exc), status=400)
        except Exception:
            app.logger.exception("Prediction failed")
            return render_home(error="Analysis could not be completed. Check the model and dataset configuration.", status=500)

    @app.get("/performance")
    def performance():
        dataset_path = Path(app.config["DATASET_PATH"])
        report = read_evaluation(app.config["MODEL_DIR"])
        dataset_rows = None
        label_counts = {}
        dataset_error = None
        try:
            live_dataset = load_dataset(dataset_path)
            dataset_rows = int(len(live_dataset))
            label_counts = {
                str(label).title(): int(count)
                for label, count in live_dataset["label"].value_counts().to_dict().items()
            }
        except Exception as exc:
            # Keep the dashboard accessible and show a clear dataset status.
            dataset_error = str(exc)

        current_signature = file_signature(dataset_path)
        evaluated_signature = report.get("dataset_signature") if report else None
        same_rows = bool(report and dataset_rows is not None and report.get("dataset_rows") == dataset_rows)
        same_signature = bool(current_signature and evaluated_signature and current_signature == evaluated_signature)
        # Older evaluation files lack a signature, so require one fresh training run
        # before claiming the scores match the current dataset.
        evaluation_current = bool(report and same_rows and same_signature)
        return render_template(
            "performance.html",
            report=report,
            model_name=detector.model_name,
            active_page="performance",
            current_dataset_rows=dataset_rows,
            current_label_counts=label_counts,
            dataset_error=dataset_error,
            evaluation_current=evaluation_current,
        )

    @app.get("/evaluation")
    def evaluation():
        return redirect("/performance", code=302)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": detector.model_name}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
