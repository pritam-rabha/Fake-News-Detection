# Fake News Detection — News Text Classification

A Flask web application for classifying news-like text with a dedicated Home page and a separate Model Performance page. The Home page supports pasted text and TXT, MD, CSV, PDF, and DOCX uploads. It includes text preprocessing, TF-IDF feature extraction, four classifiers, hyperparameter tuning, standard evaluation metrics, and a browser interface.

> **Important:** This is an educational ML demonstration, not a fact-checking authority. The bundled dataset contains illustrative examples created for software demonstration. It is not a validated news corpus and cannot support real-world accuracy claims. Replace it with a properly sourced, licensed, representative dataset before evaluating practical usefulness.

## Project information
- **Author:** Pritam Rabha
- **Programme:** MCA, 3rd Semester
- **Course:** Python Programming
- **Project Guide:** Dr. Pangambam Sendash Singh
- **Institution:** Dr. Hari Singh Gour Vishwavidyalaya, Sagar, Madhya Pradesh
- **Copyright:** Copyright © 2026 Pritam Rabha. All rights reserved.

This is an independent academic project, not an official university application or an endorsement by the institution.

## Assignment requirements
1. **Text preprocessing:** lowercasing, tokenization, stopword removal, stemming, and lemmatization when NLTK WordNet is available.
2. **TF-IDF:** word unigrams and bigrams.
3. **Multiple models:** Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest.
4. **Hyperparameter tuning:** GridSearchCV with stratified cross-validation.
5. **Evaluation:** accuracy, macro precision, macro recall, macro F1, confusion matrix, and classification report.
6. **Deployment:** Flask, Gunicorn, and Render configuration.

## Structure
```text
Fake_News_Detection/
├── app.py
├── train_model.py
├── requirements.txt
├── requirements-dev.txt
├── render.yaml
├── gunicorn.conf.py
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_REPORT.md
├── detector/
├── data/news_dataset.csv
├── templates/
├── static/css/style.css
├── static/js/app.js
├── static/images/newspaper-background.png
└── tests/
```

## Setup on Ubuntu
```bash
cd Fake_News_Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
For full WordNet lemmatization, download the NLTK resource once:
```bash
python -m nltk.downloader wordnet
```
If WordNet is unavailable, the preprocessor continues with stemming.

## Train and evaluate
Quick baseline:
```bash
python train_model.py --no-tune
```
Compare models with hyperparameter tuning:
```bash
python train_model.py
```
The training script saves model files and `evaluation.json` in `artifacts/`. The metrics are for a held-out split and do not establish real-world fact-checking accuracy. The included data is illustrative and too small for meaningful claims.

## Run the web app
```bash
python app.py
```
Open <http://127.0.0.1:5000>. The app also exposes `/performance` for model metrics and `/health` for service status. The legacy `/evaluation` route redirects to `/performance`. If no model artifact exists, the app trains a baseline model at first startup; training can take time.

## Dataset format
Use a CSV with `text` and `label` columns; labels must be `real` or `fake`:
```csv
text,label
"Example article text","real"
"Another example text","fake"
```
Use a sourced, licensed, representative dataset for actual evaluation. Do not treat the included illustrative examples as verified news.

## Metrics
Accuracy, macro precision, macro recall, macro F1, confusion matrix, classification report, and tuned hyperparameters are saved in the evaluation report. Inspect the metrics together rather than relying on accuracy alone.

## Configuration
Optional environment variables: `SECRET_KEY`, `MAX_NEWS_CHARS`, `DATASET_PATH`, `MODEL_DIR`, and `MODEL_NAME`. The app reads process environment variables; it does not automatically load `.env`. For local use:
```bash
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

## Tests
```bash
python -m pytest -q
```
The tests cover routes, input validation, preprocessing, dataset validation, and baseline model training. Report the actual result after running the tests.

## Deploy to Render
1. Push the project to GitHub.
2. In Render choose **New → Blueprint** and select the repository.
3. Review `render.yaml` and deploy.
4. Check build and runtime logs.

Build command trains a baseline model; start command is:
```bash
gunicorn app:app -c gunicorn.conf.py
```
Health check path: `/health`. Model artifacts are ignored by Git, so the Render build command creates them. Free instances may have limited memory/time; if build or startup is constrained, use a larger dataset/model workflow or a suitable hosting plan.

## Limitations and responsible use
- Predictions are not proof that a story is true or false.
- The bundled dataset is synthetic/illustrative; its performance scores are not meaningful evidence of real-world accuracy.
- Satire, opinion, emerging events, paraphrases, and unfamiliar topics can be misclassified.
- Model probability outputs are not necessarily calibrated.
- The app cannot replace primary sources or independent fact-checking.

## Copyright
Copyright © 2026 Pritam Rabha. All rights reserved. Third-party libraries remain under their own licences. The university name is included for academic identification only.


## Interface pages

- `/` — Home page with newspaper background, text entry, supported news-file upload, and prediction result.
- `/performance` — separate Model Performance page with dataset summary, classifier chart, and measured results.
- `/health` — service health status.
- `/evaluation` — compatibility redirect to `/performance`.

The Home page uses the supplied newspaper image as a softened background. The Model Performance page uses a clean, light dashboard layout. Evaluation values are loaded from the saved training report; the page does not fabricate metrics.


## Interface

The interface is intentionally focused on two screens: the Detector workspace and Model performance report. The design uses responsive layouts and compact navigation without separate About or Methodology pages. Evaluation values are loaded from the saved training report; no placeholder metrics are generated.


The responsive interface has two separate screens: Home and Model Performance. The Home page includes text input, file upload, character counting, prediction details, and a verification reminder.


Navigation contains only the News detector and Model performance screens. Separate Methodology and About pages/routes are not included.


## Interface

The Home page contains only the news detector workspace. Model Performance is a separate route and page. Prediction and evaluation data are read from the Flask/Scikit-learn backend. The evaluation chart and table are populated from `artifacts/evaluation.json`; the app does not invent placeholder model scores.

The train/test split is 80/20. Displayed scores are held-out test-set metrics for this project's dataset, not guarantees of real-world fake-news detection accuracy. About and Methodology pages are intentionally omitted.
