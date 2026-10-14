# Fake News Detection — Project Report

## Student Information
- **Author:** Pritam Rabha
- **Programme:** MCA, 3rd Semester
- **Course:** Python Programming
- **Project Guide:** Dr. Pangambam Sendash Singh
- **Institution:** Dr. Hari Singh Gour Vishwavidyalaya, Sagar, Madhya Pradesh

## Abstract
This project demonstrates a supervised machine-learning workflow that classifies news-like text into `real` and `fake` labels. It uses text preprocessing, TF-IDF, multiple classifiers, hyperparameter tuning, standard evaluation metrics, and a Flask interface. The bundled dataset is illustrative, not a validated news corpus.

## Problem Statement
This project explores how natural-language text features can be used in a supervised classification exercise. A model prediction does not independently establish the truth of a claim.

## Objectives
- Preprocess text with stopword removal, stemming, and optional lemmatization.
- Convert text into numeric TF-IDF features.
- Train and compare multiple classifiers.
- Tune hyperparameters with cross-validation.
- Evaluate models with accuracy, precision, recall, F1, and confusion matrix.
- Build a browser interface using Flask.

## Methodology
1. Load and validate a labelled CSV.
2. Normalize and preprocess text.
3. Split examples into training and held-out test sets.
4. Fit TF-IDF and classifiers in scikit-learn pipelines.
5. Optionally run GridSearchCV.
6. Evaluate test predictions and save metrics.
7. Load a trained model in Flask to classify submitted text.

## Technology
Python, Flask, pandas, NLTK, scikit-learn, joblib, HTML, CSS, Bootstrap, pytest, and Gunicorn.

## Implementation
Code is separated into preprocessing, data loading, model definitions, training/evaluation, prediction, and Flask routes. The interface validates input length and displays the model result with a limitation notice.

## Testing
Run `python -m pytest -q` and record the actual output before submission. Tests cover core routes, input validation, preprocessing, dataset validation, and baseline training.

## Results and Limitations
The training script generates model metrics in `artifacts/evaluation.json`. Because the supplied dataset is illustrative and small, those metrics only demonstrate the workflow and must not be interpreted as real-world detection accuracy. A documented, licensed corpus and robust external validation are required for meaningful performance evaluation.

## Conclusion
The project demonstrates an end-to-end text-classification pipeline and Flask interface, covering preprocessing, TF-IDF, model comparison, tuning, evaluation, and deployment. It is not a standalone fact-checking system.

Copyright © 2026 Pritam Rabha. All rights reserved.


## Interface design
The interface has two separate pages: a Home page with text/file input and prediction output, and a Model Performance page with evaluation charts and a detailed metrics table. Evaluation metrics are read from the training report and are not hardcoded.


The revised interface contains only the Detector and Model Performance screens, keeping the application focused on the main user task.
