import pytest


def test_homepage(app):
    """The homepage should load with text-based news input."""
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Fake News" in response.data
    assert b"news_text" in response.data
    assert b"Upload News File" not in response.data


def test_performance_page_is_separate(app):
    """The performance dashboard should have its own page."""
    response = app.test_client().get("/performance")

    assert response.status_code == 200
    assert b"Model Performance" in response.data
    assert b"Analyze a News Article" not in response.data


def test_health(app):
    """The health endpoint should return the application status."""
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert "model" in response.json


def test_about_removed(app):
    """The removed About page should return 404."""
    response = app.test_client().get("/about")

    assert response.status_code == 404


def test_methodology_removed(app):
    """The removed Methodology page should return 404."""
    response = app.test_client().get("/methodology")

    assert response.status_code == 404


def test_empty_submission(app):
    """Empty or whitespace-only news should be rejected."""
    response = app.test_client().post(
        "/predict",
        data={"news_text": "   "},
    )

    assert response.status_code == 400
    assert b"Please enter" in response.data


def test_too_long_submission(app):
    """News exceeding the configured character limit should be rejected."""
    app.config["MAX_NEWS_CHARS"] = 20

    response = app.test_client().post(
        "/predict",
        data={"news_text": "x" * 21},
    )

    assert response.status_code == 400
    assert b"exceeds the" in response.data


def test_evaluation_redirect(app):
    """The legacy evaluation URL should redirect to the performance page."""
    response = app.test_client().get("/evaluation")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/performance")
