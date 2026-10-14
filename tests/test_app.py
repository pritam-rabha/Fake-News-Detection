def test_homepage(app):
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Fake News" in response.data
    assert b"Upload News File" in response.data
    assert b"Model Performance" in response.data

def test_performance_page_is_separate(app):
    response = app.test_client().get("/performance")
    assert response.status_code == 200
    assert b"Model Performance" in response.data
    assert b"Analyze a News Article" not in response.data

def test_health(app):
    response = app.test_client().get("/health")
    assert response.status_code == 200 and response.json["status"] == "ok"

def test_about_removed(app):
    response = app.test_client().get("/about")
    assert response.status_code == 404

def test_methodology_removed(app):
    response = app.test_client().get("/methodology")
    assert response.status_code == 404

def test_empty_submission(app):
    response = app.test_client().post("/predict", data={"news_text": "   "})
    assert response.status_code == 400 and b"Please enter" in response.data

def test_too_long_submission(app):
    app.config["MAX_NEWS_CHARS"] = 20
    response = app.test_client().post("/predict", data={"news_text": "x" * 21})
    assert response.status_code == 400


def test_text_file_upload_extracts_content():
    import io
    from werkzeug.datastructures import FileStorage
    from app import extract_uploaded_text
    upload = FileStorage(stream=io.BytesIO(b"Headline\nArticle body text."), filename="news.txt")
    text, filename = extract_uploaded_text(upload)
    assert "Article body text." in text
    assert filename == "news.txt"


def test_unsupported_file_type_rejected():
    import io
    import pytest
    from werkzeug.datastructures import FileStorage
    from app import extract_uploaded_text
    upload = FileStorage(stream=io.BytesIO(b"not executable"), filename="news.exe")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_uploaded_text(upload)
