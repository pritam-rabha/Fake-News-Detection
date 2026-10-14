from detector.preprocessing import preprocess_text
def test_returns_string():
    assert isinstance(preprocess_text("A headline about running stories"), str)
def test_lowercases_and_removes_stopwords():
    result = preprocess_text("THE news is about the people")
    assert "the" not in result.split()
    assert result == result.lower()
def test_non_string_input():
    assert isinstance(preprocess_text(None), str)
