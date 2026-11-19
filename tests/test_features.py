import pytest

from hallucination_detector.features import extract_features


def test_extract_features() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "The Eiffel Tower is in Paris and was built in 1889."
    assert extract_features(answer, context) == {
        "word_overlap": pytest.approx(7 / 9),
        "length_ratio": pytest.approx(9 / 11),
    }


def test_extract_features_empty_ctx() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "..."
    with pytest.raises(ValueError):
        extract_features(answer, context)
