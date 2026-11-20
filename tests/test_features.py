import pytest

from hallucination_detector.features import extract_features


def test_extract_features() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "The Eiffel Tower is in Paris and was built in 1889."
    assert extract_features(answer, context) == {
        "word_overlap": pytest.approx(7 / 9),
        "length_ratio": pytest.approx(9 / 11),
        "missing_numbers": pytest.approx(1),
        "entity_overlap": pytest.approx(1 / 3),
    }


def test_extract_features_empty_ctx() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "..."
    with pytest.raises(ValueError):
        extract_features(answer, context)


def test_extract_features_no_num() -> None:
    answer = "The Eiffel Tower was built in Lyon."
    context = "The Eiffel Tower is in Paris and was built in 1889."
    assert extract_features(answer, context) == {
        "word_overlap": pytest.approx(6 / 7),
        "length_ratio": pytest.approx(7 / 11),
        "missing_numbers": 0.0,
        "entity_overlap": pytest.approx(1 / 2),
    }


def test_extract_features_no_num_ctx() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "The Eiffel Tower is in Paris and was built back then."
    assert extract_features(answer, context) == {
        "word_overlap": pytest.approx(7 / 9),
        "length_ratio": pytest.approx(9 / 11),
        "missing_numbers": 1.0,
        "entity_overlap": pytest.approx(1 / 3),
    }


def test_extract_features_entity_overlap() -> None:
    answer = "The Eiffel Tower is in Paris and was built in 1925."
    context = "The Eiffel Tower was built in 1925 in Paris."
    assert extract_features(answer, context)["entity_overlap"] == pytest.approx(3 / 3)


def test_extract_features_no_entity() -> None:
    answer = "It was built a long time ago."
    context = "The Eiffel Tower is in Paris and was built back then."
    assert extract_features(answer, context)["entity_overlap"] == 0.0
