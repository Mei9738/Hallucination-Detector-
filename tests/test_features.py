import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from hallucination_detector.features import (
    build_features,
    extract_features,
    tfidf_cosine,
)

vect = TfidfVectorizer().fit(
    [
        "The Eiffel Tower is in Paris.",
        "The Colosseum is in Rome.",
    ]
)


def test_extract_features() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "The Eiffel Tower is in Paris and was built in 1889."
    features = extract_features(answer, context, vect)

    assert set(features) == {
        "word_overlap",
        "length_ratio",
        "missing_numbers",
        "entity_overlap",
        "tfidf_cosine",
    }


def test_extract_features_empty_ctx() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "..."
    with pytest.raises(ValueError):
        extract_features(answer, context, vect)


def test_extract_features_no_num() -> None:
    answer = "The Eiffel Tower was built in Lyon."
    context = "The Eiffel Tower is in Paris and was built in 1889."
    features = extract_features(answer, context, vect)

    assert features["word_overlap"] == pytest.approx(6 / 7)
    assert features["length_ratio"] == pytest.approx(7 / 11)
    assert features["missing_numbers"] == 0.0
    assert features["entity_overlap"] == pytest.approx(1 / 2)


def test_extract_features_no_num_ctx() -> None:
    answer = "The Eiffel Tower was built in 1925 in Lyon."
    context = "The Eiffel Tower is in Paris and was built back then."
    features = extract_features(answer, context, vect)

    assert features["word_overlap"] == pytest.approx(7 / 9)
    assert features["length_ratio"] == pytest.approx(9 / 11)
    assert features["missing_numbers"] == 1.0
    assert features["entity_overlap"] == pytest.approx(1 / 3)


def test_extract_features_entity_overlap() -> None:
    answer = "The Eiffel Tower is in Paris and was built in 1925."
    context = "The Eiffel Tower was built in 1925 in Paris."
    assert extract_features(answer, context, vect)["entity_overlap"] == pytest.approx(
        3 / 3
    )


def test_extract_features_no_entity() -> None:
    answer = "It was built a long time ago."
    context = "The Eiffel Tower is in Paris and was built back then."
    assert extract_features(answer, context, vect)["entity_overlap"] == 0.0


def test_tfid_cosine() -> None:
    answer = "Paris has a Tower."
    context = "Paris has a Tower."
    assert tfidf_cosine(answer, context, vect) == pytest.approx(1.0)


def test_tfid_cosine_no_shared_words() -> None:
    answer = "Rome Colosseum"
    context = "Paris Tower"
    assert tfidf_cosine(answer, context, vect) == pytest.approx(0.0)


def make_small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "output": [
                "The Eiffel Tower was built in 1925 in Lyon.",
                "It was built a long time ago.",
            ],
            "context": [
                "The Eiffel Tower is in Paris and was built in 1889.",
                "The Eiffel Tower is in Paris and was built back then.",
            ],
        },
        index=[10, 20],  # deliberately not 0, 1, to test the index
    )


def test_build_features_shape() -> None:
    df = make_small_df()
    result = build_features(df, vect)
    assert result.shape == (2, 5)


def test_build_features_keeps_index() -> None:
    df = make_small_df()
    result = build_features(df, vect)
    assert list(result.index) == [10, 20]


def test_build_features_matches_extract_features() -> None:
    df = make_small_df()
    result = build_features(df, vect)
    expected = extract_features(df.loc[10, "output"], df.loc[10, "context"], vect)
    assert result.loc[10].to_dict() == expected
