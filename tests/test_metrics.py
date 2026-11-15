import pytest
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from hallucination_detector.metrics import (
    confusion_counts,
    f1,
    fbeta,
    precision,
    recall,
)

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]


def test_confusion_counts() -> None:
    assert confusion_counts(y_true, y_pred) == (2, 1, 2, 5)


def test_precision() -> None:
    assert precision(y_true, y_pred) == pytest.approx(precision_score(y_true, y_pred))


def test_recall() -> None:
    assert recall(y_true, y_pred) == pytest.approx(recall_score(y_true, y_pred))


def test_f1() -> None:
    assert f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred))


def test_zero_division() -> None:
    assert recall([0, 0, 0], [1, 0, 0]) == 0.0

    with pytest.warns(UserWarning):
        assert precision([1, 0, 1], [0, 0, 0]) == 0.0
        assert f1([1, 0, 1], [0, 0, 0]) == 0.0
        assert fbeta([1, 0, 1], [0, 0, 0], 2) == 0.0


def test_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        confusion_counts([1, 1, 1], [1])


def test_fbeta() -> None:
    assert fbeta(y_true, y_pred, beta=2) == pytest.approx(
        fbeta_score(y_true, y_pred, beta=2)
    )
    assert fbeta(y_true, y_pred, beta=0.5) == pytest.approx(
        fbeta_score(y_true, y_pred, beta=0.5)
    )
