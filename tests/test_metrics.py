import time

import pytest
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hallucination_detector.metrics import (
    confusion_counts,
    f1,
    fbeta,
    mean_latency_ms,
    precision,
    recall,
    roc_auc,
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


def test_roc_auc() -> None:
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.7, 0.4, 0.6, 0.3, 0.2]
    assert roc_auc(y_true, y_score) == pytest.approx(8 / 9)
    assert roc_auc(y_true, y_score) == pytest.approx(roc_auc_score(y_true, y_score))


def test_roc_all_positives() -> None:
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1, 1, 1, 1], [0.9, 0.7, 0.4, 0.6, 0.3, 0.2])


def test_roc_all_negatives() -> None:
    with pytest.raises(ValueError):
        roc_auc([0, 0, 0, 0, 0, 0], [0.9, 0.7, 0.4, 0.6, 0.3, 0.2])


def test_roc_auc_same_score() -> None:
    y_true = [1, 0]
    y_score = [0.5, 0.5]
    assert roc_auc(y_true, y_score) == 0.5


def slow_predict(text: str) -> int:
    time.sleep(0.01)  # 10 ms
    return 0


def test_mean_latency_ms_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        mean_latency_ms(slow_predict, [])


def test_mean_latency_ms_measures_time() -> None:
    assert mean_latency_ms(slow_predict, ["a", "b", "c"]) >= 10
