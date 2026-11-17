import pytest

from hallucination_detector.baselines import MajorityBaseline


def test_majority_predicts_1_when_mostly_hallucinated() -> None:
    model = MajorityBaseline()
    model.fit([1, 1, 1, 1, 0])
    assert model.predict("anything") == 1
    assert model.score("anything") == 1.0


def test_majority_predicts_0_when_mostly_supported() -> None:
    model = MajorityBaseline()
    model.fit([0, 0, 1, 1, 0])
    assert model.predict("hello") == 0
    assert model.score("hello") == model.score("a completely different answer")


def test_majority_tie_predicts_1() -> None:
    model = MajorityBaseline()
    model.fit([1, 0])
    assert model.predict("tie is a tie") == 1
    assert model.score("tie is a tie") == 1.0


def test_baseline_empty_labels() -> None:
    model = MajorityBaseline()
    with pytest.raises(ValueError):
        model.fit([])
