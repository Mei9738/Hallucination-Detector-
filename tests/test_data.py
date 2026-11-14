import pandas as pd
import pytest

from hallucination_detector.data import (
    shared_contexts,
    split_by_context,
    split_stratified,
    to_label,
)


def make_df() -> pd.DataFrame:
    rows = []
    for c in range(10):
        for m in range(3):
            rows.append(
                {
                    "context": f"ctx{c}",
                    "model": f"m{m}",
                    "label": m % 2,
                    "task_type": f"task{c % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_split_by_context_has_no_shared_contexts() -> None:
    train, val = split_by_context(make_df(), val_frac=0.2, seed=0)
    assert shared_contexts(train, val) == set()


def test_to_label() -> None:
    supported = to_label({"evident_conflict": 0, "baseless_info": 0})
    hallucinated = to_label({"evident_conflict": 1, "baseless_info": 0})
    assert supported == 0
    assert hallucinated == 1


def test_split_by_context_keeps_all_rows() -> None:
    df = make_df()
    train, val = split_by_context(df, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(df)


def test_split_by_context_same_seed_gives_same_split() -> None:
    df = make_df()
    train1, val1 = split_by_context(df, val_frac=0.2, seed=0)
    train2, val2 = split_by_context(df, val_frac=0.2, seed=0)
    assert set(val1["context"]) == set(val2["context"])


def test_split_by_context_different_seed_gives_different_split() -> None:
    df = make_df()
    train1, val1 = split_by_context(df, val_frac=0.2, seed=1)
    train2, val2 = split_by_context(df, val_frac=0.2, seed=0)
    assert set(val1["context"]) != set(val2["context"])


def test_split_by_context_does_not_modify_input() -> None:
    df = make_df()
    original = df.copy()
    split_by_context(df, val_frac=0.2, seed=0)
    pd.testing.assert_frame_equal(df, original)


def test_split_stratified() -> None:
    df = make_df()
    train, val = split_stratified(df, val_frac=0.2, seed=0)
    shared_contexts(train, val) == set()
    expected = set(df["task_type"])
    assert set(train["task_type"]) == expected
    assert set(val["task_type"]) == expected


def test_split_by_context_raises_with_one_context() -> None:
    df = make_df()
    one_context = df[df["context"] == "ctx0"]
    with pytest.raises(ValueError):
        split_by_context(one_context, val_frac=0.2, seed=0)
