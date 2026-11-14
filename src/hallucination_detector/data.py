import numpy as np
import pandas as pd


def to_label(processed: dict[str, int]) -> int:
    """1 means hallucinated."""
    return 0 if all(v == 0 for v in processed.values()) else 1


def shared_contexts(a: pd.DataFrame, b: pd.DataFrame) -> set[str]:
    return set(a["context"]) & set(b["context"])


def split_by_context(
    df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val) so that no context is in both."""
    ctx = df["context"].unique()

    if len(ctx) < 2:
        raise ValueError(
            "Cannot split by context: at least 2 unique contexts are required."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ctx)

    n = max(1, int(len(shuffled) * val_frac))

    val_ctx = shuffled[:n]
    mask = df["context"].isin(val_ctx)

    val = df[mask].copy()
    train = df[~mask].copy()
    return train, val


def split_stratified(
    df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_list = []
    val_list = []
    for task in df["task_type"].unique():
        mask = df["task_type"] == task
        task_train, task_val = split_by_context(df[mask], val_frac=val_frac, seed=seed)
        train_list.append(task_train)
        val_list.append(task_val)
    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True)
    return train, val
