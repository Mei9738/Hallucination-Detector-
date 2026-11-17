import warnings


def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn), with 1 = hallucinated as the positive class."""
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for true, pred in zip(y_true, y_pred, strict=True):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision(y_true: list[int], y_pred: list[int]) -> float:
    """Share of flagged answers that really are hallucinated (0.0 if none flagged)."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    if tp + fp == 0:
        warnings.warn("No positive predictions; precision set to 0.0")
        return 0.0
    p = tp / (tp + fp)
    return p


def recall(y_true: list[int], y_pred: list[int]) -> float:
    """Share of hallucinated answers that were flagged (0.0 if there are none)."""
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    r = tp / (tp + fn)
    return r


def f1(y_true: list[int], y_pred: list[int]) -> float:
    """Harmonic mean of precision and recall (0.0 if both are 0)."""

    return fbeta(y_true, y_pred, beta=1)


def fbeta(y_true: list[int], y_pred: list[int], beta: float) -> float:
    """Weighted F-score: beta > 1 favours recall, beta < 1 favours precision (0.0 if both are 0)."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    sqr_b = beta**2
    fb = ((1 + sqr_b) * p * r) / (sqr_b * p + r)
    return fb


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    """Chance a random hallucinated answer scores higher than a random supported one.

    Ties count as half a win. Raises ValueError if only one class is present.
    """
    pos_scores = []
    neg_scores = []
    for true, score in zip(y_true, scores, strict=True):
        if true == 1:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    numb_pairs = len(pos_scores) * len(neg_scores)

    if numb_pairs == 0:
        raise ValueError("Need at least one positive and one negative.")

    wins = 0.0
    for positive in pos_scores:
        for negative in neg_scores:
            if positive > negative:
                wins += 1
            elif positive == negative:
                wins += 0.5

    auc = wins / numb_pairs
    return auc
