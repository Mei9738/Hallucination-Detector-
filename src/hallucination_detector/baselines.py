import re

from hallucination_detector.metrics import fbeta


class MajorityBaseline:
    """Always predicts the most common training label."""

    def fit(self, labels: list[int]) -> "MajorityBaseline":
        if not labels:
            raise ValueError("Empty labels.")

        positives = sum(labels)
        negatives = len(labels) - positives
        # tie -> 1: a guardrail should lean toward flagging
        if positives >= negatives:
            self.label = 1
        else:
            self.label = 0

        return self

    def predict(self, example: object) -> int:
        return self.label

    def score(self, example: object) -> float:
        return float(self.label)


def word_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's words that also appear in the context."""

    answer_words = re.findall(r"\w+", answer.lower())
    context_words = re.findall(r"\w+", context.lower())

    if not answer_words:
        raise ValueError("The answer contains no words.")

    context_words = set(context_words)
    count = 0

    for word in answer_words:
        if word in context_words:
            count += 1

    return count / len(answer_words)


class WordOverlapBaseline:
    """Flags an answer when too few of its words appear in the context."""

    def fit(
        self, examples: list[tuple[str, str]], labels: list[int]
    ) -> "WordOverlapBaseline":
        """Pick the overlap threshold with the best F2 on the training data."""
        overlaps = [word_overlap(answer, context) for answer, context in examples]
        thresholds = [i / 100 for i in range(101)]
        best_t = 0.0
        best_score = -1.0

        for t in thresholds:
            preds = [1 if o < t else 0 for o in overlaps]
            score = fbeta(labels, preds, beta=2)
            if score > best_score:
                best_score = score
                best_t = t

        self.threshold = best_t
        return self

    def score(self, example: tuple[str, str]) -> float:
        # higher = more likely hallucinated
        answer, context = example
        overlap = word_overlap(answer, context)
        return 1 - overlap

    def predict(self, example: tuple[str, str]) -> int:
        # 1 if overlap is below the threshold
        answer, context = example
        overlap = word_overlap(answer, context)
        if overlap < self.threshold:
            return 1
        return 0
