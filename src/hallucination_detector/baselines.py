import re


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
