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
