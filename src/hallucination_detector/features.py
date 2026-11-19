import re

from hallucination_detector.baselines import word_overlap


def extract_features(answer: str, context: str) -> dict[str, float]:
    """Turn one (answer, context) pair into numeric features."""

    len_answer = len(re.findall(r"\w+", answer.lower()))
    len_context = len(re.findall(r"\w+", context.lower()))
    if len_context == 0:
        raise ValueError("The context contains no words.")

    length_ratio = len_answer / len_context
    word_fraction = word_overlap(answer, context)

    answer_numbers = re.findall(r"\d+", answer)
    context_numbers = set(re.findall(r"\d+", context))

    if not answer_numbers:
        missing_numbers = 0.0
    else:
        missing = 0
        for n in answer_numbers:
            if n not in context_numbers:
                missing += 1
        missing_numbers = missing / len(answer_numbers)

    return {
        "word_overlap": word_fraction,
        "length_ratio": length_ratio,
        "missing_numbers": missing_numbers,
    }
