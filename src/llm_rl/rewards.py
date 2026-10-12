"""GSM8K answer extraction and the GRPO reward function."""

import math
import re
from collections.abc import Callable
from statistics import mean

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_TEXT = re.compile(r"\\(?:text|mathrm|textbf)\{[^{}]*\}")
_FRAC = re.compile(r"(-?)\\[dt]?frac\{(\d+)\}\{(\d+)\}")


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} in `text`, or None if absent or unclosed."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    begin = start + len("\\boxed{")
    depth = 1
    for i in range(begin, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i]
    return None


def parse_number(answer: str) -> float | None:
    """Parse a boxed answer such as '1,234', '\\$18', '18 \\text{ apples}', '50\\%' or '\\frac{3}{4}'.

    Returns None unless the answer contains exactly one number (after stripping units and symbols).
    """
    s = _TEXT.sub("", _THOUSANDS_SEP.sub("", answer))
    for token in ("\\$", "$", "\\%", "%", "\\!", "\\,"):
        s = s.replace(token, "")
    s = s.strip()
    if frac := _FRAC.fullmatch(s):
        sign, num, den = frac.groups()
        return (-1 if sign else 1) * int(num) / int(den) if int(den) else None
    numbers = _NUMBER.findall(s)
    return float(numbers[0]) if len(numbers) == 1 else None


def extract_last_number(text: str) -> float | None:
    """Last number anywhere in `text`; a lenient fallback that ignores the requested answer format."""
    numbers = _NUMBER.findall(_THOUSANDS_SEP.sub("", text))
    return float(numbers[-1]) if numbers else None


def is_correct(pred: float | None, gold: str) -> bool:
    return pred is not None and math.isclose(pred, float(gold), rel_tol=0.0, abs_tol=1e-6)


def strict_answer(text: str) -> float | None:
    boxed = extract_boxed(text)
    return parse_number(boxed) if boxed is not None else None


def _completion_text(completion: str | list[dict[str, str]]) -> str:
    # Conversational completions arrive as [{"role": "assistant", "content": ...}].
    return completion if isinstance(completion, str) else completion[0]["content"]


def gsm8k_reward(
    completions: list,
    gold: list[str],
    log_metric: Callable[[str, float], None] | None = None,
    **kwargs,
) -> list[float]:
    """1.0 if the last \\boxed{} answer equals the GSM8K reference answer, else 0.0.

    Also logs the \\boxed{} usage rate and lenient (last-number) accuracy, so format learning can be
    told apart from better reasoning.
    """
    texts = [_completion_text(c) for c in completions]
    rewards = [float(is_correct(strict_answer(t), g)) for t, g in zip(texts, gold, strict=True)]
    if log_metric is not None:
        log_metric("gsm8k/boxed_rate", mean(extract_boxed(t) is not None for t in texts))
        log_metric(
            "gsm8k/lenient_acc",
            mean(is_correct(extract_last_number(t), g) for t, g in zip(texts, gold, strict=True)),
        )
    return rewards
