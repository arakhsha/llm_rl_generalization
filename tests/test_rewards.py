import pytest

from llm_rl.data import parse_gold
from llm_rl.rewards import extract_boxed, extract_last_number, gsm8k_reward, parse_number


def test_parse_gold():
    assert parse_gold("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 1,000") == "1000"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so the answer is \\boxed{18}.", "18"),
        ("\\boxed{1} then \\boxed{2}", "2"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("no box here: 18", None),
        ("truncated \\boxed{18", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("18", 18.0),
        ("1,234", 1234.0),
        ("\\$18.00", 18.0),
        ("18 \\text{ apples}", 18.0),
        ("50\\%", 50.0),
        ("-3", -3.0),
        ("\\frac{3}{4}", 0.75),
        ("2:30", None),
        ("x", None),
    ],
)
def test_parse_number(answer, expected):
    assert parse_number(answer) == expected


def test_extract_last_number():
    assert extract_last_number("It costs $1,234. Then 5 more, so 1,239.") == 1239.0
    assert extract_last_number("no digits") is None


def test_gsm8k_reward():
    completions = [
        [{"role": "assistant", "content": "3 + 4 = 7, so \\boxed{7}"}],
        [{"role": "assistant", "content": "The answer is 7."}],  # right number, wrong format
        [{"role": "assistant", "content": "\\boxed{8}"}],
    ]
    logged = {}
    rewards = gsm8k_reward(completions, gold=["7", "7", "7"], log_metric=logged.__setitem__)
    assert rewards == [1.0, 0.0, 0.0]
    assert logged["gsm8k/boxed_rate"] == pytest.approx(2 / 3)
    assert logged["gsm8k/lenient_acc"] == pytest.approx(2 / 3)
