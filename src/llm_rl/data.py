"""GSM8K loading and prompt formatting."""

from datasets import Dataset, load_dataset
from omegaconf import DictConfig


def parse_gold(solution: str) -> str:
    """GSM8K reference solutions end with '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "")


def build_prompt(question: str, instruction: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{question}\n{instruction}"}]


def load_gsm8k(cfg: DictConfig, split: str) -> Dataset:
    """Load a GSM8K split as conversational prompts with a `gold` answer column."""
    ds = load_dataset(cfg.dataset, cfg.config, split=split)
    return ds.map(
        lambda ex: {
            "prompt": build_prompt(ex["question"], cfg.instruction),
            "gold": parse_gold(ex["answer"]),
        },
        remove_columns=ds.column_names,
    )


def load_train_val(cfg: DictConfig, seed: int) -> tuple[Dataset, Dataset]:
    """Train split minus `cfg.val_size` held-out prompts, and those held-out prompts."""
    split = load_gsm8k(cfg, "train").train_test_split(test_size=cfg.val_size, seed=seed)
    return split["train"], split["test"]
