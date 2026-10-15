"""Merge a LoRA adapter into its base model and save a standalone model (weights + processor).

Usage: python -m llm_rl.merge <adapter_dir> <out_dir>
"""

import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def merge_adapter(adapter_dir: str, out_dir: str) -> None:
    base = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    # The full image-text model, so the merged checkpoint is a drop-in replacement for the base model.
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(base).save_pretrained(out_dir)
    print(f"Merged {adapter_dir} into {base} -> {out_dir}")


if __name__ == "__main__":
    merge_adapter(sys.argv[1], sys.argv[2])
