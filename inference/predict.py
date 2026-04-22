"""
Load the trained model and run inference on a raw log snippet.

Accepts input via CLI (--input for .jsonl file, --text for raw string).
Outputs both raw model text and structured JSON from RoutineExtractor.

Usage:
    python -m inference.predict --input data/raw/synthetic_logs.jsonl
    python -m inference.predict --text "06:30 bathroom_light ON ..."
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import yaml
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from inference.routine_extractor import RoutineExtractor
from training.format_data import format_sample, INSTRUCTION_TEXT

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")


def _load_first_day_from_jsonl(path: str) -> tuple[str, list[dict]]:
    """Load events from a JSONL file and return the first day's data."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        raise ValueError(f"No events found in {path}")

    days = defaultdict(list)
    for ev in events:
        date_str = ev["timestamp"][:10]
        days[date_str].append(ev)

    first_date = sorted(days.keys())[0]
    return first_date, days[first_date]


def _format_text_input(text: str) -> str:
    """Wrap raw text into instruction format."""
    return (
        f"### Instruction:\n{INSTRUCTION_TEXT}\n\n"
        f"### Input:\n{text}\n\n"
        f"### Response:\n"
    )


def predict(input_path: str = None, text: str = None,
            max_new_tokens: int = 256, temperature: float = 0.3,
            config_path: str = None) -> dict:

    if config_path is None:
        config_path = os.path.join(_BASE_DIR, "config", "training.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    local_path = os.path.join(_BASE_DIR, config["model"]["local_path"])
    lora_path = os.path.join(_BASE_DIR, config["training"]["output_dir"], "final")
    trust_remote = "phi" in model_name.lower()

    # Build prompt
    if input_path:
        date_str, day_events = _load_first_day_from_jsonl(input_path)
        prompt = format_sample(date_str, day_events)
        idx = prompt.find("### Response:")
        prompt = prompt[:idx] + "### Response:\n"
    elif text:
        prompt = _format_text_input(text)
    else:
        raise ValueError("Either --input or --text must be provided")

    logger.info("Loading model...")

    # =========================
    # Quantization
    # =========================
    qcfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    # =========================
    # Model Config Fix (Phi-3 rope_scaling)
    # =========================
    config_obj = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote,
        cache_dir=local_path,
    )
    if hasattr(config_obj, "rope_scaling"):
        config_obj.rope_scaling = None

    # =========================
    # Load base model + LoRA
    # =========================
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config_obj,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=local_path,
        trust_remote_code=trust_remote,
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=local_path, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # Generate
    # =========================
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["model"]["max_length"],
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    logger.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # =========================
    # Parse + print
    # =========================
    extractor = RoutineExtractor()
    structured = extractor.extract(generated)
    summary = extractor.to_summary(structured)

    result = {
        "raw_output": generated,
        "structured": structured,
        "summary": summary,
    }

    print("\n" + "=" * 60)
    print("RAW MODEL OUTPUT:")
    print("=" * 60)
    print(generated)
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT:")
    print("=" * 60)
    print(json.dumps(structured, indent=2))
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(summary)

    return result


def main(args=None):
    parser = argparse.ArgumentParser(description="Run inference on smart-home logs")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to .jsonl log file")
    parser.add_argument("--text", type=str, default=None,
                        help="Raw text input")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--config", type=str, default=None)

    parsed = parser.parse_args(args)

    if not parsed.input and not parsed.text:
        parser.error("Either --input or --text must be provided")

    predict(
        input_path=parsed.input,
        text=parsed.text,
        max_new_tokens=parsed.max_new_tokens,
        temperature=parsed.temperature,
        config_path=parsed.config,
    )


if __name__ == "__main__":
    main()
