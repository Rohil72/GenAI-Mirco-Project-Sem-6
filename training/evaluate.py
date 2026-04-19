"""
Run the fine-tuned model on the validation set and report metrics.

Computes ROUGE-L and exact routine name match rate.

Usage:
    python -m training.evaluate
"""

import os
import sys
import json
import argparse
import re

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from utils.logging_utils import get_logger
logger = get_logger("evaluate")


def _extract_prompt(text):
    """Extract only ### Instruction + ### Input from a full sample."""
    idx = text.find("### Response:")
    if idx == -1:
        return text
    return text[:idx].strip()


def _extract_response(text):
    """Extract only ### Response section from a full sample."""
    idx = text.find("### Response:")
    if idx == -1:
        return ""
    return text[idx + len("### Response:"):].strip()


def _extract_routine_names(text):
    """Extract routine names from text like 'Routine detected: morning_routine'."""
    return set(re.findall(r"Routine detected:\s*(\w+)", text))


def _compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score between two strings (token-level LCS)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(config_path=None):
    if config_path is None:
        config_path = os.path.join(_BASE_DIR, "config", "training.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    local_path = os.path.join(_BASE_DIR, config["model"]["local_path"])
    lora_path = os.path.join(_BASE_DIR, config["training"]["output_dir"], "final")
    val_path = os.path.join(_BASE_DIR, config["data"]["val_path"])
    trust_remote = "phi" in model_name.lower()

    # Load model with quantization
    qcfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", cache_dir=local_path, trust_remote_code=trust_remote,
    )

    logger.info(f"Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=local_path, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load val samples
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Evaluating {len(samples)} validation samples...")

    rouge_scores = []
    match_counts = []
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        text = sample["text"]
        prompt = _extract_prompt(text)
        gt_response = _extract_response(text)
        gt_routines = _extract_routine_names(gt_response)

        # Generate
        inputs = tokenizer(prompt + "\n\n### Response:\n",
                          return_tensors="pt", truncation=True,
                          max_length=config["model"]["max_length"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.2, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        # Metrics
        rouge_l = _compute_rouge_l(gt_response, generated)
        rouge_scores.append(rouge_l)

        pred_routines = _extract_routine_names(generated)
        if gt_routines:
            match = len(gt_routines & pred_routines) / len(gt_routines)
        else:
            match = 1.0 if not pred_routines else 0.0
        match_counts.append(match)

        results.append({
            "prompt_preview": prompt[:100] + "...",
            "ground_truth": gt_response[:200],
            "generated": generated[:200],
            "rouge_l": round(rouge_l, 4),
            "routine_match": round(match, 4),
        })

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_match = sum(match_counts) / len(match_counts) if match_counts else 0

    logger.info("=" * 50)
    logger.info(f"Average ROUGE-L:          {avg_rouge:.4f}")
    logger.info(f"Routine Name Match Rate:  {avg_match:.4f}")
    logger.info("=" * 50)

    # Save results
    output = {
        "avg_rouge_l": round(avg_rouge, 4),
        "avg_routine_match": round(avg_match, 4),
        "num_samples": len(samples),
        "per_sample": results,
    }
    results_path = os.path.join(_BASE_DIR, config["training"]["output_dir"],
                                "eval_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_path}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--config", type=str, default=None)
    parsed = parser.parse_args(args)
    evaluate(config_path=parsed.config)


if __name__ == "__main__":
    main()
