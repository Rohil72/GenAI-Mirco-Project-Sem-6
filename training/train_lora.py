"""
Fine-tune Phi-2 / Phi-3-mini with QLoRA using processed training data.

All hyperparameters are read from config/training.yaml.

Usage:
    python -m training.train_lora
"""

import os
import sys
import argparse

import yaml
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from utils.logging_utils import get_logger
logger = get_logger("train_lora")


def _detect_target_modules(model_name, config_modules):
    name_lower = model_name.lower()
    if "phi-3" in name_lower or "phi3" in name_lower:
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif "phi-2" in name_lower or "phi2" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "dense"]
    return config_modules


def _get_dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(s, torch.float16)


def train(config_path=None):
    if config_path is None:
        config_path = os.path.join(_BASE_DIR, "config", "training.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    local_path = os.path.join(_BASE_DIR, config["model"]["local_path"])
    max_length = config["model"]["max_length"]
    trust_remote = "phi" in model_name.lower()

    logger.info(f"Model: {model_name}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM before load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    # Quantization config
    qcfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=_get_dtype(qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", cache_dir=local_path, trust_remote_code=trust_remote,
    )
    if torch.cuda.is_available():
        logger.info(f"VRAM after model load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    model = prepare_model_for_kbit_training(model)

    # LoRA
    lcfg = config["lora"]
    target_modules = _detect_target_modules(model_name, lcfg["target_modules"])
    lora_config = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["lora_alpha"],
        target_modules=target_modules, lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"], task_type=lcfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=local_path, trust_remote_code=trust_remote)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Datasets
    train_path = os.path.join(_BASE_DIR, config["data"]["train_path"])
    val_path = os.path.join(_BASE_DIR, config["data"]["val_path"])
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # Training arguments
    tcfg = config["training"]
    output_dir = os.path.join(_BASE_DIR, tcfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        save_steps=tcfg["save_steps"], logging_steps=tcfg["logging_steps"],
        eval_steps=tcfg["eval_steps"], eval_strategy="steps",
        fp16=tcfg["fp16"], optim=tcfg["optim"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to="none", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        tokenizer=tokenizer, dataset_text_field="text",
        max_seq_length=max_length, packing=False,
    )
    logger.info("Starting training...")
    trainer.train()

    # Save
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"LoRA adapter saved to {final_path}")

    if torch.cuda.is_available():
        logger.info(f"VRAM after training: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    logger.info("Training complete!")


def main(args=None):
    parser = argparse.ArgumentParser(description="Fine-tune with QLoRA")
    parser.add_argument("--config", type=str, default=None)
    parsed = parser.parse_args(args)
    train(config_path=parsed.config)


if __name__ == "__main__":
    main()
