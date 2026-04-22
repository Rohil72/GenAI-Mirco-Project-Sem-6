"""QLoRA fine-tuning script (stable FP16, TRL 1.2.0)."""

import os
import sys
import argparse
import yaml
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# =========================
# PATH SETUP
# =========================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_lora")

# =========================
# HELPERS
# =========================

def _detect_target_modules(model_name, config_modules):
    name_lower = model_name.lower()
    if "phi-3" in name_lower:
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif "phi-2" in name_lower:
        return ["q_proj", "v_proj", "k_proj", "dense"]
    return config_modules


def _sft_config_kwargs(max_length, text_field):
    """Return only the SFTConfig kwargs that exist in this install."""
    import inspect
    sig = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {}
    candidates = {
        "max_seq_length": max_length,   # older TRL
        "max_length": max_length,        # newer TRL
        "dataset_text_field": text_field,
    }
    for k, v in candidates.items():
        if k in sig:
            kwargs[k] = v
    return kwargs

# =========================
# TRAIN FUNCTION
# =========================

def train(config_path=None):
    if config_path is None:
        config_path = os.path.join(_BASE_DIR, "config", "training.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    local_path = os.path.join(_BASE_DIR, config["model"]["local_path"])
    max_length = config["model"]["max_length"]
    trust_remote = "phi" in model_name.lower()

    os.makedirs(local_path, exist_ok=True)
    logger.info(f"Model: {model_name}")

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
    # Model Config Fix (Phi)
    # =========================
    config_obj = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote,
        cache_dir=local_path,
    )

    if hasattr(config_obj, "rope_scaling"):
        config_obj.rope_scaling = None

    # =========================
    # Load Model
    # =========================
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config_obj,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=local_path,
        trust_remote_code=trust_remote,
        attn_implementation="eager",
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # =========================
    # LoRA
    # =========================
    lcfg = config["lora"]
    target_modules = _detect_target_modules(model_name, lcfg["target_modules"])

    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # =========================
    # Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=local_path,
        trust_remote_code=trust_remote,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # Dataset
    # =========================
    train_path = os.path.join(_BASE_DIR, config["data"]["train_path"])
    val_path = os.path.join(_BASE_DIR, config["data"]["val_path"])

    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path,
    })

    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # =========================
    # SFTConfig
    # =========================
    tcfg = config["training"]
    output_dir = os.path.join(_BASE_DIR, tcfg["output_dir"])

    warmup_steps = int(
        tcfg.get("warmup_ratio", 0.05)
        * tcfg["num_train_epochs"]
        * tcfg["gradient_accumulation_steps"]
    )

    # Probe which SFT-specific kwargs exist at runtime to avoid version churn
    sft_extra = _sft_config_kwargs(
        max_length=max_length,
        text_field=tcfg.get("dataset_text_field", "text"),
    )
    logger.info(f"SFTConfig extra kwargs: {list(sft_extra.keys())}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        report_to="none",
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=2,
        eval_steps=tcfg["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        fp16=False,
        bf16=False,  # fp16 also off: Phi-3 has bfloat16 internals that break the fp16 grad scaler
        optim=tcfg["optim"],
        gradient_checkpointing=True,
        remove_unused_columns=False,
        **sft_extra,
    )

    # =========================
    # Trainer
    # =========================
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # =========================
    # TRAIN
    # =========================
    logger.info("Starting training...")
    trainer.train()

    # =========================
    # SAVE
    # =========================
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"Saved to {final_path}")


# =========================
# ENTRY
# =========================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parsed = parser.parse_args(args)
    train(config_path=parsed.config)


if __name__ == "__main__":
    main()
