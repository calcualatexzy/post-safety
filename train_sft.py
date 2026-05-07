import json
import os
from typing import Any

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def disable_peft_bitsandbytes():
    import peft.import_utils as peft_import_utils

    peft_import_utils.is_bnb_available = lambda: False
    peft_import_utils.is_bnb_4bit_available = lambda: False

    try:
        import peft.tuners.lora.model as peft_lora_model
        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass

def normalize_messages(messages: Any):
    if isinstance(messages, str):
        return json.loads(messages)
    return messages


def is_valid_chat(example):
    messages = normalize_messages(example["messages"])
    return (
        isinstance(messages, list)
        and len(messages) >= 2
        and messages[-1].get("role") == "assistant"
    )


def setup_wandb(cfg: DictConfig):
    if not cfg.wandb.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return "none"

    os.environ["WANDB_PROJECT"] = str(cfg.wandb.project)

    if cfg.wandb.entity is not None:
        os.environ["WANDB_ENTITY"] = str(cfg.wandb.entity)

    os.environ["WANDB_LOG_MODEL"] = str(cfg.wandb.log_model).lower()

    try:
        import wandb

        wandb.init(
            project=str(cfg.wandb.project),
            entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            name=str(cfg.wandb.name),
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except ImportError as exc:
        raise ImportError(
            "wandb.enabled=true but wandb is not installed. Run: pip install -U wandb"
        ) from exc

    return "wandb"


@hydra.main(version_base=None, config_path="./configs", config_name="sft")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    report_to = setup_wandb(cfg)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )

    model.config.use_cache = False

    raw = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    raw = raw.filter(is_valid_chat, num_proc=cfg.dataset.num_proc)
    raw = raw.shuffle(seed=cfg.training.seed)

    if cfg.dataset.max_train_samples is not None:
        n = min(int(cfg.dataset.max_train_samples), len(raw))
        raw = raw.select(range(n))

    split = raw.train_test_split(
        test_size=float(cfg.dataset.val_size),
        seed=cfg.training.seed,
    )

    train_ds = split["train"]
    eval_ds = split["test"]

    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
    )

    args = SFTConfig(
        output_dir=cfg.training.output_dir,

        max_length=cfg.training.max_seq_length,
        packing=cfg.training.packing,
        assistant_only_loss=cfg.training.assistant_only_loss,
        dataset_num_proc=cfg.dataset.num_proc,
        pad_to_multiple_of=8,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,

        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,

        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",

        report_to=report_to,
        run_name=str(cfg.wandb.name) if cfg.wandb.enabled else None,

        push_to_hub=cfg.training.push_to_hub,
        hub_model_id=cfg.training.hub_model_id,
        seed=cfg.training.seed,
        remove_unused_columns=True,
        use_cache=False,
    )

    disable_peft_bitsandbytes()
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    if cfg.training.push_to_hub:
        trainer.push_to_hub()

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()