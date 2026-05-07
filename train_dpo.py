import os
from typing import Any, Dict

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def disable_peft_bitsandbytes():
    """
    Force PEFT to ignore an installed-but-broken bitsandbytes package.

    This is only for bf16 LoRA/DPO.
    Do NOT use this if you actually want QLoRA / 8-bit / 4-bit training.
    """
    import peft.import_utils as peft_import_utils

    peft_import_utils.is_bnb_available = lambda: False
    peft_import_utils.is_bnb_4bit_available = lambda: False

    try:
        import peft.tuners.lora.model as peft_lora_model

        peft_lora_model.is_bnb_available = lambda: False
        peft_lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass


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


def valid_raw_example(example: Dict[str, Any]) -> bool:
    """
    Keep only clear preference examples.

    preference_ranking == 1:
        response1 is chosen, response2 is rejected

    preference_ranking == 6:
        response2 is chosen, response1 is rejected

    Anything else is discarded.
    """
    prompt = example.get("prompt")
    response1 = example.get("response1")
    response2 = example.get("response2")
    ranking = example.get("preference_ranking")

    if not isinstance(prompt, str):
        return False
    if not isinstance(response1, str):
        return False
    if not isinstance(response2, str):
        return False

    if len(prompt.strip()) == 0:
        return False
    if len(response1.strip()) == 0:
        return False
    if len(response2.strip()) == 0:
        return False

    try:
        ranking = int(ranking)
    except Exception:
        return False

    return ranking in (1, 6)


def build_dpo_example(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Convert Nemotron-RL-Safety-v1 format into TRL DPO format.

    Output format:
        {
            "prompt": str,
            "chosen": str,
            "rejected": str,
        }

    We pre-format the prompt with Qwen3 chat template and enable_thinking=False.
    The chosen/rejected strings are raw assistant completions.
    """
    ranking = int(example["preference_ranking"])

    if ranking == 1:
        chosen_text = example["response1"].strip()
        rejected_text = example["response2"].strip()
    elif ranking == 6:
        chosen_text = example["response2"].strip()
        rejected_text = example["response1"].strip()
    else:
        raise ValueError(f"Unexpected preference_ranking={ranking}")

    prompt_messages = [
        {
            "role": "user",
            "content": example["prompt"].strip(),
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def load_policy_and_reference(cfg: DictConfig, dtype: torch.dtype):
    """
    Load two models:

    policy_model:
        base Qwen3 + SFT LoRA adapter, trainable

    ref_model:
        base Qwen3 + same SFT LoRA adapter, frozen

    DPO then optimizes the policy relative to the SFT reference model.
    """
    disable_peft_bitsandbytes()

    model_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )

    print("===== Loading trainable policy base model =====")
    policy_base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **model_kwargs,
    )

    print("===== Loading trainable SFT adapter =====")
    policy_model = PeftModel.from_pretrained(
        policy_base,
        cfg.model.sft_adapter_path,
        is_trainable=True,
    )
    policy_model.config.use_cache = False

    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()

    print("===== Trainable parameters =====")
    policy_model.print_trainable_parameters()

    print("===== Loading frozen reference base model =====")
    ref_base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **model_kwargs,
    )

    print("===== Loading frozen reference SFT adapter =====")
    ref_model = PeftModel.from_pretrained(
        ref_base,
        cfg.model.sft_adapter_path,
        is_trainable=False,
    )
    ref_model.config.use_cache = False
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad_(False)

    return policy_model, ref_model


@hydra.main(version_base=None, config_path="./configs", config_name="dpo")
def main(cfg: DictConfig):
    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    disable_peft_bitsandbytes()

    report_to = setup_wandb(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script expects GPU training.")

    bf16 = torch.cuda.is_bf16_supported()
    fp16 = not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16

    print("===== CUDA =====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"bf16 supported: {bf16}")
    print(f"dtype: {dtype}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("===== Loading tokenizer =====")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("===== Loading dataset =====")
    raw = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    print(f"Raw examples: {len(raw)}")
    print(f"Raw columns: {raw.column_names}")

    raw = raw.filter(
        valid_raw_example,
        num_proc=cfg.dataset.num_proc,
        desc="Filtering preference_ranking == 1 or 6",
    )
    print(f"Examples after filtering: {len(raw)}")

    raw = raw.map(
        lambda ex: build_dpo_example(ex, tokenizer),
        remove_columns=raw.column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Converting to TRL DPO prompt/chosen/rejected format",
    )

    raw = raw.shuffle(seed=cfg.training.seed)

    if cfg.dataset.max_train_samples is not None:
        n = min(int(cfg.dataset.max_train_samples), len(raw))
        raw = raw.select(range(n))
        print(f"Using max_train_samples: {len(raw)}")

    split = raw.train_test_split(
        test_size=float(cfg.dataset.val_size),
        seed=cfg.training.seed,
    )

    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")

    print("===== Example after conversion =====")
    print(train_ds[0])

    policy_model, ref_model = load_policy_and_reference(cfg, dtype=dtype)

    dpo_args = DPOConfig(
        output_dir=cfg.training.output_dir,

        max_length=cfg.training.max_seq_length,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,

        beta=cfg.training.beta,
        loss_type=cfg.training.loss_type,

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
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("===== Starting DPO training =====")
    trainer.train()

    print("===== Saving DPO model =====")
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    if cfg.training.push_to_hub:
        trainer.push_to_hub()

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()