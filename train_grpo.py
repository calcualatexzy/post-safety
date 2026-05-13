import inspect
import os
import re
from typing import Any, Dict, List

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def disable_peft_bitsandbytes():
    """
    Force PEFT / TRL / Transformers to ignore an installed-but-broken bitsandbytes package.

    This is only for bf16 LoRA / GRPO.
    Do NOT use this if you actually want QLoRA / 8-bit / 4-bit training.
    """
    try:
        import transformers

        if hasattr(transformers, "is_bitsandbytes_available"):
            transformers.is_bitsandbytes_available = lambda: False

        try:
            import transformers.utils as transformers_utils
            transformers_utils.is_bitsandbytes_available = lambda: False
        except Exception:
            pass
    except Exception:
        pass

    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False

        try:
            import peft.tuners.lora.model as peft_lora_model

            peft_lora_model.is_bnb_available = lambda: False
            peft_lora_model.is_bnb_4bit_available = lambda: False
        except Exception:
            pass
    except Exception:
        pass


def get_grpo_classes():
    disable_peft_bitsandbytes()

    try:
        from trl import GRPOConfig, GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "Your TRL version does not expose GRPOConfig / GRPOTrainer. "
            "Try upgrading TRL, for example: pip install -U trl"
        ) from exc


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


def extract_boxed_letter(text: Any) -> str:
    """
    Extract gold answer from forms like:
        "\\boxed{A}"
        "Final answer: \\boxed{A}"
        "A"
    """
    if text is None:
        return ""

    text = str(text).strip().upper()

    match = re.search(r"\\BOXED\{([A-Z])\}", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    return ""


def completion_to_text(completion: Any) -> str:
    """
    GRPO completions can be strings or conversational message-like objects
    depending on TRL version / dataset format.
    """
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return str(completion.get("content", ""))

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def extract_pred_letter(completion: Any) -> tuple[str, bool]:
    """
    Return:
        pred_letter: extracted A-Z option letter
        strict_format_ok: whether final non-empty line exactly matches:
            Final answer: \\boxed{X}
    """
    text = completion_to_text(completion).strip()
    text_upper = text.upper()

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    last_line = nonempty_lines[-1] if nonempty_lines else ""

    strict_match = re.match(
        r"^Final answer:\s*\\boxed\{([A-Z])\}\s*$",
        last_line,
        flags=re.IGNORECASE,
    )

    if strict_match:
        return strict_match.group(1).upper(), True

    match = re.search(
        r"Final answer:\s*\\boxed\{([A-Z])\}",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper(), False

    boxed_matches = re.findall(
        r"\\boxed\{([A-Z])\}",
        text,
        flags=re.IGNORECASE,
    )
    if boxed_matches:
        return boxed_matches[-1].upper(), False

    match = re.search(
        r"Final answer:\s*([A-Z])\b",
        text_upper,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper(), False

    standalone_letters = re.findall(r"\b([A-Z])\b", text_upper)
    if standalone_letters:
        return standalone_letters[-1].upper(), False

    return "", False


def make_mcq_reward_func(correct_reward: float, format_reward: float):
    def mcq_boxed_reward(
        prompts: List[Any],
        completions: List[Any],
        answer: List[Any] | Any = None,
        **kwargs,
    ) -> List[float]:
        """
        Reward:
            + correct_reward if extracted final answer matches gold
            + format_reward if final line exactly matches:
                Final answer: \\boxed{X}

        Expected reward values:
            correct + strict format: 1.2
            correct but imperfect format: 1.0
            wrong but strict format: 0.2
            wrong and bad format: 0.0
        """
        if answer is None:
            answer = kwargs.get("answers", None)

        if answer is None:
            return [0.0 for _ in completions]

        if isinstance(answer, str):
            answers = [answer]
        else:
            answers = list(answer)

        # GRPO generates num_generations completions per original prompt.
        # Some TRL versions duplicate dataset columns before passing them to reward funcs;
        # some do not. This handles both cases.
        if len(answers) != len(completions):
            if len(answers) > 0 and len(completions) % len(answers) == 0:
                repeat = len(completions) // len(answers)
                answers = [a for a in answers for _ in range(repeat)]
            else:
                answers = [answers[min(i, len(answers) - 1)] for i in range(len(completions))]

        rewards = []

        correct_count = 0
        strict_format_count = 0

        for completion, gold_answer in zip(completions, answers):
            gold = extract_boxed_letter(gold_answer)
            pred, strict_format_ok = extract_pred_letter(completion)

            reward = 0.0

            if pred and gold and pred == gold:
                reward += float(correct_reward)
                correct_count += 1

            if strict_format_ok:
                reward += float(format_reward)
                strict_format_count += 1

            rewards.append(reward)

        return rewards

    mcq_boxed_reward.__name__ = "mcq_boxed_reward"
    return mcq_boxed_reward


def valid_raw_example(example: Dict[str, Any]) -> bool:
    prompt = example.get("prompt")
    answer = example.get("answer")

    if not isinstance(prompt, str):
        return False
    if not isinstance(answer, str):
        return False
    if len(prompt.strip()) == 0:
        return False

    gold = extract_boxed_letter(answer)
    return len(gold) == 1


def build_grpo_example(example: Dict[str, Any], tokenizer, cfg: DictConfig) -> Dict[str, str]:
    user_prompt = str(cfg.prompt.template).format(prompt=example["prompt"].strip())

    messages = [
        {
            "role": "system",
            "content": str(cfg.prompt.system).strip(),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {
        "prompt": prompt_text,
        "answer": extract_boxed_letter(example["answer"]),
    }


def load_policy_model(cfg: DictConfig, dtype: torch.dtype):
    """
    Load:
        base Qwen3 + input adapter, trainable

    GRPOTrainer itself handles the GRPO training loop.
    With beta=0.0, no reference model is needed.
    """
    disable_peft_bitsandbytes()

    model_kwargs = dict(
        dtype=dtype,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )

    print("===== Loading policy base model =====")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        **model_kwargs,
    )

    print("===== Loading trainable input adapter =====")
    model = PeftModel.from_pretrained(
        base_model,
        cfg.model.input_adapter_path,
        is_trainable=True,
    )
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("===== Trainable parameters =====")
    model.print_trainable_parameters()

    return model


def make_grpo_config(GRPOConfig, **kwargs):
    """
    TRL GRPOConfig changes between versions.
    This keeps the script from crashing on unsupported args.
    """
    allowed = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    allowed.discard("self")

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))

    if dropped:
        print(f"===== Dropped unsupported GRPOConfig args: {dropped} =====")

    return GRPOConfig(**filtered)


@hydra.main(version_base=None, config_path="./configs", config_name="grpo")
def main(cfg: DictConfig):
    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    disable_peft_bitsandbytes()

    GRPOConfig, GRPOTrainer = get_grpo_classes()

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

    # GRPO uses generation; left padding is preferred/expected for decoder-only generation.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("===== Loading dataset splits =====")
    train_raw = load_dataset(cfg.dataset.name, split=cfg.dataset.train_split)
    eval_raw = load_dataset(cfg.dataset.name, split=cfg.dataset.eval_split)

    print(f"Raw train examples: {len(train_raw)}")
    print(f"Raw eval examples: {len(eval_raw)}")
    print(f"Raw train columns: {train_raw.column_names}")

    train_raw = train_raw.filter(
        valid_raw_example,
        num_proc=cfg.dataset.num_proc,
        desc="Filtering valid train MCQ examples",
    )

    eval_raw = eval_raw.filter(
        valid_raw_example,
        num_proc=cfg.dataset.num_proc,
        desc="Filtering valid eval MCQ examples",
    )

    print(f"Train examples after filtering: {len(train_raw)}")
    print(f"Eval examples after filtering: {len(eval_raw)}")

    train_ds = train_raw.map(
        lambda ex: build_grpo_example(ex, tokenizer, cfg),
        remove_columns=train_raw.column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Converting train split to GRPO prompt format",
    )

    eval_ds = eval_raw.map(
        lambda ex: build_grpo_example(ex, tokenizer, cfg),
        remove_columns=eval_raw.column_names,
        num_proc=cfg.dataset.num_proc,
        desc="Converting eval split to GRPO prompt format",
    )

    train_ds = train_ds.shuffle(seed=cfg.training.seed)

    if cfg.dataset.max_train_samples is not None:
        n = min(int(cfg.dataset.max_train_samples), len(train_ds))
        train_ds = train_ds.select(range(n))
        print(f"Using max_train_samples: {len(train_ds)}")

    if cfg.dataset.max_eval_samples is not None:
        n = min(int(cfg.dataset.max_eval_samples), len(eval_ds))
        eval_ds = eval_ds.select(range(n))
        print(f"Using max_eval_samples: {len(eval_ds)}")

    print(f"Train examples: {len(train_ds)}")
    print(f"Eval examples: {len(eval_ds)}")

    print("===== Example after conversion =====")
    print(train_ds[0])

    effective_batch_size = (
        int(cfg.training.per_device_train_batch_size)
        * int(cfg.training.gradient_accumulation_steps)
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    num_generations = int(cfg.training.num_generations)

    if effective_batch_size % num_generations != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch_size} must be divisible by "
            f"num_generations={num_generations}. Change gradient_accumulation_steps "
            f"or num_generations."
        )

    model = load_policy_model(cfg, dtype=dtype)

    reward_func = make_mcq_reward_func(
        correct_reward=float(cfg.reward.correct_reward),
        format_reward=float(cfg.reward.format_reward),
    )

    generation_kwargs = {
        "repetition_penalty": float(cfg.training.repetition_penalty),
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    grpo_args = make_grpo_config(
        GRPOConfig,
        output_dir=cfg.training.output_dir,

        num_train_epochs=cfg.training.num_train_epochs,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,

        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,

        num_generations=cfg.training.num_generations,
        num_generations_eval=cfg.training.num_generations_eval,
        max_completion_length=cfg.training.max_completion_length,

        beta=cfg.training.beta,
        loss_type=cfg.training.loss_type,
        scale_rewards=cfg.training.scale_rewards,

        temperature=cfg.training.temperature,
        top_p=cfg.training.top_p,
        top_k=cfg.training.top_k,
        generation_kwargs=generation_kwargs,

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
        pad_to_multiple_of=8,

        log_completions=cfg.training.log_completions,
        num_completions_to_print=cfg.training.num_completions_to_print,

        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("===== Starting GRPO training =====")
    trainer.train()

    print("===== Saving GRPO model =====")
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    if cfg.training.push_to_hub:
        trainer.push_to_hub()

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()