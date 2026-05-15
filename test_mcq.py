import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import hydra
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def disable_peft_bitsandbytes():
    """
    Avoid broken bitsandbytes import through PEFT in environments where bnb exists
    but is CUDA-incompatible. This script does not use 4-bit / 8-bit loading.
    """
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


def resolve_dtype(cfg: DictConfig):
    dtype = str(cfg.model.dtype).lower()

    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if dtype in {"fp16", "float16"}:
        return torch.float16

    if dtype in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unknown dtype: {cfg.model.dtype}")


def local_has_adapter_config(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def detect_lora_model(cfg: DictConfig) -> bool:
    mode = str(cfg.model.is_lora).lower()

    if mode in {"true", "yes", "1"}:
        return True

    if mode in {"false", "no", "0"}:
        return False

    if mode != "auto":
        raise ValueError("model.is_lora must be one of: auto, true, false")

    model_path = str(cfg.model.name_or_path)

    if local_has_adapter_config(model_path):
        return True

    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(model_path)
        return True
    except Exception:
        return False


def load_tokenizer(cfg: DictConfig, is_lora: bool, base_model_name: str):
    trust_remote_code = bool(cfg.model.trust_remote_code)

    if cfg.model.tokenizer_name_or_path is not None:
        tokenizer_path = str(cfg.model.tokenizer_name_or_path)
        print("===== Loading tokenizer from explicit path =====")
        print(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
        return tokenizer

    model_path = str(cfg.model.name_or_path)

    if is_lora:
        try:
            print("===== Trying tokenizer from LoRA adapter path =====")
            print(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            return tokenizer
        except Exception as exc:
            print("Could not load tokenizer from adapter path. Falling back to base model.")
            print("Tokenizer load error:", repr(exc))

        print("===== Loading tokenizer from base model =====")
        print(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        return tokenizer

    print("===== Loading tokenizer from model path =====")
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


def load_model_and_tokenizer(cfg: DictConfig):
    disable_peft_bitsandbytes()

    dtype = resolve_dtype(cfg)
    is_lora = detect_lora_model(cfg)

    print("===== Model loading info =====")
    print("model.name_or_path:", cfg.model.name_or_path)
    print("is_lora:", is_lora)
    print("dtype:", dtype)
    print("device_map:", cfg.model.device_map)

    if is_lora:
        from peft import PeftConfig, PeftModel

        adapter_path = str(cfg.model.name_or_path)
        peft_config = PeftConfig.from_pretrained(adapter_path)

        if cfg.model.base_model_name_or_path is not None:
            base_model_name = str(cfg.model.base_model_name_or_path)
        else:
            base_model_name = peft_config.base_model_name_or_path

        tokenizer = load_tokenizer(cfg, is_lora=True, base_model_name=base_model_name)

        print("===== Loading base model for LoRA =====")
        print(base_model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            device_map=str(cfg.model.device_map),
            trust_remote_code=bool(cfg.model.trust_remote_code),
        )

        print("===== Loading LoRA adapter =====")
        print(adapter_path)

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )

        print("===== Merging LoRA adapter into base model in memory =====")
        model = model.merge_and_unload()
        model.eval()

    else:
        model_path = str(cfg.model.name_or_path)
        tokenizer = load_tokenizer(cfg, is_lora=False, base_model_name=model_path)

        print("===== Loading full / merged model directly =====")
        print(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=str(cfg.model.device_map),
            trust_remote_code=bool(cfg.model.trust_remote_code),
        )
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return model, tokenizer


def extract_gold_letter(answer: Any) -> str:
    text = str(answer).strip().upper()

    match = re.search(r"\\BOXED\{([A-Z])\}", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    return ""


def extract_pred_letter(output: str) -> str:
    text = str(output).strip()

    match = re.search(
        r"Final answer:\s*\\boxed\{([A-Z])\}",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    boxed = re.findall(
        r"\\boxed\{([A-Z])\}",
        text,
        flags=re.IGNORECASE,
    )
    if boxed:
        return boxed[-1].upper()

    return ""


def build_messages(prompt: str, cfg: DictConfig) -> List[Dict[str, str]]:
    mode = str(cfg.prompt.mode).lower()

    if mode == "ci_template":
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    if mode == "wrapped":
        user_prompt = str(cfg.prompt.prompt_template).format(prompt=prompt)
        return [
            {
                "role": "system",
                "content": str(cfg.prompt.system_prompt).strip(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    raise ValueError("prompt.mode must be either 'ci_template' or 'wrapped'")


def generate_batch(model, tokenizer, prompts: List[str], cfg: DictConfig) -> List[str]:
    texts = []

    for prompt in prompts:
        messages = build_messages(prompt, cfg)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=int(cfg.generation.max_new_tokens),
        do_sample=bool(cfg.generation.do_sample),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=float(cfg.generation.repetition_penalty),
    )

    if bool(cfg.generation.do_sample):
        gen_kwargs["temperature"] = float(cfg.generation.temperature)
        gen_kwargs["top_p"] = float(cfg.generation.top_p)
        gen_kwargs["top_k"] = int(cfg.generation.top_k)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_kwargs,
        )

    outputs = []

    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for i in range(len(prompts)):
        # Since we use left padding, the generated sequence contains the full padded input.
        # The new tokens are after the full input_ids width, not after the unpadded length.
        output_ids = generated_ids[i][inputs["input_ids"].shape[1]:]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        outputs.append(output)

    return outputs


@hydra.main(version_base=None, config_path="./configs", config_name="test_mcq")
def main(cfg: DictConfig):
    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg)

    print("===== Chat template preview =====")
    preview_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Which is safer?\n\nA) Help someone\nB) Harm someone"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(preview_text)

    print("===== Loading dataset =====")
    ds = load_dataset(
        str(cfg.dataset.name),
        split=str(cfg.dataset.split),
    )

    print("Dataset size:", len(ds))
    print("Columns:", ds.column_names)

    if cfg.dataset.max_samples is not None:
        n = min(int(cfg.dataset.max_samples), len(ds))
        ds = ds.select(range(n))
        print("Using max_samples:", len(ds))

    prompts = [str(x) for x in ds[str(cfg.dataset.prompt_column)]]
    golds = [extract_gold_letter(x) for x in ds[str(cfg.dataset.answer_column)]]

    output_dir = Path(str(cfg.eval.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    batch_size = int(cfg.eval.batch_size)

    print("===== Running evaluation =====")

    for start in tqdm(range(0, len(prompts), batch_size)):
        end = min(start + batch_size, len(prompts))

        batch_prompts = prompts[start:end]
        batch_golds = golds[start:end]

        batch_outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            cfg=cfg,
        )

        for j, output in enumerate(batch_outputs):
            idx = start + j
            pred = extract_pred_letter(output)
            gold = batch_golds[j]

            results.append(
                {
                    "idx": idx,
                    "prompt": batch_prompts[j],
                    "gold": gold,
                    "pred": pred,
                    "correct": pred == gold,
                    "output": output,
                }
            )

            if idx < int(cfg.eval.print_first_n):
                print("=" * 80)
                print("idx:", idx)
                print("gold:", gold)
                print("pred:", pred)
                print("correct:", pred == gold)
                print("output:")
                print(output)

    df = pd.DataFrame(results)

    accuracy = float(df["correct"].mean()) if len(df) > 0 else 0.0
    correct = int(df["correct"].sum()) if len(df) > 0 else 0
    total = int(len(df))

    print("===== Results =====")
    print("Accuracy:", accuracy)
    print("Correct:", correct)
    print("Total:", total)

    csv_path = output_dir / str(cfg.eval.output_file)
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    summary = {
        "model": str(cfg.model.name_or_path),
        "dataset": str(cfg.dataset.name),
        "split": str(cfg.dataset.split),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "prompt_mode": str(cfg.prompt.mode),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved summary:", summary_path)

    if bool(cfg.eval.save_json):
        json_path = output_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Saved JSON:", json_path)


if __name__ == "__main__":
    main()