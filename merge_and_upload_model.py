import json
import os
import shutil
from pathlib import Path

import hydra
import torch
from huggingface_hub import HfApi, upload_folder
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def disable_peft_bitsandbytes():
    """
    Force PEFT to ignore broken bitsandbytes in this environment.

    This script only merges bf16/fp16 LoRA weights.
    It does not use 4-bit / 8-bit / QLoRA.
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


def normalize_thinking_mode(cfg: DictConfig) -> str:
    """
    Accept:
      mode: "off"
      mode: off
      mode: false
      mode: "false"
      mode: "on"
      mode: true
    """
    thinking_mode_raw = cfg.thinking.mode

    if isinstance(thinking_mode_raw, bool):
        thinking_mode = "on" if thinking_mode_raw else "off"
    else:
        thinking_mode = str(thinking_mode_raw).lower().strip()

    if thinking_mode in {"false", "no", "0"}:
        thinking_mode = "off"
    elif thinking_mode in {"true", "yes", "1"}:
        thinking_mode = "on"

    if thinking_mode not in {"off", "on"}:
        raise ValueError(
            f"thinking.mode must be 'off' or 'on', got {cfg.thinking.mode!r}"
        )

    return thinking_mode


def build_chat_template_from_official(tokenizer, cfg: DictConfig) -> str:
    """
    Patch the official Qwen3 chat template instead of rewriting it.

    We prepend a small Jinja header that:
      1. forces enable_thinking true/false;
      2. injects the safety boxed-answer instruction as a system message;
      3. then lets the official Qwen3 template handle all formatting.

    This is important because CI only calls:

        tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    so prompt preference must be baked into the tokenizer template itself.
    """
    official_template = tokenizer.chat_template

    if official_template is None:
        raise ValueError("The base tokenizer does not have a chat_template.")

    safety_system = str(cfg.prompt.system).strip()
    safety_system_json = json.dumps(safety_system, ensure_ascii=False)

    thinking_mode = normalize_thinking_mode(cfg)
    enable_thinking_value = "false" if thinking_mode == "off" else "true"

    patch_header = r"""{%- set enable_thinking = __ENABLE_THINKING__ -%}
{%- set safety_system = __SAFETY_SYSTEM__ -%}
{%- if messages|length > 0 and messages[0]["role"] == "system" -%}
{%- set messages = [{"role": "system", "content": safety_system + "\n\nAdditional system instruction:\n" + messages[0]["content"]}] + messages[1:] -%}
{%- else -%}
{%- set messages = [{"role": "system", "content": safety_system}] + messages -%}
{%- endif -%}
"""

    patch_header = patch_header.replace(
        "__ENABLE_THINKING__",
        enable_thinking_value,
    ).replace(
        "__SAFETY_SYSTEM__",
        safety_system_json,
    )

    return patch_header + "\n" + official_template


def patch_tokenizer_chat_template(tokenizer, output_dir: Path, cfg: DictConfig):
    """
    Save tokenizer with patched official Qwen3 template.
    Also explicitly writes chat_template.jinja because the CI requires it.
    """
    chat_template = build_chat_template_from_official(tokenizer, cfg)

    tokenizer.padding_side = str(cfg.tokenizer.padding_side)
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(output_dir)

    chat_template_path = output_dir / "chat_template.jinja"
    with open(chat_template_path, "w", encoding="utf-8") as f:
        f.write(chat_template)

    tokenizer_config_path = output_dir / "tokenizer_config.json"

    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    else:
        tokenizer_config = {}

    tokenizer_config["chat_template"] = chat_template
    tokenizer_config["padding_side"] = str(cfg.tokenizer.padding_side)

    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)


def save_generation_config(output_dir: Path, cfg: DictConfig):
    """
    Save generation_config.json.

    Transformers refuses to save temperature/top_p/top_k when do_sample=False,
    so we unset them in deterministic mode.
    """
    try:
        generation_config = GenerationConfig.from_pretrained(
            output_dir,
            local_files_only=True,
        )
    except Exception:
        generation_config = GenerationConfig.from_pretrained(
            cfg.base_model.name,
            trust_remote_code=cfg.base_model.trust_remote_code,
        )

    generation_config.do_sample = bool(cfg.generation.do_sample)

    if generation_config.do_sample:
        generation_config.temperature = float(cfg.generation.temperature)
        generation_config.top_p = float(cfg.generation.top_p)
        generation_config.top_k = int(cfg.generation.top_k)
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    generation_config.save_pretrained(output_dir)


def validate_output_dir(output_dir: Path):
    """
    Validate local merged model directory against the CI-style requirements.
    """
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]

    missing = []

    for file_name in required_files:
        if not (output_dir / file_name).exists():
            missing.append(file_name)

    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    safetensors_files = list(output_dir.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(
            "No .safetensors weight files found at the root of the output directory."
        )

    print("===== Local validation passed =====")
    print("Required files:")
    for file_name in required_files:
        print(" -", output_dir / file_name)

    print("Safetensors:")
    for file_path in safetensors_files:
        print(" -", file_path.name)


def preview_chat_template(output_dir: Path, cfg: DictConfig):
    """
    Verify the exact CI-style call:
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    The rendered prompt should include:
      - boxed-answer instruction
      - Qwen3 thinking mode behavior according to config
    """
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        trust_remote_code=cfg.base_model.trust_remote_code,
    )

    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": str(cfg.validation.preview_prompt).strip(),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("===== Chat template preview =====")
    print(rendered)

    if r"\boxed{A}" not in rendered:
        raise RuntimeError(
            "The rendered chat template does not contain boxed-answer instruction."
        )

    thinking_mode = normalize_thinking_mode(cfg)

    if thinking_mode == "off":
        if "<think>" not in rendered or "</think>" not in rendered:
            raise RuntimeError(
                "thinking.mode=off expects the Qwen3 non-thinking <think></think> block."
            )
    else:
        if "<think>" not in rendered:
            raise RuntimeError(
                "thinking.mode=on expects the <think> opener."
            )


def upload_to_hub(output_dir: Path, cfg: DictConfig):
    """
    Upload only the target safety_model repo.
    Does not create group_model/math_model/general_knowledge_model/multilingual_model.
    """
    token_env = str(cfg.upload.token_env)
    token = os.environ.get(token_env)

    if token is None or len(token.strip()) == 0:
        raise ValueError(
            f"Missing Hugging Face token. Set it with: export {token_env}=hf_xxx"
        )

    api = HfApi(token=token)

    print("===== Creating/checking target repo only =====")
    api.create_repo(
        repo_id=str(cfg.upload.repo_id),
        repo_type="model",
        private=bool(cfg.upload.private),
        exist_ok=True,
    )

    print("===== Uploading merged model folder =====")
    upload_folder(
        repo_id=str(cfg.upload.repo_id),
        folder_path=str(output_dir),
        repo_type="model",
        token=token,
        commit_message=str(cfg.upload.commit_message),
    )

    print("===== Upload done =====")
    print("Uploaded to:", cfg.upload.repo_id)


@hydra.main(version_base=None, config_path="./configs", config_name="merge")
def main(cfg: DictConfig):
    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    disable_peft_bitsandbytes()

    output_dir = Path(str(cfg.local.merged_output_dir))

    if bool(cfg.local.clean_output_dir) and output_dir.exists():
        print("===== Cleaning existing output directory =====")
        print(output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Paths =====")
    print("Base model:", cfg.base_model.name)
    print("Adapter path:", cfg.adapter.path)
    print("Merged output dir:", output_dir)
    print("Target repo:", cfg.upload.repo_id)

    print("===== Loading tokenizer =====")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model.name,
        trust_remote_code=cfg.base_model.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("===== Loading base model =====")
    print("dtype:", dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model.name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=cfg.base_model.trust_remote_code,
    )

    print("===== Loading LoRA adapter =====")
    model = PeftModel.from_pretrained(
        base_model,
        str(cfg.adapter.path),
        is_trainable=False,
    )

    print("===== Merging LoRA adapter into base model =====")
    model = model.merge_and_unload()
    model.eval()

    print("===== Saving merged full model =====")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=str(cfg.local.max_shard_size),
    )

    print("===== Saving tokenizer with patched official chat template =====")
    patch_tokenizer_chat_template(tokenizer, output_dir, cfg)

    print("===== Saving generation_config.json =====")
    save_generation_config(output_dir, cfg)

    print("===== Validating local output directory =====")
    validate_output_dir(output_dir)

    print("===== Previewing chat template =====")
    preview_chat_template(output_dir, cfg)

    if bool(cfg.upload.enabled):
        upload_to_hub(output_dir, cfg)
    else:
        print("===== Upload disabled =====")
        print("Local merged model is ready at:", output_dir)


if __name__ == "__main__":
    main()