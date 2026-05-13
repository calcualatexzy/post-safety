#!/usr/bin/env python3
"""Build a merged safety multiple-choice dataset.

The output format is JSONL with one object per line:

    {
      "prompt": "Question text\n\nA) ...\nB) ...",
      "answer": "\\boxed{A}",
      "source": "safetybench",
      "metadata": {...}
    }

The builder intentionally keeps only single-answer examples.  SafetyBench is
loaded from the released raw test question/answer files because some versions
of ``datasets`` fail to read the Hugging Face test config schema.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable


DEFAULT_CONFIG_PATH = Path("configs/mcq_data.yaml")

SAFETYBENCH_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/thu-coai/SafetyBench/main/"
    "opensource_data/test_en.json"
)
SAFETYBENCH_ANSWERS_URL = (
    "https://raw.githubusercontent.com/thu-coai/SafetyBench/main/"
    "opensource_data/test_answers_en.json"
)

SALAD_DATASET = "OpenSafetyLab/Salad-Data"
SALAD_CONFIG = "mcq_set"
SALAD_SPLIT = "train"

AEGIS_DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
AEGIS_SPLIT = "train"

# Aegis/Nemotron safety taxonomy options.  This intentionally includes more
# than 15 choices because Aegis 2.0 is used as a harm-category classification
# task, not a fixed four-choice task.
AEGIS_TAXONOMY = [
    "Safe",
    "Needs Caution",
    "Hate/Identity Hate",
    "Sexual",
    "Suicide and Self Harm",
    "Violence",
    "Guns/Illegal Weapons",
    "Threat",
    "PII/Privacy",
    "Sexual Minor",
    "Criminal Planning/Confessions",
    "Harassment",
    "Controlled/Regulated Substances",
    "Profanity",
    "Illegal Activity",
    "Immoral/Unethical",
    "Unauthorized Advice",
    "Political/Misinformation/Conspiracy",
    "Fraud/Deception",
    "Copyright/Trademark/Plagiarism",
    "High Risk Gov. Decision Making",
    "Malware",
    "Manipulation",
]


@dataclass
class BuildStats:
    loaded: int = 0
    kept: int = 0
    skipped: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.skipped is None:
            self.skipped = Counter()

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "kept": self.kept,
            "skipped": dict(self.skipped or {}),
        }


def index_to_label(index: int) -> str:
    """Convert a zero-based index to spreadsheet-like labels: A..Z, AA.."""

    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")

    label = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def label_to_index(label: str) -> int:
    label = label.strip().upper()
    if not re.fullmatch(r"[A-Z]+", label):
        raise ValueError(f"invalid option label: {label!r}")

    value = 0
    for char in label:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def boxed(label: str) -> str:
    return rf"\boxed{{{label.strip().upper()}}}"


def format_answer(label: str, answer_format: str) -> str:
    label = label.strip().upper()
    if answer_format == "plain":
        return label
    if answer_format == "boxed":
        return boxed(label)
    raise ValueError("answer_format must be one of: plain, boxed")


def format_options(options: list[str]) -> str:
    lines = []
    for index, option in enumerate(options):
        text = " ".join(str(option).strip().split())
        lines.append(f"{index_to_label(index)}) {text}")
    return "\n".join(lines)


def build_prompt(problem: str, options: list[str]) -> str:
    return f"{problem.strip()}\n\n{format_options(options)}"


def option_order_seed(seed: int, source: str, source_id: Any) -> str:
    return f"{seed}:{source}:{source_id}"


def maybe_shuffle_options(
    options: list[str],
    answer_index: int,
    *,
    enabled: bool,
    seed: str,
) -> tuple[list[str], int, list[int]]:
    """Optionally shuffle options while preserving the correct answer.

    Returns ``(new_options, new_answer_index, original_indices)`` where
    ``original_indices[new_index]`` points to the option's original index.
    """

    if answer_index < 0 or answer_index >= len(options):
        raise ValueError(f"answer index {answer_index} out of range for {len(options)} options")

    order = list(range(len(options)))
    if enabled and len(options) > 1:
        random.Random(seed).shuffle(order)

    new_options = [options[index] for index in order]
    new_answer_index = order.index(answer_index)
    return new_options, new_answer_index, order


def fetch_json(url: str, timeout: int = 60) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for SALAD and Aegis downloads. "
            "Install it with `pip install datasets`."
        ) from exc
    return load_dataset


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise RuntimeError(
            "The `huggingface_hub` package is required for Hugging Face uploads. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return HfApi, create_repo


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read mcq_format.py config files. "
            "Install it with `pip install pyyaml`, or run with a config-free path."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    return data


def resolve_config_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.parts[:1] == ("configs",):
        return path
    return config_path.parent / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def choose_arg(cli_value: Any, config: dict[str, Any], path: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config_get(config, path, default)


def normalize_prompt_templates(value: Any) -> dict[str, list[str]]:
    templates: dict[str, list[str]] = {}
    if value is None:
        return templates
    if not isinstance(value, dict):
        raise ValueError("aegis.prompt_templates must be a mapping")

    for target in ("prompt", "response", "both"):
        if target not in value:
            continue
        target_templates = value[target]
        if isinstance(target_templates, str):
            target_templates = [target_templates]
        if not isinstance(target_templates, list) or not target_templates:
            raise ValueError(f"aegis.prompt_templates.{target} must be a non-empty list")
        if not all(isinstance(template, str) and template.strip() for template in target_templates):
            raise ValueError(f"aegis.prompt_templates.{target} contains an invalid template")
        templates[target] = target_templates
    return templates


def load_prompt_template_jsonl(path: Path, allowed_targets: set[str]) -> dict[str, list[str]]:
    templates: dict[str, list[str]] = {target: [] for target in allowed_targets}
    for record in load_jsonl(path):
        target = str(record.get("target", "")).strip()
        template = record.get("template")
        if target not in allowed_targets:
            raise ValueError(f"{path}: invalid prompt template target {target!r}")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"{path}: prompt template must be non-empty text")
        templates[target].append(template)
    return {target: items for target, items in templates.items() if items}


def require_template_targets(
    templates: dict[str, list[str]],
    required_targets: Iterable[str],
    *,
    label: str,
) -> dict[str, list[str]]:
    missing = [target for target in required_targets if not templates.get(target)]
    if missing:
        raise ValueError(f"{label} missing templates for targets: {missing}")
    return templates


def normalize_category_synonyms(value: Any) -> tuple[dict[str, list[str]], dict[str, str]]:
    synonyms: dict[str, list[str]] = {}
    aliases = {canonical_category(label): label for label in AEGIS_TAXONOMY}
    if value is None:
        return synonyms, aliases
    if not isinstance(value, dict):
        raise ValueError("aegis.category_synonyms must be a mapping")

    for category, category_synonyms in value.items():
        if category not in AEGIS_TAXONOMY:
            continue
        if isinstance(category_synonyms, str):
            category_synonyms = [category_synonyms]
        if not isinstance(category_synonyms, list) or not category_synonyms:
            raise ValueError(f"aegis.category_synonyms.{category} must be a non-empty list")
        clean = [str(item).strip() for item in category_synonyms if str(item).strip()]
        if not clean:
            raise ValueError(f"aegis.category_synonyms.{category} must contain text labels")
        if category not in clean:
            clean.insert(0, category)
        synonyms[category] = clean
        for item in clean:
            aliases[canonical_category(item)] = category
    return synonyms, aliases


def load_category_synonym_jsonl(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    synonyms: dict[str, list[str]] = {}
    aliases = {canonical_category(label): label for label in AEGIS_TAXONOMY}
    for record in load_jsonl(path):
        category = str(record.get("category", "")).strip()
        if category not in AEGIS_TAXONOMY:
            raise ValueError(f"{path}: unknown Aegis category {category!r}")
        raw_synonyms = record.get("synonyms")
        if isinstance(raw_synonyms, str):
            raw_synonyms = [raw_synonyms]
        if not isinstance(raw_synonyms, list):
            raise ValueError(f"{path}: synonyms for {category!r} must be a list")
        clean_synonyms = [str(item).strip() for item in raw_synonyms if str(item).strip()]
        if category not in clean_synonyms:
            clean_synonyms.insert(0, category)
        synonyms[category] = clean_synonyms
        for item in clean_synonyms:
            aliases[canonical_category(item)] = category

        raw_aliases = record.get("aliases", [])
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        if not isinstance(raw_aliases, list):
            raise ValueError(f"{path}: aliases for {category!r} must be a list")
        for item in raw_aliases:
            text = str(item).strip()
            if text:
                aliases[canonical_category(text)] = category

    missing = [category for category in AEGIS_TAXONOMY if category not in synonyms]
    if missing:
        raise ValueError(f"{path}: missing synonyms for Aegis categories: {missing}")
    return synonyms, aliases


def normalize_target_ratios(value: Any, *, default_target: str) -> dict[str, float]:
    if value is None:
        return {default_target: 1.0}
    if not isinstance(value, dict):
        raise ValueError("aegis.target_ratios must be a mapping")

    ratios = {}
    for target, ratio in value.items():
        target = str(target).strip()
        if target not in {"prompt", "response", "both"}:
            raise ValueError(f"invalid Aegis target ratio key: {target!r}")
        ratio = float(ratio)
        if ratio < 0:
            raise ValueError("aegis.target_ratios values must be non-negative")
        if ratio > 0:
            ratios[target] = ratio
    if not ratios:
        raise ValueError("aegis.target_ratios must contain at least one positive ratio")
    return ratios


def normalize_split_ratios(value: Any) -> dict[str, float]:
    if value is None:
        return {"train": 0.8, "valid": 0.1, "test": 0.1}
    if not isinstance(value, dict):
        raise ValueError("splits.ratios must be a mapping")

    ratios: dict[str, float] = {}
    for split, ratio in value.items():
        split = str(split).strip()
        if not split:
            raise ValueError("split names must be non-empty")
        ratio = float(ratio)
        if ratio < 0:
            raise ValueError("splits.ratios values must be non-negative")
        if ratio > 0:
            ratios[split] = ratio

    if not ratios:
        raise ValueError("splits.ratios must contain at least one positive ratio")
    return ratios


def resolve_source_sampling(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sampling = config_get(config, "sampling")
    if sampling is None:
        return {
            "safetybench": {"mode": "all"},
            "salad": {"mode": "all"},
            "aegis": {"mode": "all"},
        }
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be a mapping")

    resolved = {}
    for source in ("safetybench", "salad", "aegis"):
        source_config = sampling.get(source, {"mode": "all"})
        if not isinstance(source_config, dict):
            raise ValueError(f"sampling.{source} must be a mapping")
        mode = str(source_config.get("mode", "all")).strip().lower()
        if mode not in {"all", "fraction", "fraction_of_total", "count"}:
            raise ValueError(
                f"sampling.{source}.mode must be all, fraction, fraction_of_total, or count"
            )
        item = {"mode": mode}
        if mode in {"fraction", "fraction_of_total"}:
            fraction = float(source_config.get("fraction", 1.0))
            if not (0.0 <= fraction <= 1.0):
                raise ValueError(f"sampling.{source}.fraction must be between 0 and 1")
            item["fraction"] = fraction
        elif mode == "count":
            count = int(source_config.get("count", 0))
            if count < 0:
                raise ValueError(f"sampling.{source}.count must be non-negative")
            item["count"] = count
        resolved[source] = item
    return resolved


def counts_from_sampling(
    sources: dict[str, list[dict[str, Any]]],
    sampling: dict[str, dict[str, Any]],
    *,
    strict: bool,
) -> dict[str, int]:
    caps = {name: len(examples) for name, examples in sources.items()}
    counts = {}
    fraction_of_total = {}
    for name, config in sampling.items():
        mode = config["mode"]
        if mode == "all":
            counts[name] = caps[name]
        elif mode == "fraction":
            counts[name] = int(round(caps[name] * float(config["fraction"])))
        elif mode == "fraction_of_total":
            fraction_of_total[name] = float(config["fraction"])
        elif mode == "count":
            counts[name] = int(config["count"])

    if fraction_of_total:
        fixed_total = sum(counts.values())
        requested_fraction_total = sum(fraction_of_total.values())
        if requested_fraction_total >= 1.0:
            raise ValueError("sum of fraction_of_total sampling fractions must be < 1")
        final_total = fixed_total / (1.0 - requested_fraction_total)
        for name, fraction in fraction_of_total.items():
            counts[name] = int(round(final_total * fraction))

    shortages = {
        name: {"requested": count, "available": caps[name]}
        for name, count in counts.items()
        if count > caps[name]
    }
    if shortages and strict:
        raise ValueError(f"not enough examples for requested sampling counts: {shortages}")
    return {name: min(count, caps[name]) for name, count in counts.items()}


def available_aegis_target_ratios(
    row: dict[str, Any],
    target_ratios: dict[str, float],
) -> dict[str, float]:
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or "").strip()
    available = {}
    for target, ratio in target_ratios.items():
        if target == "prompt" and prompt:
            available[target] = ratio
        elif target == "response" and response:
            available[target] = ratio
        elif target == "both" and (prompt or response):
            available[target] = ratio
    return available


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)

    resolved = argparse.Namespace(**vars(args))
    resolved.output = Path(choose_arg(args.output, config, "output", "data/safety_mcq.jsonl"))
    resolved.samples = args.samples
    resolved.seed = int(choose_arg(args.seed, config, "seed", 42))
    resolved.strict = bool(choose_arg(args.strict, config, "strict", True))
    resolved.shuffle_options = bool(config_get(config, "shuffle_options", True))
    resolved.answer_format = str(config_get(config, "answer_format", "plain"))
    if resolved.answer_format not in {"plain", "boxed"}:
        raise ValueError("answer_format must be one of: plain, boxed")
    resolved.include_source = bool(config_get(config, "include_source", False))
    resolved.include_metadata = bool(config_get(config, "include_metadata", False))
    resolved.splits_enabled = bool(choose_arg(args.splits, config, "splits.enabled", False))
    resolved.split_output_dir = Path(
        choose_arg(args.split_output_dir, config, "splits.output_dir", "data/mcq_safety")
    )
    resolved.split_seed = int(config_get(config, "splits.seed", resolved.seed))
    resolved.split_ratios = normalize_split_ratios(config_get(config, "splits.ratios"))
    resolved.split_metadata_json_string = bool(
        config_get(config, "splits.metadata_json_string", True)
    )

    resolved.hf_repo_id = str(config_get(config, "huggingface.repo_id", "")).strip()
    resolved.hf_repo_type = str(config_get(config, "huggingface.repo_type", "dataset")).strip()
    resolved.hf_path_in_repo = str(config_get(config, "huggingface.path_in_repo", ".")).strip()
    resolved.hf_create_repo = bool(config_get(config, "huggingface.create_repo", True))
    resolved.hf_upload = bool(choose_arg(args.hf_upload, config, "huggingface.upload", False))
    resolved.hf_dry_run = bool(choose_arg(args.hf_dry_run, config, "huggingface.dry_run", False))
    resolved.hf_commit_message = str(
        config_get(config, "huggingface.commit_message", "Upload MCQ safety split files")
    )
    if resolved.hf_upload or resolved.hf_dry_run:
        if not resolved.hf_repo_id:
            raise ValueError("huggingface.repo_id is required for upload or dry-run checks")
        if resolved.hf_repo_type != "dataset":
            raise ValueError("only huggingface.repo_type=dataset is supported")

    resolved.source_sampling = resolve_source_sampling(config)

    resolved.safetybench_question_url = str(
        config_get(config, "safetybench.question_url", SAFETYBENCH_QUESTIONS_URL)
    )
    resolved.safetybench_answer_url = str(
        config_get(config, "safetybench.answer_url", SAFETYBENCH_ANSWERS_URL)
    )

    resolved.salad_dataset = str(config_get(config, "salad.dataset", SALAD_DATASET))
    resolved.salad_config = str(config_get(config, "salad.config", SALAD_CONFIG))
    resolved.salad_split = str(config_get(config, "salad.split", SALAD_SPLIT))
    resolved.aegis_dataset = str(config_get(config, "aegis.dataset", AEGIS_DATASET))
    resolved.aegis_split = str(config_get(config, "aegis.split", AEGIS_SPLIT))
    resolved.aegis_target = str(config_get(config, "aegis.target", "prompt"))
    if resolved.aegis_target not in {"prompt", "response", "both"}:
        raise ValueError("aegis.target must be one of: prompt, response, both")
    resolved.aegis_target_ratios = normalize_target_ratios(
        config_get(config, "aegis.target_ratios"),
        default_target=resolved.aegis_target,
    )

    template_file = config_get(config, "aegis.prompt_template_file")
    if template_file is not None:
        resolved.aegis_prompt_templates = load_prompt_template_jsonl(
            resolve_config_path(args.config, template_file),
            {"prompt", "response", "both"},
        )
    else:
        resolved.aegis_prompt_templates = normalize_prompt_templates(
            config_get(config, "aegis.prompt_templates")
        )
    resolved.aegis_prompt_templates = require_template_targets(
        resolved.aegis_prompt_templates,
        resolved.aegis_target_ratios.keys(),
        label="aegis prompt templates",
    )

    synonym_file = config_get(config, "aegis.category_synonym_file")
    if synonym_file is not None:
        (
            resolved.aegis_category_synonyms,
            resolved.aegis_category_aliases,
        ) = load_category_synonym_jsonl(resolve_config_path(args.config, synonym_file))
    else:
        (
            resolved.aegis_category_synonyms,
            resolved.aegis_category_aliases,
        ) = normalize_category_synonyms(config_get(config, "aegis.category_synonyms"))
    missing_synonyms = [
        category for category in AEGIS_TAXONOMY if category not in resolved.aegis_category_synonyms
    ]
    if missing_synonyms:
        raise ValueError(f"missing Aegis category synonyms: {missing_synonyms}")
    resolved.aegis_option_count_min = int(config_get(config, "aegis.option_count_min", 3))
    resolved.aegis_option_count_max = int(config_get(config, "aegis.option_count_max", 22))
    resolved.aegis_binary_option_ratio = float(config_get(config, "aegis.binary_option_ratio", 0.1))
    resolved.aegis_binary_unsafe_labels = config_get(
        config, "aegis.binary_unsafe_labels", ["Unsafe", "Harmful"]
    )
    if isinstance(resolved.aegis_binary_unsafe_labels, str):
        resolved.aegis_binary_unsafe_labels = [resolved.aegis_binary_unsafe_labels]
    resolved.aegis_binary_unsafe_labels = [
        str(label).strip() for label in resolved.aegis_binary_unsafe_labels if str(label).strip()
    ]
    if not resolved.aegis_binary_unsafe_labels:
        raise ValueError("aegis.binary_unsafe_labels must contain at least one label")
    if not (0.0 <= resolved.aegis_binary_option_ratio <= 1.0):
        raise ValueError("aegis.binary_option_ratio must be between 0 and 1")
    if not (
        2
        <= resolved.aegis_option_count_min
        <= resolved.aegis_option_count_max
        <= len(AEGIS_TAXONOMY)
    ):
        raise ValueError(
            f"aegis option count range must satisfy 2 <= min <= max <= {len(AEGIS_TAXONOMY)}"
        )
    return resolved


def normalize_safetybench(
    question_url: str,
    answer_url: str,
    *,
    strict: bool = True,
    shuffle_options: bool = True,
    seed: int = 42,
    answer_format: str = "plain",
) -> tuple[list[dict[str, Any]], BuildStats]:
    questions = fetch_json(question_url)
    answers = fetch_json(answer_url)
    stats = BuildStats(loaded=len(questions))
    examples: list[dict[str, Any]] = []

    if not isinstance(questions, list):
        raise ValueError("SafetyBench questions JSON must be a list")
    if not isinstance(answers, dict):
        raise ValueError("SafetyBench answers JSON must be an object keyed by id")

    for row in questions:
        source_id = row.get("id")
        answer_row = answers.get(str(source_id))
        if not answer_row:
            stats.skip("missing_answer")
            continue

        question = row.get("question")
        options = row.get("options")
        answer_index = answer_row.get("answer")
        category = row.get("category")

        if not isinstance(question, str) or not question.strip():
            stats.skip("invalid_question")
            continue
        if not isinstance(options, list) or len(options) < 2:
            stats.skip("invalid_options")
            continue
        if not isinstance(answer_index, int):
            stats.skip("invalid_answer")
            continue
        if answer_index < 0 or answer_index >= len(options):
            stats.skip("answer_out_of_range")
            continue

        answer_category = answer_row.get("category")
        if strict and answer_category and category and answer_category != category:
            raise ValueError(
                "SafetyBench category mismatch for id "
                f"{source_id}: question={category!r}, answer={answer_category!r}"
            )

        shuffled_options, shuffled_answer_index, option_order = maybe_shuffle_options(
            [str(option) for option in options],
            answer_index,
            enabled=shuffle_options,
            seed=option_order_seed(seed, "safetybench", source_id),
        )
        label = index_to_label(shuffled_answer_index)
        examples.append(
            {
                "prompt": build_prompt(question, shuffled_options),
                "answer": format_answer(label, answer_format),
                "source": "safetybench",
                "metadata": {
                    "source_dataset": "SafetyBench",
                    "source_id": source_id,
                    "category": category,
                    "answer_index": answer_index,
                    "shuffled_answer_index": shuffled_answer_index,
                    "option_order_original_indices": option_order,
                    "question_url": question_url,
                    "answer_url": answer_url,
                },
            }
        )

    stats.kept = len(examples)
    return examples, stats


def parse_salad_choices(choices: Any) -> list[tuple[str, str]]:
    """Parse SALAD choice strings such as ``A. safe; B. unsafe; C. ...``."""

    if isinstance(choices, list):
        return [(index_to_label(index), str(text).strip()) for index, text in enumerate(choices)]

    if not isinstance(choices, str):
        return []

    text = choices.strip()
    pattern = re.compile(r"(?:^|;\s*)([A-Z])\.\s*(.*?)(?=;\s*[A-Z]\.\s*|$)", re.S)
    parsed = [(match.group(1), match.group(2).strip(" ;\n\t")) for match in pattern.finditer(text)]
    return [(label, option) for label, option in parsed if option]


def salad_choice_kind_from_mcq(mcq: Any) -> str:
    if not isinstance(mcq, str):
        return "safe"

    lowered = mcq.lower()
    if "choose the unsafe answer" in lowered:
        return "unsafe"
    if "choose the safe answer" in lowered:
        return "safe"
    return "safe"


def normalize_salad(
    dataset_name: str,
    config: str,
    split: str,
    *,
    shuffle_options: bool = True,
    seed: int = 42,
    answer_format: str = "plain",
) -> tuple[list[dict[str, Any]], BuildStats]:
    load_dataset = require_datasets()
    dataset = load_dataset(dataset_name, config, split=split)
    stats = BuildStats(loaded=len(dataset))
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        gt = str(row.get("gt", "")).strip().upper()
        if not re.fullmatch(r"[A-Z]", gt):
            stats.skip("multi_or_invalid_ground_truth")
            continue

        parsed = parse_salad_choices(row.get("choices"))
        if len(parsed) < 2:
            stats.skip("invalid_choices")
            continue

        labels = [label for label, _ in parsed]
        if gt not in labels:
            stats.skip("ground_truth_not_in_choices")
            continue

        # Re-index options to A), B), ... in case the source labels have gaps.
        options = [option for _, option in parsed]
        answer_index = labels.index(gt)
        base_question = row.get("baseq") or row.get("question") or row.get("mcq")
        if not isinstance(base_question, str) or not base_question.strip():
            stats.skip("invalid_question")
            continue

        choice_kind = salad_choice_kind_from_mcq(row.get("mcq"))
        shuffled_options, shuffled_answer_index, option_order = maybe_shuffle_options(
            options,
            answer_index,
            enabled=shuffle_options,
            seed=option_order_seed(seed, "salad", index),
        )
        answer_label = index_to_label(shuffled_answer_index)

        examples.append(
            {
                "prompt": build_prompt(str(base_question), shuffled_options),
                "answer": format_answer(answer_label, answer_format),
                "source": "salad",
                "metadata": {
                    "source_dataset": dataset_name,
                    "source_index": index,
                    "original_gt": gt,
                    "choice_kind": choice_kind,
                    "answer_index": answer_index,
                    "shuffled_answer_index": shuffled_answer_index,
                    "option_order_original_indices": option_order,
                    "category_1": row.get("1-category"),
                    "category_2": row.get("2-category"),
                    "category_3": row.get("3-category"),
                    "dataset": dataset_name,
                    "config": config,
                    "split": split,
                },
            }
        )

    stats.kept = len(examples)
    return examples, stats


def canonical_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def map_aegis_category(
    label: Any,
    violated_categories: Any,
    *,
    aliases: dict[str, str],
    stats: BuildStats | None = None,
) -> str | None:
    label_text = str(label or "").strip().lower()
    violated_text = str(violated_categories or "").strip()

    if label_text == "safe":
        return "Safe"

    parts = [part.strip() for part in violated_text.split(",") if part.strip()]
    if label_text in {"needs caution", "needs_caution", "caution"} and not parts:
        return "Needs Caution"

    if label_text in {"needs caution", "needs_caution", "caution"}:
        parts = ["Needs Caution", *parts]

    for part in parts:
        mapped = aliases.get(canonical_category(part))
        if mapped:
            return mapped

    if label_text in {"unsafe", "needs caution", "needs_caution", "caution"}:
        if stats is not None:
            stats.skip(f"unmapped_category:{violated_text or label_text}")
        return None

    if stats is not None:
        stats.skip(f"invalid_label:{label_text or '<empty>'}")
    return None


def render_aegis_template(
    templates: dict[str, list[str]],
    target: str,
    index: int,
    *,
    prompt: str,
    response: str,
) -> tuple[str, str]:
    target_templates = templates[target]
    template_index = index % len(target_templates)
    template = target_templates[template_index]
    text = template.format(
        prompt=prompt or "[missing]",
        response=response or "[missing]",
        text=prompt or response or "[missing]",
    ).strip()
    return text, f"{target}:{template_index}"


def aegis_problem(
    row: dict[str, Any],
    target: str,
    templates: dict[str, list[str]],
    index: int,
) -> tuple[str | None, str | None, str | None]:
    prompt = str(row.get("prompt") or "").strip()
    response = str(row.get("response") or "").strip()

    if target == "prompt":
        if not prompt:
            return None, None, None
        text, template_id = render_aegis_template(
            templates, target, index, prompt=prompt, response=response
        )
        return text, row.get("prompt_label"), template_id

    if target == "response":
        if not response:
            return None, None, None
        text, template_id = render_aegis_template(
            templates, target, index, prompt=prompt, response=response
        )
        return text, row.get("response_label"), template_id

    if not prompt and not response:
        return None, None, None

    text, template_id = render_aegis_template(
        templates, target, index, prompt=prompt, response=response
    )
    # For a combined interaction, prefer an unsafe/needs-caution response label
    # when present; otherwise classify the prompt.
    response_label = str(row.get("response_label") or "").strip().lower()
    if response_label and response_label != "safe":
        return text, row.get("response_label"), template_id
    return text, row.get("prompt_label"), template_id


def build_aegis_options(
    category: str,
    *,
    synonyms: dict[str, list[str]],
    seed: str,
    option_count_min: int,
    option_count_max: int,
    binary_option_ratio: float,
    binary_unsafe_labels: list[str],
) -> tuple[list[str], int, list[str], str]:
    """Build variable Aegis MCQ options while always including the correct label."""

    rng = random.Random(seed)
    if rng.random() < binary_option_ratio:
        option_categories = ["Safe", "__unsafe__"]
        options = [rng.choice(synonyms.get("Safe", ["Safe"])), rng.choice(binary_unsafe_labels)]
        answer_index = 0 if category == "Safe" else 1
        return options, answer_index, option_categories, "binary"

    count = rng.randint(option_count_min, option_count_max)
    distractor_pool = [candidate for candidate in AEGIS_TAXONOMY if candidate != category]
    option_categories = [category] + rng.sample(
        distractor_pool,
        k=min(count - 1, len(distractor_pool)),
    )
    rng.shuffle(option_categories)
    answer_index = option_categories.index(category)
    options = [
        rng.choice(synonyms.get(option_category, [option_category]))
        for option_category in option_categories
    ]
    return options, answer_index, option_categories, "category"


def normalize_aegis(
    dataset_name: str,
    split: str,
    *,
    target_ratios: dict[str, float],
    prompt_templates: dict[str, list[str]],
    category_synonyms: dict[str, list[str]],
    category_aliases: dict[str, str],
    option_count_min: int,
    option_count_max: int,
    binary_option_ratio: float,
    binary_unsafe_labels: list[str],
    shuffle_options: bool = True,
    seed: int = 42,
    answer_format: str = "plain",
) -> tuple[list[dict[str, Any]], BuildStats]:
    load_dataset = require_datasets()
    dataset = load_dataset(dataset_name, split=split)
    stats = BuildStats(loaded=len(dataset))
    target_pools: dict[str, list[dict[str, Any]]] = {target: [] for target in target_ratios}

    for index, row in enumerate(dataset):
        available_ratios = available_aegis_target_ratios(row, target_ratios)
        if not available_ratios:
            stats.skip("missing_text")
            continue

        for target in available_ratios:
            problem, label, template_id = aegis_problem(row, target, prompt_templates, index)
            if not problem:
                continue

            category = map_aegis_category(
                label,
                row.get("violated_categories"),
                aliases=category_aliases,
                stats=stats,
            )
            if category is None:
                continue

            source_key = row.get("id") or index
            target_key = f"{source_key}:{target}"
            options, answer_index, option_categories, option_mode = build_aegis_options(
                category,
                synonyms=category_synonyms,
                seed=option_order_seed(seed, "aegis-options", target_key),
                option_count_min=option_count_min,
                option_count_max=option_count_max,
                binary_option_ratio=binary_option_ratio,
                binary_unsafe_labels=binary_unsafe_labels,
            )
            shuffled_options, shuffled_answer_index, option_order = maybe_shuffle_options(
                options,
                answer_index,
                enabled=shuffle_options,
                seed=option_order_seed(seed, "aegis-shuffle", target_key),
            )
            answer_label = index_to_label(shuffled_answer_index)
            shuffled_option_categories = [
                option_categories[old_index] for old_index in option_order
            ]
            target_pools[target].append(
                {
                    "prompt": build_prompt(problem, shuffled_options),
                    "answer": format_answer(answer_label, answer_format),
                    "source": "aegis",
                    "metadata": {
                        "source_dataset": dataset_name,
                        "source_index": index,
                        "source_id": row.get("id"),
                        "target": target,
                        "prompt_template_id": template_id,
                        "category": category,
                        "option_mode": option_mode,
                        "option_categories": shuffled_option_categories,
                        "answer_index": answer_index,
                        "shuffled_answer_index": shuffled_answer_index,
                        "option_order_original_indices": option_order,
                        "prompt_label": row.get("prompt_label"),
                        "response_label": row.get("response_label"),
                        "violated_categories": row.get("violated_categories"),
                        "dataset": dataset_name,
                        "split": split,
                    },
                }
            )

    target_caps = {target: len(examples) for target, examples in target_pools.items()}
    balanced_total = max_balanced_total(target_ratios, target_caps)
    target_counts = allocate_counts(balanced_total, target_ratios, target_caps, strict=True)
    examples = sample_and_merge(target_pools, target_counts, seed + 17)

    stats.kept = len(examples)
    return examples, stats


def allocate_counts(
    total: int,
    ratios: dict[str, float],
    caps: dict[str, int] | None = None,
    *,
    strict: bool = True,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not ratios:
        raise ValueError("at least one ratio is required")
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")

    normalized = {name: value / ratio_sum for name, value in ratios.items()}
    raw = {name: total * ratio for name, ratio in normalized.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())

    for name, _ in sorted(
        raw.items(),
        key=lambda item: (item[1] - math.floor(item[1]), item[0]),
        reverse=True,
    ):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    if caps:
        shortages = {
            name: {"requested": count, "available": caps.get(name, 0)}
            for name, count in counts.items()
            if caps.get(name, 0) < count
        }
        if shortages and strict:
            raise ValueError(f"not enough examples for requested ratios: {shortages}")
        if shortages:
            return {name: min(count, caps.get(name, 0)) for name, count in counts.items()}

    return counts


def max_balanced_total(ratios: dict[str, float], caps: dict[str, int]) -> int:
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")
    normalized = {name: value / ratio_sum for name, value in ratios.items()}
    return min(int(caps[name] // ratio) for name, ratio in normalized.items() if ratio > 0)


def option_labels_in_prompt(prompt: str) -> set[str]:
    return set(re.findall(r"(?m)^([A-Z]+)\) ", prompt))


def answer_label(answer: str) -> str | None:
    if re.fullmatch(r"[A-Z]+", answer):
        return answer
    match = re.fullmatch(r"\\boxed\{([A-Z]+)\}", answer)
    return match.group(1) if match else None


def validate_examples(examples: list[dict[str, Any]]) -> None:
    for index, example in enumerate(examples):
        prompt = example.get("prompt")
        answer = example.get("answer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row {index}: missing prompt")
        if not isinstance(answer, str):
            raise ValueError(f"row {index}: missing answer")

        label = answer_label(answer)
        if not label:
            raise ValueError(f"row {index}: answer is not a single label: {answer!r}")
        labels = option_labels_in_prompt(prompt)
        if label not in labels:
            raise ValueError(
                f"row {index}: answer label {label!r} not present in "
                f"prompt options {sorted(labels)}"
            )


def write_jsonl(path: Path, examples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def output_row(
    example: dict[str, Any],
    *,
    include_source: bool,
    include_metadata: bool,
    metadata_json_string: bool = False,
) -> dict[str, Any]:
    row = {
        "prompt": example["prompt"],
        "answer": example["answer"],
    }
    if include_source:
        row["source"] = example["source"]
    if include_metadata:
        metadata = example["metadata"]
        if metadata_json_string:
            metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        row["metadata"] = metadata
    return row


def write_output_jsonl(
    path: Path,
    examples: Iterable[dict[str, Any]],
    *,
    include_source: bool,
    include_metadata: bool,
    metadata_json_string: bool = False,
) -> None:
    write_jsonl(
        path,
        (
            output_row(
                example,
                include_source=include_source,
                include_metadata=include_metadata,
                metadata_json_string=metadata_json_string,
            )
            for example in examples
        ),
    )


def split_examples(
    examples: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    counts = allocate_counts(len(examples), ratios, strict=True)
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split, count in counts.items():
        splits[split] = shuffled[cursor : cursor + count]
        cursor += count
    return splits


def split_summary(
    splits: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    ratios: dict[str, float],
    output_dir: Path,
    metadata_json_string: bool,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "seed": seed,
        "ratios": ratios,
        "total": sum(len(rows) for rows in splits.values()),
        "metadata_json_string": metadata_json_string,
        "splits": {
            name: {
                "rows": len(rows),
                "source_counts": dict(Counter(row["source"] for row in rows)),
            }
            for name, rows in splits.items()
        },
    }


def write_split_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    splits = summary["splits"]
    table = "\n".join(f"| {name} | {data['rows']} |" for name, data in splits.items())
    data_files_yaml = "".join(
        f"  - split: {name}\n    path: {name}.jsonl\n" for name in splits
    )
    metadata_description = (
        "JSON-encoded source and normalization metadata string"
        if summary["metadata_json_string"]
        else "source and normalization metadata object"
    )
    readme = f"""---
language:
- en
task_categories:
- text-classification
- question-answering
task_ids:
- multiple-choice-qa
pretty_name: MCQ Safety
configs:
- config_name: default
  data_files:
{data_files_yaml}
---

# MCQ Safety

Merged safety multiple-choice dataset built from SafetyBench test-en, SALAD
Bench MCQ data, and Aegis 2.0 safety category data.

## Splits

Deterministic random split with seed `{summary['seed']}`:

| split | rows |
|---|---:|
{table}

## Format

Each JSONL row contains:

- `prompt`: problem plus options formatted as `A) ...`, `B) ...`
- `answer`: single boxed option label, e.g. `\\boxed{{C}}`
- `source`: source dataset name
- `metadata`: {metadata_description}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_split_files(
    output_dir: Path,
    examples: list[dict[str, Any]],
    *,
    ratios: dict[str, float],
    seed: int,
    include_source: bool,
    include_metadata: bool,
    metadata_json_string: bool,
) -> dict[str, Any]:
    splits = split_examples(examples, ratios, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, rows in splits.items():
        write_output_jsonl(
            output_dir / f"{split}.jsonl",
            rows,
            include_source=include_source,
            include_metadata=include_metadata,
            metadata_json_string=metadata_json_string,
        )

    summary = split_summary(
        splits,
        seed=seed,
        ratios=ratios,
        output_dir=output_dir,
        metadata_json_string=metadata_json_string,
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_split_readme(output_dir, summary)
    return summary


def upload_to_huggingface(
    folder: Path,
    *,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    create_repo: bool,
    commit_message: str,
    dry_run: bool,
) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(folder)),
            "bytes": path.stat().st_size,
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise ValueError(f"no files found to upload in {folder}")

    HfApi, hf_create_repo = require_huggingface_hub()
    api = HfApi()
    whoami = api.whoami()

    result = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "folder": str(folder),
        "dry_run": dry_run,
        "user": whoami.get("name"),
        "files": files,
    }
    if dry_run:
        return result

    if create_repo:
        hf_create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    result["commit_url"] = str(commit)
    return result


def sample_and_merge(
    sources: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    merged: list[dict[str, Any]] = []
    for offset, (name, examples) in enumerate(sources.items()):
        shuffled = list(examples)
        random.Random(seed + offset + 1).shuffle(shuffled)
        merged.extend(shuffled[: counts[name]])
    rng.shuffle(merged)
    return merged


def sample_final_examples(
    examples: list[dict[str, Any]],
    samples: int | None,
    seed: int,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    if samples is None:
        return examples
    if samples < 0:
        raise ValueError("--samples must be non-negative")
    if samples > len(examples):
        if strict:
            raise ValueError(f"--samples={samples} exceeds dataset size {len(examples)}")
        samples = len(examples)
    shuffled = list(examples)
    random.Random(f"{seed}:final-samples").shuffle(shuffled)
    return shuffled[:samples]


def run_self_test() -> None:
    assert index_to_label(0) == "A"
    assert index_to_label(3) == "D"
    assert index_to_label(25) == "Z"
    assert index_to_label(26) == "AA"
    assert label_to_index("AA") == 26
    assert boxed("c") == r"\boxed{C}"
    assert format_answer("c", "plain") == "C"
    assert format_answer("c", "boxed") == r"\boxed{C}"
    assert format_options(["one", "two"]) == "A) one\nB) two"

    parsed = parse_salad_choices("A. one; B. two; C. three;")
    assert parsed == [("A", "one"), ("B", "two"), ("C", "three")]
    assert re.fullmatch(r"[A-Z]", "A")
    assert not re.fullmatch(r"[A-Z]", "BC")
    assert (
        salad_choice_kind_from_mcq("Here is the multi-choice question, choose the safe answer.")
        == "safe"
    )
    assert (
        salad_choice_kind_from_mcq("Here is the multi-choice question, choose the unsafe answer.")
        == "unsafe"
    )

    stats = BuildStats()
    test_synonyms = {category: [category] for category in AEGIS_TAXONOMY}
    test_synonyms["Violence"] = ["Violence", "Violent Harm"]
    test_aliases = {canonical_category(category): category for category in AEGIS_TAXONOMY}
    assert map_aegis_category("safe", "", aliases=test_aliases, stats=stats) == "Safe"
    assert (
        map_aegis_category("unsafe", "Violence, Needs Caution", aliases=test_aliases, stats=stats)
        == "Violence"
    )
    sampled_options, sampled_answer, sampled_categories, sampled_mode = build_aegis_options(
        "Violence",
        synonyms=test_synonyms,
        seed="sampled-options",
        option_count_min=3,
        option_count_max=5,
        binary_option_ratio=0.0,
        binary_unsafe_labels=["Unsafe"],
    )
    assert sampled_mode == "category"
    assert 3 <= len(sampled_options) <= 5
    assert sampled_categories[sampled_answer] == "Violence"
    binary_options, binary_answer, binary_categories, binary_mode = build_aegis_options(
        "Violence",
        synonyms=test_synonyms,
        seed="binary-options",
        option_count_min=3,
        option_count_max=5,
        binary_option_ratio=1.0,
        binary_unsafe_labels=["Unsafe"],
    )
    assert binary_mode == "binary"
    assert binary_options == ["Safe", "Unsafe"] or binary_options[1] == "Unsafe"
    assert binary_categories[binary_answer] == "__unsafe__"
    shuffled, new_answer, order = maybe_shuffle_options(
        ["a", "b", "c"], 1, enabled=True, seed="unit-test"
    )
    assert shuffled[new_answer] == "b"
    assert order[new_answer] == 1
    same, same_answer, same_order = maybe_shuffle_options(
        ["a", "b", "c"], 1, enabled=False, seed="unit-test"
    )
    assert same == ["a", "b", "c"]
    assert same_answer == 1
    assert same_order == [0, 1, 2]
    rendered, template_id = render_aegis_template(
        {
            "prompt": ["Template one: {prompt}", "Template two: {prompt}"],
            "response": ["Response: {response}"],
            "both": ["Both: {prompt} / {response}"],
        },
        "prompt",
        1,
        prompt="hello",
        response="world",
    )
    assert rendered == "Template two: hello"
    assert template_id == "prompt:1"
    assert allocate_counts(
        10, {"safetybench": 0.4, "salad": 0.3, "aegis": 0.3}
    ) == {"safetybench": 4, "salad": 3, "aegis": 3}
    assert allocate_counts(
        30, {"safetybench": 0.4, "salad": 0.3, "aegis": 0.3}
    ) == {"safetybench": 12, "salad": 9, "aegis": 9}
    assert allocate_counts(
        10, {"train": 0.8, "valid": 0.1, "test": 0.1}
    ) == {"train": 8, "valid": 1, "test": 1}

    unit_examples = [
        {
            "prompt": build_prompt(f"Question {index}?", ["x", "y", "z"]),
            "answer": "C",
            "source": "test",
            "metadata": {"index": index},
        }
        for index in range(10)
    ]
    validate_examples(unit_examples)
    unit_splits = split_examples(unit_examples, {"train": 0.8, "valid": 0.1, "test": 0.1}, 42)
    assert {name: len(rows) for name, rows in unit_splits.items()} == {
        "train": 8,
        "valid": 1,
        "test": 1,
    }
    with TemporaryDirectory() as temp_dir:
        summary = write_split_files(
            Path(temp_dir),
            unit_examples,
            ratios={"train": 0.8, "valid": 0.1, "test": 0.1},
            seed=42,
            include_source=True,
            include_metadata=True,
            metadata_json_string=True,
        )
        assert summary["total"] == 10
        assert (Path(temp_dir) / "train.jsonl").exists()
        first_train = json.loads((Path(temp_dir) / "train.jsonl").read_text().splitlines()[0])
        assert isinstance(first_train["metadata"], str)
    print("self-test: ok")


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safetybench, safetybench_stats = normalize_safetybench(
        args.safetybench_question_url,
        args.safetybench_answer_url,
        strict=args.strict,
        shuffle_options=args.shuffle_options,
        seed=args.seed,
        answer_format=args.answer_format,
    )
    salad, salad_stats = normalize_salad(
        args.salad_dataset,
        args.salad_config,
        args.salad_split,
        shuffle_options=args.shuffle_options,
        seed=args.seed,
        answer_format=args.answer_format,
    )
    aegis, aegis_stats = normalize_aegis(
        args.aegis_dataset,
        args.aegis_split,
        target_ratios=args.aegis_target_ratios,
        prompt_templates=args.aegis_prompt_templates,
        category_synonyms=args.aegis_category_synonyms,
        category_aliases=args.aegis_category_aliases,
        option_count_min=args.aegis_option_count_min,
        option_count_max=args.aegis_option_count_max,
        binary_option_ratio=args.aegis_binary_option_ratio,
        binary_unsafe_labels=args.aegis_binary_unsafe_labels,
        shuffle_options=args.shuffle_options,
        seed=args.seed,
        answer_format=args.answer_format,
    )

    sources = {
        "safetybench": safetybench,
        "salad": salad,
        "aegis": aegis,
    }
    counts = counts_from_sampling(
        sources,
        args.source_sampling,
        strict=args.strict,
    )
    full_examples = sample_and_merge(sources, counts, args.seed)
    examples = sample_final_examples(full_examples, args.samples, args.seed, strict=args.strict)
    validate_examples(examples)

    stats = {
        "requested_samples": args.samples,
        "configured_total": len(full_examples),
        "actual_total": len(examples),
        "seed": args.seed,
        "config": str(args.config),
        "shuffle_options": args.shuffle_options,
        "answer_format": args.answer_format,
        "include_source": args.include_source,
        "include_metadata": args.include_metadata,
        "source_sampling": args.source_sampling,
        "aegis_target_ratios": args.aegis_target_ratios,
        "aegis_option_count_range": [
            args.aegis_option_count_min,
            args.aegis_option_count_max,
        ],
        "aegis_binary_option_ratio": args.aegis_binary_option_ratio,
        "counts": dict(Counter(example["source"] for example in examples)),
        "target_counts": counts,
        "source_stats": {
            "safetybench": safetybench_stats.as_dict(),
            "salad": salad_stats.as_dict(),
            "aegis": aegis_stats.as_dict(),
        },
    }
    return examples, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config file for dataset sources, sampling, output, and prompt templates.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--samples",
        "--total-samples",
        dest="samples",
        type=int,
        default=None,
        help="Randomly choose this many rows from the configured full dataset.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--splits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write train/valid/test split files using the split config.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="Directory for split JSONL files, README.md, and split_summary.json.",
    )
    parser.add_argument(
        "--hf-upload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload the split output directory to the configured Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--hf-dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Check Hugging Face auth and upload manifest without creating a commit.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail when requested sampling counts cannot be met or source integrity checks fail.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run local unit checks and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(parse_args(argv))

    if args.self_test:
        run_self_test()
        return 0

    examples, stats = build_dataset(args)
    write_output_jsonl(
        args.output,
        examples,
        include_source=args.include_source,
        include_metadata=args.include_metadata,
    )

    result = {"output": str(args.output), **stats}
    if args.splits_enabled:
        result["splits"] = write_split_files(
            args.split_output_dir,
            examples,
            ratios=args.split_ratios,
            seed=args.split_seed,
            include_source=args.include_source,
            include_metadata=args.include_metadata,
            metadata_json_string=args.split_metadata_json_string,
        )

    if args.hf_upload or args.hf_dry_run:
        result["huggingface"] = upload_to_huggingface(
            args.split_output_dir,
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            path_in_repo=args.hf_path_in_repo,
            create_repo=args.hf_create_repo,
            commit_message=args.hf_commit_message,
            dry_run=args.hf_dry_run or not args.hf_upload,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
