#!/usr/bin/env python3
"""Generate paper-track Global-MMLU experiment configs for all languages (no CLI args)."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/netscratch/anonymous_user/projects/BA-hydra")
CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiment"
PERMANENT_ROOT = (
    PROJECT_ROOT / "evaluation" / "converted_checkpoints" / "clt_multilingual_paper_spm32k_permanent"
)

LANGUAGES = [
    ("deu_Latn", "de"),
    ("jpn_Jpan", "ja"),
    ("spa_Latn", "es"),
    ("rus_Cyrl", "ru"),
    ("pol_Latn", "pl"),
    ("por_Latn", "pt"),
]


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _model_path(lang: str, name: str) -> str:
    return str(PERMANENT_ROOT / lang / name)


def _write_one(lang: str, mmlu_code: str) -> Path:
    exp_name = f"clt_multi_mmlu_paper_{lang}"
    out_path = CONFIG_ROOT / f"{exp_name}.yaml"

    lines: list[str] = []
    lines.append("# @package _global_")
    lines.append("")
    lines.append(f"experiment_name: {_q(exp_name)}")
    lines.append(
        f"description: {_q(f'Global-MMLU paper-track A/B for {lang} (old vs paper tokenizer-trained WECHSEL).')}"
    )
    lines.append("")
    lines.append("models:")

    model_names = [
        f"hf-monolingual-{lang}",
        f"hf-monolingual-continued-{lang}",
        f"wechsel-init-{lang}",
        f"wechsel-continued-{lang}",
        f"wechsel-init-paper-spm32k-{lang}",
        f"wechsel-continued-paper-spm32k-{lang}",
    ]

    for name in model_names:
        lines.append(f"  - source_path: {_q(_model_path(lang, name))}")
        lines.append(f"    name: {_q(name)}")
        lines.append("")

    lines.append("benchmarks:")
    lines.append(f"  - name: {_q(f'global_mmlu_{mmlu_code}')}")
    lines.append(f"    tasks: [{_q(f'global_mmlu_full_{mmlu_code}')}]")
    lines.append("    shots: [0]")
    lines.append("    seed: 42")
    lines.append('    device: "cuda"')
    lines.append('    batch_size: "auto"')
    lines.append("")

    lines.append("logger:")
    lines.append("  wandb:")
    lines.append('    project: "BA-CrossLingual-Multilingual"')
    lines.append('    group: "clt-multilingual-mmlu-paper-spm32k"')
    lines.append(f"    name: {_q(exp_name)}")
    lines.append(
        "    tags: ["
        + ", ".join(
            _q(x)
            for x in [
                "clt-multilingual",
                "paper-spm32k",
                lang,
                "mmlu",
                f"global-mmlu-{mmlu_code}",
                "shots-0",
            ]
        )
        + "]"
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for lang, mmlu_code in LANGUAGES:
        written.append(_write_one(lang, mmlu_code))

    print("Generated paper MMLU configs:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
