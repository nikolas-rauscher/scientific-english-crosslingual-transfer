#!/usr/bin/env python3
"""Train paper-track target tokenizers for all multilingual languages.

No CLI arguments by design.

Per language:
- train SentencePiece BPE tokenizer on
  data/languages/<LANG>/splits/sub/sub_charcap43gb_seed42/train/docs.parquet
- export T5-compatible tokenizer with extra_ids=100
- write tokenizer_training_metadata.json
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import T5Tokenizer


PROJECT_ROOT = Path("/netscratch/anonymous_user/projects/BA-hydra")
SUBPROJECT_ROOT = PROJECT_ROOT / "cross_lingual_transfer_multilingual"
OUTPUT_ROOT = SUBPROJECT_ROOT / "models" / "tokenizers_paper_spm32k"
SUBSPLIT_NAME = "sub_charcap43gb_seed42"

LANGUAGES = [
    "deu_Latn",
    "jpn_Jpan",
    "spa_Latn",
    "rus_Cyrl",
    "pol_Latn",
    "por_Latn",
]

# Fixed tokenizer settings for the paper-track
SPM_VOCAB_SIZE = 32000
SPM_MODEL_TYPE = "bpe"
SPM_CHARACTER_COVERAGE = 1.0
SPM_BYTE_FALLBACK = True
SPM_HARD_VOCAB_LIMIT = False
SPM_INPUT_SENTENCE_SIZE = 0
SPM_SHUFFLE_INPUT_SENTENCE = True
SPM_NUM_THREADS = 32
TEXT_COLUMN = "text"
PARQUET_BATCH_SIZE = 1024
MAX_CHARS_PER_DOC = 20000
EXTRA_IDS = 100


@dataclass
class LanguageResult:
    language: str
    docs_parquet: str
    output_dir: str
    status: str
    duration_sec: float
    docs_rows: int
    tokenizer_length: int | None
    fast_tokenizer_export_ok: bool
    error: str | None = None


class ParquetTextIterator:
    """Streaming iterator over parquet text rows for SentencePiece training."""

    def __init__(
        self,
        parquet_path: Path,
        text_column: str = TEXT_COLUMN,
        batch_size: int = PARQUET_BATCH_SIZE,
        max_chars: int = MAX_CHARS_PER_DOC,
    ) -> None:
        self.parquet_path = parquet_path
        self.text_column = text_column
        self.batch_size = batch_size
        self.max_chars = max_chars

    def __iter__(self) -> Iterable[str]:
        pf = pq.ParquetFile(str(self.parquet_path))
        for batch in pf.iter_batches(
            batch_size=self.batch_size,
            columns=[self.text_column],
            use_threads=True,
        ):
            column = batch.column(0)
            for value in column:
                text = value.as_py()
                if not text:
                    continue
                text = text.replace("\n", " ").replace("\r", " ").strip()
                if not text:
                    continue
                if self.max_chars > 0 and len(text) > self.max_chars:
                    text = text[: self.max_chars]
                yield text


def _docs_parquet_path(language: str) -> Path:
    return (
        SUBPROJECT_ROOT
        / "data"
        / "languages"
        / language
        / "splits"
        / "sub"
        / SUBSPLIT_NAME
        / "train"
        / "docs.parquet"
    )


def _docs_rows(parquet_path: Path) -> int:
    return int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)


def _train_sentencepiece(docs_path: Path, out_dir: Path) -> Path:
    model_prefix = out_dir / "spm"
    iterator = iter(ParquetTextIterator(docs_path))

    spm.SentencePieceTrainer.train(
        sentence_iterator=iterator,
        model_prefix=str(model_prefix),
        model_type=SPM_MODEL_TYPE,
        vocab_size=SPM_VOCAB_SIZE,
        character_coverage=SPM_CHARACTER_COVERAGE,
        byte_fallback=SPM_BYTE_FALLBACK,
        hard_vocab_limit=SPM_HARD_VOCAB_LIMIT,
        input_sentence_size=SPM_INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SPM_SHUFFLE_INPUT_SENTENCE,
        num_threads=SPM_NUM_THREADS,
        bos_id=-1,
        eos_id=1,
        pad_id=0,
        unk_id=2,
    )

    return out_dir / "spm.model"


def _export_t5_tokenizer(spm_model_path: Path, out_dir: Path) -> tuple[int, bool]:
    # Save canonical slow T5 tokenizer artifacts only.
    # This keeps compatibility with the training environment's tokenizer stack.
    slow_tokenizer = T5Tokenizer(vocab_file=str(spm_model_path), extra_ids=EXTRA_IDS)
    slow_tokenizer.save_pretrained(str(out_dir))

    # Remove fast artifact if present to avoid cross-version tokenizer.json parsing issues.
    tok_json = out_dir / "tokenizer.json"
    if tok_json.exists():
        tok_json.unlink()

    spiece_model = out_dir / "spiece.model"
    if not spiece_model.exists():
        shutil.copy2(spm_model_path, spiece_model)

    return int(len(slow_tokenizer)), False


def _write_metadata(
    language: str,
    docs_path: Path,
    out_dir: Path,
    docs_rows: int,
    tokenizer_len: int,
    fast_ok: bool,
    duration_sec: float,
) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "track": "paper_spm32k",
        "training_data": str(docs_path),
        "training_data_rows": docs_rows,
        "subsplit": SUBSPLIT_NAME,
        "sentencepiece": {
            "model_type": SPM_MODEL_TYPE,
            "vocab_size": SPM_VOCAB_SIZE,
            "character_coverage": SPM_CHARACTER_COVERAGE,
            "byte_fallback": SPM_BYTE_FALLBACK,
            "hard_vocab_limit": SPM_HARD_VOCAB_LIMIT,
            "input_sentence_size": SPM_INPUT_SENTENCE_SIZE,
            "shuffle_input_sentence": SPM_SHUFFLE_INPUT_SENTENCE,
            "num_threads": SPM_NUM_THREADS,
            "pad_id": 0,
            "eos_id": 1,
            "unk_id": 2,
            "bos_id": -1,
        },
        "t5": {
            "extra_ids": EXTRA_IDS,
            "tokenizer_length": tokenizer_len,
            "fast_tokenizer_export_ok": fast_ok,
        },
        "duration_sec": duration_sec,
    }
    (out_dir / "tokenizer_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _train_one(language: str) -> LanguageResult:
    start = time.time()
    docs_path = _docs_parquet_path(language)
    out_dir = OUTPUT_ROOT / language

    if not docs_path.exists():
        return LanguageResult(
            language=language,
            docs_parquet=str(docs_path),
            output_dir=str(out_dir),
            status="error",
            duration_sec=0.0,
            docs_rows=0,
            tokenizer_length=None,
            fast_tokenizer_export_ok=False,
            error=f"Missing docs parquet: {docs_path}",
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_rows = _docs_rows(docs_path)

    try:
        spm_model = _train_sentencepiece(docs_path, out_dir)
        tokenizer_len, fast_ok = _export_t5_tokenizer(spm_model, out_dir)
        duration = time.time() - start

        _write_metadata(
            language=language,
            docs_path=docs_path,
            out_dir=out_dir,
            docs_rows=docs_rows,
            tokenizer_len=tokenizer_len,
            fast_ok=fast_ok,
            duration_sec=duration,
        )

        return LanguageResult(
            language=language,
            docs_parquet=str(docs_path),
            output_dir=str(out_dir),
            status="ok",
            duration_sec=duration,
            docs_rows=docs_rows,
            tokenizer_length=tokenizer_len,
            fast_tokenizer_export_ok=fast_ok,
        )
    except Exception as exc:  # pylint: disable=broad-except
        duration = time.time() - start
        return LanguageResult(
            language=language,
            docs_parquet=str(docs_path),
            output_dir=str(out_dir),
            status="error",
            duration_sec=duration,
            docs_rows=docs_rows,
            tokenizer_length=None,
            fast_tokenizer_export_ok=False,
            error=str(exc),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info("Starting paper tokenizers for languages: %s", ", ".join(LANGUAGES))

    results: list[LanguageResult] = []
    for language in LANGUAGES:
        logging.info("[%s] Training tokenizer...", language)
        res = _train_one(language)
        results.append(res)
        if res.status == "ok":
            logging.info(
                "[%s] done | rows=%d | tokenizer_len=%s | fast=%s | %.1fs",
                language,
                res.docs_rows,
                res.tokenizer_length,
                res.fast_tokenizer_export_ok,
                res.duration_sec,
            )
        else:
            logging.error("[%s] failed: %s", language, res.error)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(OUTPUT_ROOT),
        "subsplit": SUBSPLIT_NAME,
        "languages": LANGUAGES,
        "results": [asdict(r) for r in results],
    }
    summary_path = OUTPUT_ROOT / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary written to %s", summary_path)

    failures = [r for r in results if r.status != "ok"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
