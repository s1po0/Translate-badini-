"""Generate reviewable training examples with a Badini translation teacher."""
import argparse
import csv
import os
import sys
from pathlib import Path

from teacher import clean_text

BADINI_MODEL = "arkanit/badini-nllb-translator"
PIVOT_MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"en", "ar", "sorani"}
FIELDS = ("source_lang", "source_text", "badini", "student_translation", "approved", "teacher_model", "pivot_model")


def translate_example(model_runner, source, lang, teacher_model, pivot_model):
    """The Badini teacher is English→Badini; Arabic/Sorani go via English."""
    if lang not in LANGS:
        raise ValueError("Unsupported source language.")
    english = source
    if lang != "en":
        english = clean_text(model_runner(pivot_model, source, lang, "en"), "English pivot")
    return clean_text(model_runner(teacher_model, english, "en", "badini"), "Badini translation")


def generate(input_file, output_file, teacher_model=BADINI_MODEL, pivot_model=PIVOT_MODEL, runner=None):
    src, dst = Path(input_file).resolve(), Path(output_file).resolve()
    if src == dst or dst.exists():
        raise ValueError("Output must be a new file, different from the input.")
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"source_lang", "source_text"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV needs source_lang and source_text columns.")
        rows = []
        for line, row in enumerate(reader, 2):
            lang = (row.get("source_lang") or "").strip()
            if lang not in LANGS:
                raise ValueError(f"Invalid source_lang on line {line}.")
            rows.append((lang, clean_text(row.get("source_text"), f"source_text on line {line}")))
    if not rows:
        raise ValueError("The input CSV is empty.")
    if runner is None:
        # Avoid accidentally loading the app's optional BADINI_MODEL_DIR.
        from app import Translator
        model = Translator.__new__(Translator)
        model.models = {}
        runner = model._run
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        for lang, source in rows:
            result = translate_example(runner, source, lang, teacher_model, pivot_model)
            writer.writerow({"source_lang": lang, "source_text": source, "badini": result,
                             "student_translation": "", "approved": "", "teacher_model": teacher_model,
                             "pivot_model": pivot_model if lang != "en" else ""})
            f.flush()
    print(f"Generated {len(rows)} unreviewed examples: {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with source_lang and source_text")
    parser.add_argument("--output", required=True, help="New review CSV")
    parser.add_argument("--teacher-model", default=BADINI_MODEL, help="Hugging Face ID or local path")
    parser.add_argument("--pivot-model", default=PIVOT_MODEL, help="Hugging Face ID or local path")
    parser.add_argument("--allow-download", action="store_true", help="Allow missing model weights to download")
    args = parser.parse_args()
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        generate(args.input, args.output, args.teacher_model, args.pivot_model)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
