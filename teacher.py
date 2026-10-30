"""Generate reviewable translation examples with a local Ollama teacher.

A trained student may also attempt each example before the teacher suggests a
correction. Generated examples are never automatically approved or trained on.
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

LANGUAGES = {"en": "English", "ar": "Arabic", "sorani": "Sorani Kurdish"}
FIELDS = ("source_lang", "source_text", "badini", "student_translation", "approved")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def ask_teacher(model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "options": {"temperature": 0.8}}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        answer = json.loads(result["response"])
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Could not use the local Ollama teacher: {exc}") from exc
    if not isinstance(answer, dict):
        raise ValueError("Teacher did not return a JSON object.")
    return answer


def clean_text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"Teacher returned invalid {field}.")
    return value.strip()


def example(model, lang, student=None):
    result = ask_teacher(model, f"Write one diverse, natural, short {LANGUAGES[lang]} sentence and translate it into Badini Kurdish using Arabic script (not Sorani). Respond only with a JSON object with string keys source_text and badini. Do not add explanations. If unsure, still provide your best attempt.")
    source = clean_text(result.get("source_text"), "source_text")
    reference = clean_text(result.get("badini"), "badini")
    student_text = ""
    if student and lang in student.own_languages:
        student_text = student.translate(source, lang, "badini")
        correction = ask_teacher(model, f"Translate this {LANGUAGES[lang]} text into Badini Kurdish in Arabic script, not Sorani. The student's attempt may be wrong; independently correct it. Return only a JSON object with a string key badini.\nSource: {json.dumps(source, ensure_ascii=False)}\nStudent attempt: {json.dumps(student_text, ensure_ascii=False)}")
        reference = clean_text(correction.get("badini"), "badini")
    return {"source_lang": lang, "source_text": source, "badini": reference,
            "student_translation": student_text, "approved": ""}


def generate(args):
    student = None
    if args.student_offline:
        import gzip
        from statistical import MODEL_PATH, StatisticalTranslator
        with gzip.open(MODEL_PATH, "rt", encoding="utf-8") as f:
            languages = set(json.load(f)["languages"])
        student = StatisticalTranslator()
        student.own_languages = languages
    elif args.student_model:
        # Import only when a student model is requested; teacher-only generation
        # works without PyTorch or Transformers.
        import app
        metadata = Path(args.student_model) / "badini_model.json"
        if not metadata.is_file():
            raise ValueError("Student model is not trained or has no badini_model.json.")
        student = app.Translator()
        student.own_model = str(Path(args.student_model).resolve())
        student.own_languages = set(json.loads(metadata.read_text(encoding="utf-8"))["source_languages"])
    output = Path(args.output)
    if output.exists() and output.stat().st_size:
        with output.open(encoding="utf-8-sig", newline="") as f:
            if tuple(next(csv.reader(f), [])) != FIELDS:
                raise ValueError("Existing output CSV has a different header; choose another file.")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
            f.flush()
        index = 0
        while args.count == 0 or index < args.count:
            lang = args.languages[index % len(args.languages)]
            row = example(args.model, lang, student)
            writer.writerow(row)
            f.flush()
            index += 1
            print(f"{index}: {lang} | {row['source_text']} → {row['badini']} (needs review)", flush=True)


def export_approved(input_path, output_path):
    src, dst = Path(input_path).resolve(), Path(output_path).resolve()
    if src == dst:
        raise ValueError("Input and output must be different files.")
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("Review CSV has an invalid header.")
        rows = []
        for line, row in enumerate(reader, 2):
            if (row.get("approved") or "").strip().lower() not in ("yes", "1", "true"):
                continue
            lang = (row.get("source_lang") or "").strip()
            if lang not in LANGUAGES:
                raise ValueError(f"Invalid language in approved row {line}.")
            rows.append({"source_lang": lang, "source_text": clean_text(row.get("source_text"), "source_text"),
                         "badini": clean_text(row.get("badini"), "badini")})
    if not rows:
        raise ValueError("No approved examples. Review them before exporting.")
    if dst.exists():
        raise ValueError("Export file already exists; choose a new output path.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("source_lang", "source_text", "badini"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} approved examples to {dst}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate unapproved examples with a local teacher")
    gen.add_argument("--model", default="qwen2.5:3b", help="Locally downloaded Ollama model name")
    students = gen.add_mutually_exclusive_group()
    students.add_argument("--student-model", help="Directory of an already trained personal neural student model")
    students.add_argument("--student-offline", action="store_true", help="Use the included small statistical student")
    gen.add_argument("--languages", nargs="+", choices=tuple(LANGUAGES), default=list(LANGUAGES))
    gen.add_argument("--count", type=int, default=10, help="Examples to generate; 0 runs until Ctrl+C")
    gen.add_argument("--output", default="data/teacher-review.csv")
    export = sub.add_parser("export", help="Export only human-approved examples for training")
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            if args.count < 0:
                parser.error("--count must be 0 or greater")
            if args.model.endswith(":cloud"):
                raise ValueError("A local, downloaded model is required; cloud models are not offline.")
            generate(args)
        else:
            export_approved(args.input, args.output)
    except KeyboardInterrupt:
        print("\nStopped. Completed examples remain in the review CSV.")
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
