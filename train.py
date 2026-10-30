"""Fine-tune a local, personal NLLB translation model on parallel Badini data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

BASE_MODEL = "arkanit/badini-nllb-translator"
LANGUAGES = {"en": "eng_Latn", "ar": "arb_Arab", "sorani": "ckb_Arab", "badini": "ckb_Arab"}
DATA = Path(__file__).resolve().parent / "training_data"


def read_pairs(extra=None):
    pairs = set()
    def add(language, source, target):
        source, target = (source or "").strip(), (target or "").strip()
        if language in ("en", "ar", "sorani") and source and target:
            pairs.add((language, source, target))
    files = [
        ("Dictionary_English-BehdiniLatin-BehdiniArabic-Arabic-Transliteration.csv", "Kurdish (Arabic Script)", True),
        ("Expressions_BehdiniArabic-BehdiniLatin-English.csv", "Kurdish (Arabic Script)", False),
        ("Proverbs_BehdiniArabic-BehdiniLatini-English.csv", "Kurdish Arabic", False),
        ("Verbs_Verb-Tense-Person-English-BehdiniLatin-BehdiniArabic.csv", "Kurdish (Arabic)", False),
    ]
    for filename, target, has_arabic in files:
        with (DATA / filename).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                add("en", row.get("English"), row.get(target))
                if has_arabic:
                    add("ar", row.get("Arabic"), row.get(target))
    # Published teacher outputs are examples, not live model inference.
    teacher_examples = DATA / "badini_model_card_examples.csv"
    with teacher_examples.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            add(row.get("source_lang"), row.get("source_text"), row.get("badini"))
    if extra:
        with Path(extra).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"source_lang", "source_text", "badini"}.issubset(reader.fieldnames):
                raise ValueError("Extra CSV must have source_lang,source_text,badini columns.")
            for line, row in enumerate(reader, 2):
                lang = (row.get("source_lang") or "").strip()
                if lang not in ("en", "ar", "sorani") or not (row.get("source_text") or "").strip() or not (row.get("badini") or "").strip():
                    raise ValueError(f"Invalid training row {line}.")
                add(lang, row["source_text"], row["badini"])
    return sorted(pairs)


def split_pairs(pairs, seed=42):
    # Keep all translations of the same source on the same side of the split.
    groups = {}
    for lang, source, target in pairs:
        groups.setdefault((lang, source.casefold()), []).append((lang, source, target))
    def is_heldout(key):
        lang, source = key
        digest = hashlib.sha256(f"{seed}\0{lang}\0{source}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") < (1 << 64) // 20
    train = [row for key in sorted(groups) if not is_heldout(key) for row in groups[key]]
    test = [row for key in sorted(groups) if is_heldout(key) for row in groups[key]]
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Optional UTF-8 CSV: source_lang,source_text,badini")
    parser.add_argument("--base", default=BASE_MODEL, help="Starting model ID or previously trained personal model directory")
    parser.add_argument("--output", default="models/my-badini-ai")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume", help="Resume from a Trainer checkpoint directory")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size < 1 or args.max_length < 2:
        parser.error("epochs, batch size and maximum length must be positive")

    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    set_seed(42)
    train, heldout = split_pairs(read_pairs(args.data))
    if not train:
        raise ValueError("No training pairs found.")
    print(f"Training: {len(train)} pairs; held-out: {len(heldout)} pairs.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            lang, source, target = self.rows[index]
            tokenizer.src_lang = LANGUAGES[lang]
            tokenizer.tgt_lang = LANGUAGES["badini"]
            inputs = tokenizer(source, truncation=True, max_length=args.max_length)
            inputs["labels"] = tokenizer(text_target=target, truncation=True, max_length=args.max_length)["input_ids"]
            return inputs

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "heldout.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_lang", "source_text", "badini"])
        writer.writerows(heldout)
    options = Seq2SeqTrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=4,
        learning_rate=2e-5, save_strategy="steps", save_steps=250, save_total_limit=1,
        logging_steps=50, report_to="none", dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(model=model, args=options, train_dataset=Pairs(train),
                            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    previous_metadata = Path(args.base) / "badini_model.json"
    previous_languages = set()
    if previous_metadata.is_file():
        previous_languages = set(json.loads(previous_metadata.read_text(encoding="utf-8"))["source_languages"])
    (output / "badini_model.json").write_text(json.dumps({"source_languages": sorted(previous_languages | {r[0] for r in train}), "base_model": args.base}, indent=2), encoding="utf-8")
    print(f"Your trained model is saved in {output}", flush=True)


if __name__ == "__main__":
    main()
