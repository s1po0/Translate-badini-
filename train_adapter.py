"""CPU-friendly pilot fine-tune of an existing Badini translation model.

Creates a small LoRA adapter and measures held-out teacher-forced loss. This is
an experimental pilot, not a guarantee of better or fluent translation.
"""
import argparse
import json
import random
from pathlib import Path

from train import BASE_MODEL, LANGUAGES, read_pairs, split_pairs


def select_examples(rows, per_language, seed):
    """Deterministically balance source languages without duplicating rows."""
    rng = random.Random(seed)
    selected = []
    for language in ("en", "ar", "sorani"):
        candidates = [row for row in rows if row[0] == language]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_language])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training_data/ferman_approximate_sorani_badini.csv")
    parser.add_argument("--output", default="models/badini-adapter")
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--initial-adapter", help="Previously trained LoRA adapter directory")
    parser.add_argument("--synthetic-data", help="Teacher-generated source_lang,source_text,badini CSV")
    parser.add_argument("--synthetic-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--eval-per-language", type=int, default=8)
    args = parser.parse_args()
    if min(args.steps, args.per_language, args.eval_per_language, args.synthetic_repeats) <= 0:
        parser.error("steps and sample sizes must be positive")

    import torch
    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    torch.set_num_threads(2)
    train_rows, test_rows = split_pairs(read_pairs(args.data))
    train_rows = select_examples(train_rows, args.per_language, args.seed)
    test_rows = select_examples(test_rows, args.eval_per_language, 43)
    synthetic_count = 0
    if args.synthetic_data:
        import csv
        with Path(args.synthetic_data).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"source_lang", "source_text", "badini"}.issubset(reader.fieldnames):
                raise ValueError("Synthetic CSV must have source_lang,source_text,badini columns.")
            synthetic_rows = []
            for row in reader:
                lang = (row.get("source_lang") or "").strip()
                source = (row.get("source_text") or "").strip()
                target = (row.get("badini") or "").strip()
                if lang not in ("en", "ar", "sorani") or not source or not target:
                    raise ValueError("Invalid synthetic example.")
                synthetic_rows.append((lang, source, target))
        synthetic_rows, _ = split_pairs(synthetic_rows)
        if not synthetic_rows:
            raise ValueError("No synthetic examples remain after holding out evaluation sources.")
        synthetic_count = len(synthetic_rows)
        train_rows += synthetic_rows * args.synthetic_repeats
        random.Random(args.seed).shuffle(train_rows)
    if not train_rows or not test_rows:
        raise ValueError("Training and held-out examples are required.")

    print(f"Downloading/loading {args.base}; {len(train_rows)} train, {len(test_rows)} held-out examples", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base, low_cpu_mem_usage=True)
    model.config.use_cache = False
    if args.initial_adapter:
        adapter_config = PeftConfig.from_pretrained(args.initial_adapter)
        if adapter_config.base_model_name_or_path != args.base:
            raise ValueError("Adapter base model does not match --base.")
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        ))
    model.print_trainable_parameters()

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            language, source, target = self.rows[index]
            tokenizer.src_lang = LANGUAGES[language]
            tokenizer.tgt_lang = LANGUAGES["badini"]
            features = tokenizer(source, truncation=True, max_length=64)
            features["labels"] = tokenizer(text_target=target, truncation=True, max_length=64)["input_ids"]
            return features

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    heldout = Pairs(test_rows)

    def measure_loss():
        model.eval()
        losses = {"en": [], "ar": [], "sorani": []}
        with torch.inference_mode():
            for (language, _, _), batch in zip(test_rows, DataLoader(heldout, batch_size=1, collate_fn=collator)):
                losses[language].append(float(model(**batch).loss))
        all_losses = [value for values in losses.values() for value in values]
        return {"overall": sum(all_losses) / len(all_losses),
                "by_language": {lang: sum(values) / len(values) if values else None for lang, values in losses.items()}}

    before = measure_loss()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    options = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=1e-4, logging_steps=10, save_strategy="no", report_to="none",
        dataloader_num_workers=0, use_cpu=True, seed=args.seed,
    )
    trainer = Seq2SeqTrainer(model=model, args=options, train_dataset=Pairs(train_rows), data_collator=collator)
    trainer.train()
    after = measure_loss()
    model.save_pretrained(str(out / "adapter"), safe_serialization=True)
    tokenizer.save_pretrained(str(out / "adapter"))
    report = {
        "base_model": args.base, "optimizer_steps": args.steps, "training_pool_pairs": len(train_rows),
        "initial_adapter": args.initial_adapter, "synthetic_unique_rows": synthetic_count, "synthetic_repeats": args.synthetic_repeats if args.synthetic_data else 0,
        "training_by_language": {lang: sum(row[0] == lang for row in train_rows) for lang in ("en", "ar", "sorani")},
        "heldout_pairs": len(test_rows), "teacher_forced_loss_before": before,
        "teacher_forced_loss_after": after,
        "limitations": "Small unverified pilot; Sorani pairs are automatically matched and machine-translated. Loss is not translation accuracy. The base model may have seen evaluation data.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
