"""Compare exact-match results on the same source-disjoint dictionary words."""
import argparse
import gc
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import read_pairs, split_pairs
from tools.filter_dictionary_agreement import normalize


def heldout_words(rows, size, seed=93):
    grouped = {}
    for lang, source, target in rows:
        if lang == 'en' and re.fullmatch(r'[A-Za-z]+', source) and target.strip():
            key = source.casefold()
            grouped.setdefault(key, set()).add(normalize(target))
    words = sorted(grouped)
    random.Random(seed).shuffle(words)
    return [(word, grouped[word]) for word in words[:size]]


def evaluate(adapter, rows, batch_size=8):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(2)
    config = PeftConfig.from_pretrained(adapter)
    tokenizer = AutoTokenizer.from_pretrained(adapter, src_lang='eng_Latn')
    base = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    correct = 0
    for offset in range(0, len(rows), batch_size):
        part = rows[offset:offset+batch_size]
        encoded = tokenizer([x[0] for x in part], return_tensors='pt', padding=True, truncation=True, max_length=64)
        with torch.inference_mode():
            out = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids('ckb_Arab'), max_new_tokens=32)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        correct += sum(normalize(pred) in refs for (_, refs), pred in zip(part, preds))
    del model, base, tokenizer
    gc.collect()
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude-reviewed', help='Trainer backup containing private sources to exclude')
    args = parser.parse_args()
    if args.size < 1:
        parser.error('size must be positive')
    _, heldout = split_pairs(read_pairs())
    if args.exclude_reviewed:
        from train_adapter import read_reviewed_examples, remove_reviewed_sources
        heldout = remove_reviewed_sources(heldout, read_reviewed_examples(args.exclude_reviewed))
    rows = heldout_words(heldout, args.size)
    if not rows:
        raise ValueError('No held-out English single-word references')
    report = {'examples': len(rows), 'before_exact': evaluate(args.before, rows), 'after_exact': evaluate(args.after, rows),
              'limitation': 'Exact string matches are not a complete measure of quality; the pretrained base may have seen these references.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('::notice title=Held-out dictionary word exact matches::' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
