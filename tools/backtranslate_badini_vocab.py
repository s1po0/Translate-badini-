"""Generate unverified English labels for observed Badini words with NLLB."""
import argparse
import csv
import gzip
from pathlib import Path
import re


def choose(path, count):
    if count < 1:
        raise ValueError('count must be positive')
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    return rows[:count]


def write(rows, batches, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    all_count = valid_count = 0
    with (root/'backtranslation-attempts.csv').open('x', encoding='utf-8', newline='') as a, (root/'synthetic.csv').open('x', encoding='utf-8', newline='') as s:
        wa, ws = csv.writer(a), csv.writer(s)
        wa.writerow(['id', 'badini', 'frequency', 'english', 'passes_filter'])
        ws.writerow(['id', 'source_lang', 'source_text', 'badini'])
        for group in batches(rows):
            for row, english in group:
                english = english.strip()
                # A simple script check, not a guarantee of meaning or dialect.
                valid = bool(english and len(english) <= 160 and re.search('[A-Za-z]', english) and not re.search(r'[\u0600-\u06ff]', english))
                wa.writerow([row['id'], row['badini_word'], row['frequency'], english, int(valid)])
                all_count += 1
                if valid:
                    ws.writerow([row['id'], 'en', english, row['badini_word']])
                    valid_count += 1
            a.flush(); s.flush()
    if all_count != len(rows) or not valid_count:
        raise ValueError('Incomplete backtranslation or no usable English outputs')
    print(f'::notice title=Backtranslation attempts::{all_count} Badini words; {valid_count} English-script candidates, unverified', flush=True)
    return all_count, valid_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--teacher', default='arkanit/badini-nllb-translator')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    rows = choose(args.vocabulary, args.count)
    if not rows:
        raise ValueError('No vocabulary')
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, src_lang='ckb_Arab')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher, low_cpu_mem_usage=True)
    model.eval()

    def batches(items):
        for pos in range(0, len(items), args.batch_size):
            group = items[pos:pos+args.batch_size]
            encoded = tokenizer([r['badini_word'] for r in group], return_tensors='pt', padding=True, truncation=True, max_length=64)
            with torch.inference_mode():
                out = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids('eng_Latn'), max_new_tokens=32)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            if len(decoded) != len(group):
                raise ValueError('Wrong output size')
            yield list(zip(group, decoded))
            print(f'Backtranslated {pos+len(group)}/{len(items)}', flush=True)
    write(rows, batches, args.output)


if __name__ == '__main__':
    main()
