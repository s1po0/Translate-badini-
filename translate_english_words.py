"""Translate numbered English words with a local Badini teacher.

Outputs every attempt, including failed/unverified results, plus a CSV containing
script-filtered candidates suitable for train_adapter.py. IDs refer to the
immutable index; script validation does not verify language or meaning.
"""
import argparse
import csv
from pathlib import Path

from synthetic_batch import valid_badini

INDEX = Path(__file__).resolve().parent / 'training_data/kagi/english-word-index.tsv'


def read_words(path, start, count):
    if start < 1 or count < 1:
        raise ValueError('start and count must be positive')
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            identifier, word = line.rstrip('\n').split('\t', 1)
            number = int(identifier)
            if start <= number < start + count:
                rows.append((identifier, word))
            if number >= start + count:
                break
    if not rows or int(rows[0][0]) != start:
        raise ValueError('Start ID not found')
    return rows


def write_results(rows, translate, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    attempts_path = output / 'attempts.csv'
    candidates_path = output / 'synthetic.csv'
    with attempts_path.open('x', encoding='utf-8', newline='') as attempts, candidates_path.open('x', encoding='utf-8', newline='') as candidates:
        all_writer = csv.writer(attempts)
        good_writer = csv.writer(candidates)
        all_writer.writerow(['id', 'english', 'badini', 'script_valid'])
        good_writer.writerow(['id', 'source_lang', 'source_text', 'badini'])
        good = 0
        for batch in translate(rows):
            for (identifier, word), result in batch:
                result = result.strip()
                valid = valid_badini(result)
                all_writer.writerow([identifier, word, result, int(valid)])
                if valid:
                    good_writer.writerow([identifier, 'en', word, result])
                    good += 1
            attempts.flush()
            candidates.flush()
        print(f'Attempted {len(rows)} words; {good} passed the script-only filter.', flush=True)
    if not good:
        raise ValueError('No generated outputs passed the script filter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default=str(INDEX))
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', required=True)
    parser.add_argument('--teacher', default='arkanit/badini-nllb-translator')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    rows = read_words(args.index, args.start, args.count)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, src_lang='eng_Latn')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher, low_cpu_mem_usage=True)
    model.eval()

    def translate(items):
        for position in range(0, len(items), args.batch_size):
            group = items[position:position + args.batch_size]
            encoded = tokenizer([word for _, word in group], return_tensors='pt', padding=True, truncation=True, max_length=64)
            with torch.inference_mode():
                tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids('ckb_Arab'), max_new_tokens=32)
            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            if len(decoded) != len(group):
                raise ValueError('Teacher returned the wrong number of translations')
            yield list(zip(group, decoded))
            print(f'Translated {min(position + len(group), len(items))}/{len(items)}', flush=True)
    write_results(rows, translate, args.output)


if __name__ == '__main__':
    main()
