"""Use local Badini and multilingual NLLB teachers to propose training pairs."""
import argparse
import csv
import gc
from pathlib import Path

from distill_badini import BADINI_MODEL, PIVOT_MODEL


def valid_badini(text):
    # Script checks reject obvious failures but cannot verify dialect or meaning.
    return bool(text and len(text) <= 1500 and any('\u0600' <= c <= '\u06ff' for c in text))


class Runner:
    def __init__(self):
        from app import Translator
        self.model = Translator.__new__(Translator)
        self.model.models = {}

    def translate(self, model, text, source, target):
        return self.model._run(model, text, source, target)

    def unload(self):
        self.model.models.clear()
        gc.collect()


def make_batch(sources, runner, teacher=BADINI_MODEL, pivot=PIVOT_MODEL):
    intermediate = []
    for language, source in sources:
        english = source if language == 'en' else runner.translate(pivot, source, language, 'en').strip()
        if not english:
            continue
        intermediate.append((language, source, english))
    runner.unload()
    result = []
    for language, source, english in intermediate:
        badini = runner.translate(teacher, english, 'en', 'badini').strip()
        if valid_badini(badini):
            result.append((language, source, badini))
    runner.unload()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='CSV with source_lang and source_text')
    parser.add_argument('--output', required=True, help='New CSV for teacher-generated pairs')
    parser.add_argument('--teacher', default=BADINI_MODEL)
    parser.add_argument('--pivot', default=PIVOT_MODEL)
    args = parser.parse_args()
    with Path(args.input).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {'source_lang', 'source_text'}.issubset(reader.fieldnames):
            raise ValueError('Input requires source_lang and source_text columns')
        sources = []
        for row in reader:
            lang, text = row['source_lang'].strip(), row['source_text'].strip()
            if lang not in ('en', 'ar', 'sorani') or not text or len(text) > 2000:
                raise ValueError('Invalid source example')
            sources.append((lang, text))
    if not sources:
        raise ValueError('Input has no examples')
    result = make_batch(sources, Runner(), args.teacher, args.pivot)
    if not result:
        raise ValueError('Teachers generated no plausible Arabic-script output')
    with Path(args.output).open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_lang', 'source_text', 'badini'])
        writer.writerows(result)
    print(f'Teachers generated {len(result)} unverified pairs from {len(sources)} sources.', flush=True)


if __name__ == '__main__':
    main()
