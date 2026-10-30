"""Extract an observed Badini-script vocabulary from Dasinya segmented text.

This is a corpus frequency list, not a dictionary or a complete list of Badini.
It preserves ZWNJ and keeps the corpus publisher's NFC spellings; numbers,
Latin-only words, markup and punctuation are not considered words.
"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
import unicodedata
import zipfile

RUN = re.compile(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\u200c]+')


def tokens(text):
    text = unicodedata.normalize('NFC', text)
    for match in RUN.finditer(text):
        # Arabic block also contains digits/punctuation. Split on them rather than
        # accidentally combining neighbouring words; retain ZWNJ inside words.
        buf = []
        for char in match.group():
            category = unicodedata.category(char)
            if category.startswith(('L', 'M')) or char == '\u200c':
                buf.append(char)
            else:
                word = ''.join(buf).strip('\u200c')
                if word and any(unicodedata.category(c).startswith('L') for c in word):
                    yield word
                buf = []
        word = ''.join(buf).strip('\u200c')
        if word and any(unicodedata.category(c).startswith('L') for c in word):
            yield word


def extract(archive, output):
    frequency = Counter()
    files = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            path = name.replace('\\', '/')
            if not (path.startswith('corpus/3_segmented/') and path.endswith('_stage6_structured.txt')):
                continue
            files.append(name)
            text = z.read(name).decode('utf-8-sig')
            # Stage 6 uses <S> sentence and <T> title markers. Do not count
            # structure markers or other metadata as content.
            for line in text.splitlines():
                if line.startswith(('<S>', '<T>')):
                    frequency.update(tokens(re.sub(r'<[^>]*>', ' ', line)))
    if not files or not frequency:
        raise ValueError('No usable Dasinya segmented text')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    words = sorted(frequency.items(), key=lambda x: (-x[1], x[0]))
    with gzip.open(root / 'badini-vocabulary.tsv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'badini_word', 'frequency'])
        for i, (word, count) in enumerate(words, 1):
            writer.writerow([f'{i:07}', word, count])
    report = {'segmented_files': len(files), 'word_tokens': sum(frequency.values()),
              'distinct_observed_forms': len(words), 'source': 'https://doi.org/10.5281/zenodo.20729060',
              'source_license': 'CC BY 4.0',
              'limitation': 'Observed corpus forms may include OCR errors, names, non-Badini words and variants; not every Badini word.'}
    (root / 'vocabulary-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print('::notice title=Badini corpus vocabulary::' + json.dumps(extract(args.archive, args.output), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
