"""Train and evaluate the small offline statistical Badini model."""
import argparse
import gzip
import json
from pathlib import Path

from statistical import MODEL_PATH, StatisticalTranslator, tokens, train_ibm1
from train import read_pairs, split_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Optional reviewed CSV with source_lang,source_text,badini")
    args = parser.parse_args()
    training, heldout = split_pairs(read_pairs(args.data))
    supported = sorted({lang for lang, _, _ in training})
    languages = {}
    for lang in supported:
        pairs = [(source, target) for language, source, target in training if language == lang]
        print(f'Training {lang} on {len(pairs)} examples...', flush=True)
        languages[lang] = train_ibm1(pairs)
    model = {"format": "IBM Model 1 + source phrase table", "iterations": 5, "languages": languages,
             "sources": ["Jad-deeb/Kurdish-behdini-language-resources, MIT", "arkanit/badini-nllb-translator model card examples; see training_data/BADINI_TEACHER_ATTRIBUTION.md"] + ([str(args.data)] if args.data else []), "seed": 42}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(MODEL_PATH, 'wt', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, separators=(',', ':'))
    translator = StatisticalTranslator()
    stats = {}
    for lang in supported:
        examples = [(s, t) for l, s, t in heldout if l == lang]
        exact = sum(tokens(translator.translate(s, lang, 'badini')) == tokens(t) for s, t in examples)
        stats[lang] = {'heldout_examples': len(examples), 'exact_token_match': exact,
                       'exact_match_rate': round(exact / len(examples), 4) if examples else None}
    Path('trained/evaluation.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print('Held-out exact token matches:', stats, flush=True)


if __name__ == '__main__':
    main()
