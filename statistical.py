"""A small, offline statistical translation model, not a neural chatbot."""
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "trained" / "badini-statistical.json.gz"
TOKEN = re.compile(r"[\w\u064b-\u065f\u0670\u06d6-\u06ed\u200c]+|[^\w\s]", re.UNICODE)


def tokens(text):
    return TOKEN.findall(text.casefold())


def train_ibm1(pairs, iterations=5):
    """Estimate p(target token | source token) by IBM Model 1 EM."""
    sentences = []
    options = defaultdict(set)
    phrase_counts = defaultdict(Counter)
    for source, target in pairs:
        src, tgt = tokens(source), tokens(target)
        if not src or not tgt or len(src) > 40 or len(tgt) > 40:
            continue
        sentences.append((src, tgt))
        phrase_counts[' '.join(src)][target.strip()] += 1
        for word in set(src):
            options[word].update(tgt)
    probabilities = {s: {t: 1 / len(targets) for t in targets} for s, targets in options.items()}
    for iteration in range(iterations):
        counts = defaultdict(lambda: defaultdict(float))
        totals = defaultdict(float)
        for src, tgt in sentences:
            for target in tgt:
                denom = sum(probabilities[s].get(target, 0.0) for s in src)
                if not denom:
                    continue
                for source in src:
                    count = probabilities[source].get(target, 0.0) / denom
                    counts[source][target] += count
                    totals[source] += count
        probabilities = {s: {t: count / totals[s] for t, count in vals.items()}
                         for s, vals in counts.items() if totals[s]}
        print(f"IBM1 iteration {iteration + 1}/{iterations}: {len(sentences)} examples", flush=True)
    lexicon = {s: max(vals, key=vals.get) for s, vals in probabilities.items() if vals}
    phrases = {key: choices.most_common(1)[0][0] for key, choices in phrase_counts.items()}
    return {"phrases": phrases, "lexicon": lexicon, "training_examples": len(sentences)}


class StatisticalTranslator:
    def __init__(self, model_path=MODEL_PATH):
        self.path = Path(model_path)
        self.data = None

    def translate(self, text, source, target):
        if target != "badini":
            raise ValueError("ئەم مۆدێلە ئۆفلاینە تەنها بۆ بادینی وەردەگێڕێت.")
        if self.data is None:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                self.data = json.load(f)
        if source not in self.data["languages"]:
            raise ValueError("ئەم مۆدێلە بۆ ئەم زمانە ڕاهێنراو نییە.")
        entry = self.data["languages"][source]
        ts = tokens(text)
        full_key = ' '.join(ts)
        if full_key in entry["phrases"]:
            return entry["phrases"][full_key]
        out, unknown = [], []
        pos = 0
        while pos < len(ts):
            match = None
            # Whole-sentence matches were checked above; use short phrases here.
            for end in range(min(pos + 5, len(ts)), pos, -1):
                key = ' '.join(ts[pos:end])
                if key in entry["phrases"]:
                    match = (entry["phrases"][key], end)
                    break
            if match:
                out.append(match[0])
                pos = match[1]
            elif ts[pos] in entry['lexicon']:
                out.append(entry['lexicon'][ts[pos]])
                pos += 1
            else:
                token = ts[pos]
                out.append(token)
                if any(ch.isalnum() for ch in token):
                    unknown.append(token)
                pos += 1
        result = ' '.join(out)
        if unknown:
            result += '\n\n[وەرنەگێڕدراو: ' + ', '.join(dict.fromkeys(unknown)) + ']'
        return result
