// Local personal memory is separate from the bundled IBM Model 1 model.
export const tokenize = text => (text.toLocaleLowerCase('und').match(/[\p{L}\p{N}\p{M}_\u200c]+|[^\p{L}\p{N}\p{M}_\u200c\s]/gu) || []);
export const key = text => tokenize(text.trim()).join(' ');
export function translate(model, corrections, source, text) {
  const entry = model?.languages?.[source];
  if (!entry) throw new Error('مۆدێلی ئەم زمانە ئامادە نییە.');
  const ts = tokenize(text), personal = new Map();
  for (const pair of corrections) if (pair.language === source) personal.set(key(pair.source), pair.target);
  const full = key(text);
  if (personal.has(full)) return {text: personal.get(full), learned: true, unknown: []};
  if (Object.hasOwn(entry.phrases, full)) return {text: entry.phrases[full], learned: false, unknown: []};
  const out = [], unknown = []; let learned = false;
  for (let i = 0; i < ts.length;) {
    let found = false;
    // A personal phrase overrides the bundled model for the same span.
    for (let end = Math.min(ts.length, i + 5); end > i; end--) {
      const k = ts.slice(i, end).join(' ');
      if (personal.has(k) || Object.hasOwn(entry.phrases, k)) {
        learned ||= personal.has(k); out.push(personal.get(k) ?? entry.phrases[k]); i = end; found = true; break;
      }
    }
    if (found) continue;
    if (Object.hasOwn(entry.lexicon, ts[i])) out.push(entry.lexicon[ts[i]]);
    else { out.push(ts[i]); if (/[\p{L}\p{N}]/u.test(ts[i])) unknown.push(ts[i]); }
    i++;
  }
  return {text: out.join(' '), learned, unknown: [...new Set(unknown)]};
}
export function upsert(pairs, pair) {
  if (!['en','ar','sorani'].includes(pair.language) || typeof pair.source !== 'string' || typeof pair.target !== 'string') throw new Error('نموونە نادروستە.');
  const source = pair.source.trim(), target = pair.target.trim();
  if (!key(source) || !target || source.length > 1000 || target.length > 1000) throw new Error('هەردوو خانە پڕ بکەوە (زۆرترین ١٠٠٠ پیت).');
  const index = pairs.findIndex(p => p.language === pair.language && key(p.source) === key(source));
  const result = [...pairs]; const item = {language: pair.language, source, target, updated: new Date().toISOString()};
  if (index < 0) result.unshift(item); else result.splice(index, 1, item);
  return result;
}
