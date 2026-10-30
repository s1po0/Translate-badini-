# Approximate Sorani–Badini data

`ferman_approximate_sorani_badini.csv` was mined from the **train split** of
[Ferman NLU v2](https://github.com/omarGH99/ferman-ai-assistant), commit
`3071ef6a198c34353bf2a081de1461411b8f6c37`, by `tools/mine_ferman.py`.
Ferman is by Omar Safar and is licensed CC BY 4.0; its English and Arabic
examples derive from Amazon MASSIVE (FitzGerald et al., CC BY 4.0).
Source dataset card: https://github.com/omarGH99/ferman-ai-assistant/blob/3071ef6a198c34353bf2a081de1461411b8f6c37/data/DATASET_CARD.md

The source contains machine-translated Kurdish assistant commands, informally
reviewed by its author, not independently checked. It has no translation-pair
IDs. We matched rows automatically by intent and character overlap; pairs may
be unrelated, mistranslated, or not genuinely Badini. These **379 approximate,
unverified pairs** are unsuitable as a gold-standard test set, cover only
assistant-command topics, and require native-speaker review. The included
Sorani model is experimental and should not be relied upon for correctness.

Reproduce the selection using the Ferman v2 CSV:

```
python tools/mine_ferman.py XLMR_dataset_4lang_v2.csv training_data/ferman_approximate_sorani_badini.csv
```
