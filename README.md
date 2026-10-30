# وەرگێڕا بادینی / Badini Translator

## Ready-to-use offline model (experimental)

The default app uses the trained model in `trained/badini-statistical.json.gz`; it runs locally with Python's standard library, no internet, paid API or model downloads. It is a small statistical translation model (IBM Model 1 token alignments plus a phrase table), **not a chatbot or a neural model**. It was trained locally on the included MIT-licensed dictionary data, two English–Badini examples published by the Badini teacher model (see `training_data/BADINI_TEACHER_ATTRIBUTION.md`), and 379 automatically mined, **unverified** Sorani–Badini pairs from the CC BY 4.0 Ferman dataset (see `training_data/FERMAN_ATTRIBUTION.md`). It supports English, Arabic and experimental Sorani → Badini. Unknown words are shown as untranslated; reverse translation is unavailable.

Its quality is very limited: on a source-disjoint approximately 5% held-out split it exactly matched 12/1021 English, 2/573 Arabic and 1/20 Sorani references (see `trained/evaluation.json`). Many examples are dictionary entries, and the Sorani reference pairs are automatically matched and unverified; these are not overall accuracy estimates. Do not rely on the output without checking it. Reproduce the current model with `python3 train_statistical.py --data training_data/ferman_approximate_sorani_badini.csv`. Alternatively pass a reviewed CSV with `--data`; that replaces the Ferman inputs unless the CSV also includes them.

```sh
python3 app.py
```

Open http://127.0.0.1:8000. For the old third-party neural pipeline, set `TRANSLATOR_MODE=neural` and install requirements. Setting `BADINI_MODEL_DIR` selects the personally fine-tuned neural model instead.

## Optional neural pipeline

The optional neural pipeline uses existing AI models, not Kagi or a paid API. Supported directions:

- English → Badini, and Badini → English: [Badini NLLB Translator](https://huggingface.co/arkanit/badini-nllb-translator).
- Sorani → Badini and Arabic → Badini: [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) first translates into English; the Badini model then translates that English into Badini.

Select the source language manually. These are third-party models, not a newly trained model. The two-step translations can lose meaning, and none of the models is guaranteed to translate every word correctly. Their quality has not been independently verified here. The Badini model card describes Arabic-script Badini. Use short passages; inputs exceeding 512 tokens are rejected.

## Run

Python 3.10 or newer and internet access for the first model downloads are required. The Badini model is roughly 1.3 GB; Sorani and Arabic additionally require the NLLB model. Inference needs sufficient memory and CPU resources. The models are cached locally for subsequent use.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
TRANSLATOR_MODE=neural python app.py
```

Open http://127.0.0.1:8000. The first translation for each model downloads and loads it, which may take a while. Subsequent translations can work offline once the needed model files are cached. To change the listening address or port, set `HOST` or `PORT`; the default only listens locally. Do not expose this unauthenticated server to the public internet.

The Badini model author labels it MIT, but it is derived from NLLB-200. The NLLB base model is licensed CC-BY-NC-4.0. Check both models' licenses and provenance before commercial use or redistribution.

## Tests

```sh
python -m unittest discover -s tests
```

The tests mock inference and do not verify translation quality or download the models.

## Train your own model

`train.py` fine-tunes the existing Badini NLLB teacher model (`arkanit/badini-nllb-translator`) into a separate personal model; it does not train a model from scratch. The teacher is an English–Badini model, so Arabic and Sorani support after fine-tuning requires substantial verified parallel examples and evaluation. The bundled English/Arabic–Badini word and phrase pairs are from [Jad-deeb/Kurdish-behdini-language-resources](https://github.com/Jad-deeb/Kurdish-behdini-language-resources), MIT licensed (see `training_data/LICENSE`). They are a limited dictionary, not a comprehensive translation corpus. The base model is CC-BY-NC-4.0. Accuracy is not guaranteed.

Training requires an initial download of the base model, substantial disk/RAM and preferably an NVIDIA GPU. A CPU-only run can take a very long time. Install the same requirements above, then:

```sh
python train.py --output models/my-badini-ai
BADINI_MODEL_DIR=models/my-badini-ai python app.py
```

This produces an English/Arabic → Badini model. To train Sorani too, supply a UTF-8 CSV of verified parallel examples with the columns `source_lang,source_text,badini`, where `source_lang` is `en`, `ar`, or `sorani`. Example (illustrative only; verify translations before training):

```csv
source_lang,source_text,badini
sorani,سڵاو,سلاڤ
```

```sh
python train.py --data my-pairs.csv --output models/my-badini-ai
BADINI_MODEL_DIR=models/my-badini-ai python app.py
```

Include many reviewed Sorani–Badini pairs; a single example is not enough to learn a language. The personal model will reject directions it was not trained on, including Badini → English. Its metadata lists trained source languages. A `heldout.csv` is saved for independent evaluation; it is not a quality score. Use `TRANSLATOR_MODE=neural` without `BADINI_MODEL_DIR` for the earlier third-party model pipeline. Training outputs and downloaded model weights are excluded from Git.

### Google Colab

Open `Badini_AI_Training.ipynb` in Google Colab, select a GPU runtime, and run its cells in order. The notebook contains the training script and downloads the MIT-licensed dictionary data from a pinned GitHub revision, so it does not depend on pushing this branch. It mounts Google Drive and saves the model and periodic checkpoints there. Colab GPU availability and runtime limits are not guaranteed; costs may apply. The optional CSV is required for Sorani examples. Interrupted runs can be resumed from a saved checkpoint using the notebook's `RESUME` setting without changing data or training options. The notebook has not been run in Colab in this environment.

## Offline teacher for collecting training examples

`teacher.py` uses a locally running Ollama model to propose English, Arabic and Sorani → Badini examples. If you have already trained your personal model, it also asks your model to translate each source and sends the attempt to the teacher for correction. This is example generation, **not automatic training** or a guarantee that either model knows Badini. An LLM can produce wrong dialects, mistranslations or repeated sentences; a Badini/Sorani speaker must review them before training. The suggested teacher `qwen2.5:3b` has not been verified for Badini and can be replaced by another local model.

Install [Ollama](https://ollama.com/) and download a local model once while online:

```sh
ollama pull qwen2.5:3b
```

Run Ollama locally (`ollama serve` if it is not already running), then in the project directory:

```sh
python teacher.py generate --model qwen2.5:3b --count 10
# Use the bundled small student:
python teacher.py generate --model qwen2.5:3b --student-offline --count 0
# Or, once you have trained and saved your own neural student model:
python teacher.py generate --model qwen2.5:3b --student-model models/my-badini-ai --count 0
```

`--count 0` runs until Ctrl+C; completed rows are saved to `data/teacher-review.csv` as they are generated. There is no per-message API charge, but speed, memory, power, disk space and model quality impose practical limits. Generation can run without internet after Ollama and the selected model have been downloaded; use a local model, not an Ollama cloud model. Disconnect the machine from the internet if offline operation is required. The student only attempts source languages listed in its training metadata; other languages are generated by the teacher alone. Before a student exists, omit `--student-model`.

Open the review CSV in a spreadsheet, check and correct both texts, and set `approved` to `yes` only on trustworthy rows. Export **only approved** rows to a fresh file and train with it:

```sh
python teacher.py export --input data/teacher-review.csv --output data/approved.csv
python train_statistical.py --data data/approved.csv
# Or train the optional neural model:
python train.py --data data/approved.csv --output models/my-badini-ai
```

`--data` adds approved examples to the included dataset. Training starts from the Badini teacher model unless `--base` specifies an earlier personal model, or a matching Trainer checkpoint is specified via `--resume`; simply passing the same output directory does not continue an earlier model. For a later round use, for example, `python train.py --base models/my-badini-ai --data data/approved.csv --output models/my-badini-ai-v2`. In Colab, set `BASE` to the earlier model directory in Drive. Keep a separate untouched evaluation set across rounds to measure whether the model actually improves. See the Colab notebook if training needs a GPU. The CSV can also be copied to Google Drive and selected as `EXTRA_CSV` in that notebook. Never include confidential text without verifying your local setup and data handling.


## Generate examples using the Badini teacher

`distill_badini.py` uses [arkanit/badini-nllb-translator](https://huggingface.co/arkanit/badini-nllb-translator) to propose Badini translations of a CSV containing `source_lang,source_text`. English goes directly to the Badini teacher; Arabic and Sorani first pass through NLLB-200 English, so meaning can be lost. The result is an unapproved CSV compatible with `teacher.py export`. Teacher examples are not guaranteed correct and must be reviewed by a Badini speaker. Never report the teacher's outputs as human-verified data. Do not use your held-out evaluation sources as teacher input or training data.

```sh
# Once both model weights are already cached locally (or pass local paths):
python distill_badini.py --input data/sources.csv --output data/badini-review.csv
# Add --allow-download only when internet access is available and model downloads are intended.
# Review and set approved=yes on trustworthy rows, then:
python teacher.py export --input data/badini-review.csv --output data/badini-approved.csv
python train_statistical.py --data data/badini-approved.csv
# Or fine-tune the Badini neural model using train.py --data data/badini-approved.csv
```

This script defaults to offline mode and needs the PyTorch/Transformers dependencies and locally cached weights; `--allow-download` explicitly allows downloading them. Model weights are not included in this repository. A failed generation can leave a partial unapproved review CSV; use a new output path to retry. These teacher-generated examples have not been produced or evaluated in this environment. The local statistical model was actually retrained with the two published model-card examples; the full neural teacher model could not be downloaded here. Check both model and data rights before commercial use or redistribution.

## Experimental Badini LoRA adapter

`train_adapter.py` fine-tunes the Badini NLLB teacher with LoRA, then measures
teacher-forced held-out loss. The GitHub Actions workflow downloads the model,
runs a small CPU pilot (300 steps), and publishes an adapter and report as the
`badini-adapter` artifact for 30 days. The Sorani examples are automatically
matched and not reviewed; this pilot and its tiny held-out set are not proof
of translation quality or improvement. Check model and data licensing before
commercial use. The adapter is not a standalone model: its base weights and
Python dependencies are required.

To serve a downloaded and extracted artifact after installing requirements,
point the app at its `adapter` directory:

```sh
BADINI_ADAPTER_DIR=/path/to/badini-adapter/adapter python app.py
```

The first load requires the original base model to be cached or downloaded.
Once cached, it can run offline. The lightweight statistical model remains the
default; selecting the adapter does not change it. `BADINI_ADAPTER_DIR` takes
precedence over the other model settings.

The completed experimental run is available at
[GitHub Actions run 40986771788](https://github.com/s1po0/Translate-badini-/actions/runs/40986771788):
download its `badini-adapter` artifact before 2026-11-29. The run loaded the
saved adapter with the cached base model and generated outputs offline. Its
24-example held-out loss report is recorded in `trained/adapter-report.json`;
that small loss comparison is not evidence of real-world translation accuracy.
The live preview continues to use the small statistical model, not this adapter.

## Continuous teacher–student training (experimental)

`continuous_training.py` runs repeated rounds without an iteration limit by
default. In each round the multilingual NLLB model translates Arabic/Sorani
sources into English, a Badini NLLB teacher proposes Badini outputs, and the
student LoRA adapter trains on those **unverified synthetic pairs** plus
replayed seed data. It rotates between compatible teacher models specified with
`--teachers`; only one Badini teacher is configured by default. It does not
make models converse, and the source pool eventually repeats.

Each round evaluates the candidate on the same source-disjoint held-out sample
and only replaces the current adapter if overall loss does not increase and
no language loss regresses by more than 3%. This is an imperfect safety check,
not a guarantee of better translations; the Sorani data is automatically
matched and unverified. `state.json` is replaced atomically after each round,
so a failed or interrupted round leaves the previous accepted checkpoint intact.
A lock prevents concurrent processes from writing to the same run. Training
continues until stopped with Ctrl+C, but requires enough compute and disk;
it stops before free disk drops below 2 GiB. Teacher and student model licenses
and synthetic data should be reviewed before commercial use.

Install the dependencies in `requirements.txt`, download/cache the Badini and
multilingual NLLB models, and provide the previously trained LoRA adapter:

```sh
python continuous_training.py --initial-adapter models/badini-adapter/adapter
```

Add `--allow-download` on the first run if models are not cached. For a finite
smoke test use `--max-cycles 2`; `0` (the default) repeats. `--output` chooses
the run directory. After restarting, the program reads its saved state and
continues from its last accepted adapter. A training failure in unlimited mode
is retried after a delay; the current adapter is not replaced. Each round's
sources, synthetic pairs, report and candidate adapter are stored under
`models/continuous/rounds`. These checkpoints consume disk and are excluded
from Git. The `adapter` value in `models/continuous/state.json` is relative to
that directory and can be supplied to `BADINI_ADAPTER_DIR` to serve it. The
currently running public demo does not automatically switch checkpoints.

`.github/workflows/continuous-badini-test.yml` performs two real rounds on a
GitHub runner and uploads the resulting checkpoints for 30 days. It is a
finite test: GitHub jobs cannot run forever and runners have usage limits.

## Word-list quality audit and safer experiment

The full teacher run attempted 370,105 numbered words but a script-only check
cannot verify Badini or semantic correctness. The audit in
`trained/word-audit.json` found 240,132 outputs without Arabic script and
202,993 unchanged English outputs; 129,973 passed the script check, which is
not validation. Some outputs resemble Sorani or are otherwise inaccurate. The
previous all-word candidate was rejected because English held-out loss
exceeded the 3% regression guard; the old accepted adapter was retained.

The balanced experiment filters outputs by normalized spelling agreement with
existing *training-side* Badini dictionary references. This conservatively
reuses known pairs, not newly verified translations, and can miss valid
alternatives. It limits their share relative to replay, uses a smaller learning
rate and evaluates a larger source-disjoint sample. It also compares normalized
exact matches on held-out English dictionary words. Loss and exact-match guards
are imperfect; dictionary references may contain errors and the base may have
seen them. No model is automatically promoted on failed guards. The 90-day
GitHub artifacts are temporary and the demo does not automatically load the
new checkpoint.

The completed balanced run is
[41036334780](https://github.com/s1po0/Translate-badini-/actions/runs/41036334780);
its `balanced-word-training` artifact contains the accepted adapter until
2027-01-28. Out of 6,182 indexed words with training-side dictionary entries,
244 teacher outputs agreed after normalization. This is not independent
verification: alternative valid forms do not count as matches. The 1,200-step
candidate passed both guards: mean loss on 60 examples was 3.3331 → 3.1830,
and normalized exact matches on 100 held-out dictionary words were 6 → 7.
These are small, imperfect tests and do not establish general accuracy. Details
are in `trained/balanced-word-report.json`.

## Checkpointed full word-pair training

`.github/workflows/full-word-training.yml` splits the 129,973 script-passing,
**unverified** teacher outputs into source-disjoint train and held-out groups,
then sequentially trains all 123,524 eligible training pairs in 25 shards.
It starts with the accepted checkpoint from run 41036334780. Each shard is
trained for one full pass of the shard and replay pool, then the held-out loss
and word exact-match guards decide whether to keep its candidate or restore the
prior adapter. The next shard starts from the selected adapter. `progress.json`
separately counts pairs seen in attempted and accepted stages: a rejected stage
is not retained, so processing every pair does **not** imply the final model
learned every word. Repeatedly using the same small tests for decisions can
also overfit the tests.

The 240,132 outputs that did not even pass the Arabic-script check are not
translations and cannot be used as valid Badini labels. The 6,449 held-out
script-passing pairs are kept out of training. Neither passing a script check
nor passing the guards establishes correctness, meaning, or Badini dialect;
human review is still needed. The model search found only one suitable
Badini text-translation teacher (`arkanit/badini-nllb-translator`); Badini
speech recognition and speech synthesis models are different tasks and are
not claimed as translation teachers. Other compatible local translation
teachers can be configured in `continuous_training.py`; the multilingual NLLB
pivot is not a Badini-specialized model.

Each GitHub job has a six-hour limit; the full workflow runs jobs sequentially
and may take much longer than six hours overall. It can fail, be queued or hit
GitHub limits. Intermediate stage artifacts expire after two days, the shards
after seven days, and the final stage after 90 days; they are not permanent
backups. GitHub Actions is not unlimited hosting or compute. The public demo
does not automatically switch to the final adapter. Re-run failures promptly
while the needed artifacts exist; there is no guaranteed automatic recovery.

## Dasinya Badini corpus vocabulary experiment

[Dasinya](https://doi.org/10.5281/zenodo.20729060), Vaman A. Saeed and Karwan
Jacksi (2026), is published under CC BY 4.0. `.github/workflows/dasinya-badini-training.yml`
downloads and checksum-verifies the publisher's archive and extracts Arabic-script
word forms and frequencies from its 107 segmented documents. The vocabulary is
an observed corpus list, not all possible Badini words; it can contain variant
spellings, names, OCR errors and non-Badini text. The corpus has Badini text,
not English translations. The experiment backtranslates only a sample of 1,024
frequent forms through the available Badini NLLB model and trains on at most
350 unverified generated pairs with replay. Automatic English-script checks do
not prove translation correctness. Source-disjoint held-out loss and dictionary
word checks choose the checkpoint, but are small and imperfect; the base model
may have seen evaluation data. Candidate/selected adapters, vocabulary and
backtranslations are uploaded as a temporary 90-day GitHub artifact. No live
server automatically uses this separate candidate.

The full follow-up `.github/workflows/full-badini-vocabulary-training.yml`
attempts English backtranslation for each of the 98,305 observed forms from
the corpus, verifies numbered coverage, excludes failed English-script outputs
and keeps a source-disjoint holdout. It then trains every remaining pair once
in 20 sequential checkpointed stages with replay and guard checks. A rejected
stage's candidate is not retained, and no claim of learning all the vocabulary
or of translation accuracy follows from processing the data. Source words and
English backtranslations may be incorrect, and the small gates can miss errors.
The public preview and the separate English-word training workflow do not
implicitly receive the selected checkpoints. Intermediate artifacts expire;
GitHub jobs and resources have limits, so an entire run is not guaranteed.

The verified first corpus experiment completed in
[run 41041856775](https://github.com/s1po0/Translate-badini-/actions/runs/41041856775):
107 segmented documents yielded 1,270,173 word tokens and 98,305 distinct
observed spellings. Of 1,024 word forms sent through the Badini-to-English
teacher, 1,011 passed a basic English-script check; 350 of those unverified
pairs were sampled for training. The candidate passed its configured guards:
loss on 60 examples fell from 3.1830 to 3.0036, while exact matches on 100
held-out English dictionary words changed from 7 to 6 (the guard permits a
small regression). The selected adapter and list are in the 90-day
`dasinya-badini-student` artifact. These small metrics do not prove improvement.

## Parallel training comparison

`.github/workflows/parallel-badini-training.yml` runs four actual CPU training
jobs concurrently from the *same* accepted Dasinya adapter, with different
learning rates, seeds and synthetic sample limits. A fifth job compares their
60-example held-out losses and 100 word exact matches and selects one candidate
only if it passes the per-language loss guard and does not reduce exact matches;
otherwise it retains the baseline. These jobs are independent experiments,
not one model trained on four machines, and LoRA weights are not merged. Their
small repeated evaluations and unverified teacher labels do not establish
real-world improvement. Candidate artifacts expire after 2 days and the
selected adapter after 90 days. This is separate from the sequential full-data
training, whose stages depend on the preceding checkpoint.

The first four-job comparison completed in
[run 41047507896](https://github.com/s1po0/Translate-badini-/actions/runs/41047507896).
The selected `balanced` candidate reduced mean held-out loss from 3.0036 to
2.9504, while exact matches remained 6/100; neither establishes accuracy or
a general improvement. The selected adapter is available in the
`parallel-badini-selected` artifact until 2027-01-28; detailed results are in
`trained/parallel-report.json`.

### Long, parallel related-teacher experiment

`multi-teacher-long-training.yml` compares 4,096 numbered English-word attempts from the original Badini NLLB teacher against newly generated outputs from its own previously fine-tuned adapter, in four parallel jobs. These are **two versions of the same base model**, not independent Badini foundation models, and no Gemini API is used. Only normalized text agreements that also pass an Arabic-script check enter the synthetic training dataset; neither test establishes dialect or semantic correctness. Four independent candidate adapters then train in parallel from the same previous selected checkpoint (6,000–9,000 optimizer steps, six-hour per-job limit). They are compared on small source-disjoint tests and the previous adapter is retained if none passes the guards. The candidates are not merged, and completion, selection, and model quality must be checked in the run; no live model is automatically updated.
