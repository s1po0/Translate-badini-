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
runs a small CPU pilot (12 steps), and publishes an adapter and report as the
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
