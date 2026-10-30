"""Local translation into Badini using published NLLB models."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BADINI_MODEL = "arkanit/badini-nllb-translator"
PIVOT_MODEL = "facebook/nllb-200-distilled-600M"
LANGUAGES = {"en": "eng_Latn", "badini": "ckb_Arab", "sorani": "ckb_Arab", "ar": "arb_Arab"}
SUPPORTED_PAIRS = {("en", "badini"), ("badini", "en"), ("sorani", "badini"), ("ar", "badini")}
MAX_TEXT_LENGTH = 4000
ROOT = Path(__file__).resolve().parent


class Translator:
    def __init__(self):
        self.lock = threading.Lock()
        self.models = {}
        self.own_model = os.environ.get("BADINI_MODEL_DIR")
        self.own_languages = None
        if self.own_model:
            metadata = Path(self.own_model) / "badini_model.json"
            self.own_languages = set(json.loads(metadata.read_text(encoding="utf-8"))["source_languages"])

    def _run(self, model_id, text, source, target):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if model_id not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            model.eval()
            self.models[model_id] = (tokenizer, model)
        tokenizer, model = self.models[model_id]
        tokenizer.src_lang = LANGUAGES[source]
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("Text is too long for the model. Please translate a shorter passage.")
        with torch.inference_mode():
            result = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANGUAGES[target]),
                max_new_tokens=512,
            )
        return tokenizer.decode(result[0], skip_special_tokens=True)

    def translate(self, text, source, target):
        if (source, target) not in SUPPORTED_PAIRS:
            raise ValueError("Unsupported language pair.")
        # Protect tokenizer state and prevent simultaneous high-memory inference.
        with self.lock:
            if self.own_model:
                if target != "badini" or source not in self.own_languages:
                    raise ValueError("This personal model was not trained for this language direction.")
                return self._run(self.own_model, text, source, target)
            if source in ("sorani", "ar"):
                text = self._run(PIVOT_MODEL, text, source, "en")
                source = "en"
            return self._run(BADINI_MODEL, text, source, target)


class AdapterTranslator(Translator):
    """Serve a trained local LoRA adapter on its recorded base model."""
    def __init__(self, adapter_path):
        self.lock = threading.Lock()
        self.models = {}
        self.adapter_path = str(Path(adapter_path).resolve())

    def translate(self, text, source, target):
        if target != "badini" or source not in ("en", "ar", "sorani"):
            raise ValueError("ئەم مۆدێلە تەنها بۆ بادینی وەردەگێڕێت.")
        with self.lock:
            if self.adapter_path not in self.models:
                from peft import PeftConfig, PeftModel
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                config = PeftConfig.from_pretrained(self.adapter_path)
                tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
                base = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(base, self.adapter_path)
                model.eval()
                self.models[self.adapter_path] = (tokenizer, model)
            return self._run(self.adapter_path, text, source, target)


if os.environ.get("BADINI_ADAPTER_DIR"):
    translator = AdapterTranslator(os.environ["BADINI_ADAPTER_DIR"])
elif os.environ.get("TRANSLATOR_MODE") == "neural" or os.environ.get("BADINI_MODEL_DIR"):
    translator = Translator()
else:
    from statistical import StatisticalTranslator
    translator = StatisticalTranslator()


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.respond(404, {"error": "Not found"})
            return
        self.respond(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/translate":
            self.respond(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 32000:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Invalid request.")
            text, source, target = data.get("text"), data.get("source"), data.get("target")
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Enter text between 1 and {MAX_TEXT_LENGTH} characters.")
            if not isinstance(source, str) or not isinstance(target, str) or (source, target) not in SUPPORTED_PAIRS:
                raise ValueError("Unsupported language pair.")
            result = translator.translate(text.strip(), source, target)
            self.respond(200, {"translation": result})
        except (ValueError, json.JSONDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception as exc:
            print(f"Translation failed: {exc}", flush=True)
            self.respond(500, {"error": "Could not load or run the model. Check the server logs, dependencies and model download."})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Translator running at http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
