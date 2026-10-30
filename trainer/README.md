# Badini trainer

A static, personal Duolingo-inspired teaching and translation interface. User-supplied source/Badini pairs are stored only in localStorage on the current browser/device and applied immediately, with exact user phrases taking precedence. Individual corrected words and short personal phrases can also be matched inside larger inputs. JSON backup/restore is available. Points reflect the current number of saved unique pairs, not a correctness assessment.

The bundled translator is the repository's small statistical model (trained/badini-statistical.json.gz), not the selected neural LoRA adapter. No neural weights are updated and nothing is synchronized or submitted to a training service. Clearing site data, using another browser/device, private browsing, or storage failure can lose access to data; back up regularly. Translation and labels in the bundled model are unverified. Service worker/automatic server-side learning are not implemented.

Run locally with `python3 -m http.server 8080 --directory trainer`. The application requires a modern browser supporting DecompressionStream. The external font is optional; system fallback is provided.
