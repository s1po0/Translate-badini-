# Badini trainer

A static, personal Duolingo-inspired teaching and translation interface. User-supplied source/Badini pairs are stored only in localStorage on the current browser/device and applied immediately, with exact user phrases taking precedence. Individual corrected words and short personal phrases can also be matched inside larger inputs. JSON backup/restore is available. Points reflect the current number of saved unique pairs, not a correctness assessment.

The bundled translator is the repository's small statistical model (trained/badini-statistical.json.gz), not the selected neural LoRA adapter. No neural weights are updated and nothing is synchronized or submitted to a training service. Clearing site data, using another browser/device, private browsing, or storage failure can lose access to data; back up regularly. Translation and labels in the bundled model are unverified. Service worker/automatic server-side learning are not implemented.

Run locally with `python3 -m http.server 8080 --directory trainer`. The application requires a modern browser supporting DecompressionStream. The external font is optional; system fallback is provided.

## Private neural fine-tuning (not connected to the website)

`tools/train_reviewed.py` can train a new LoRA adapter using a JSON backup from this site, starting from a previously saved compatible adapter. It includes every unique reviewed pair at least once, excludes matching sources from replay and held-out evaluation, and preserves the previous adapter if either of its regression guards fails. If any reviewed pair exceeds 64 tokenizer tokens, the run stops rather than silently truncating it. A pass through the optimizer is not proof that the model can reproduce each example. The accepted adapter must separately be loaded by the neural translator; the website does not do this automatically. A suitable private machine with the model weights and training dependencies is required. This command is not a deployed service.

Backups and trained weights can reveal personal examples: do not upload them to public GitHub repositories, public workflow artifacts, or an unauthenticated service. The website still stores examples only in the browser. The user's real examples are never used by the demonstration integration test.
