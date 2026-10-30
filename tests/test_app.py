import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import app


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, data=None):
        conn = HTTPConnection("127.0.0.1", self.server.server_port)
        body = json.dumps(data) if data is not None else None
        conn.request(method, path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def test_page(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("وەرگێڕا بادینی", body.decode())

    def test_translation(self):
        with patch.object(app.translator, "translate", return_value="سلاڤ") as translate:
            status, body = self.request("POST", "/translate", {"text": "Hello", "source": "en", "target": "badini"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["translation"], "سلاڤ")
        translate.assert_called_once_with("Hello", "en", "badini")

    def test_new_source_languages(self):
        for source in ("ar", "sorani"):
            with self.subTest(source=source), patch.object(app.translator, "translate", return_value="بادینی") as translate:
                status, body = self.request("POST", "/translate", {"text": "test", "source": source, "target": "badini"})
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["translation"], "بادینی")
                translate.assert_called_once_with("test", source, "badini")

    def test_pivot_pipeline(self):
        translator = app.Translator()
        for source in ("ar", "sorani"):
            with self.subTest(source=source), patch.object(translator, "_run", side_effect=["English pivot", "بادینی"]) as run:
                self.assertEqual(translator.translate("source", source, "badini"), "بادینی")
                self.assertEqual(run.call_args_list[0].args, (app.PIVOT_MODEL, "source", source, "en"))
                self.assertEqual(run.call_args_list[1].args, (app.BADINI_MODEL, "English pivot", "en", "badini"))

    def test_direct_pipeline(self):
        translator = app.Translator()
        with patch.object(translator, "_run", return_value="بادینی") as run:
            self.assertEqual(translator.translate("hello", "en", "badini"), "بادینی")
            run.assert_called_once_with(app.BADINI_MODEL, "hello", "en", "badini")

    def test_invalid_inputs(self):
        for data in ({"text": " ", "source": "en", "target": "badini"},
                     {"text": "hi", "source": "en", "target": "en"},
                     {"text": "hi", "source": "ar", "target": "en"},
                     {"text": "x" * 4001, "source": "en", "target": "badini"}):
            self.assertEqual(self.request("POST", "/translate", data)[0], 400)

    def test_model_failure(self):
        with patch.object(app.translator, "translate", side_effect=RuntimeError("no model")):
            status, body = self.request("POST", "/translate", {"text": "hi", "source": "en", "target": "badini"})
        self.assertEqual(status, 500)
        self.assertIn("error", json.loads(body))


if __name__ == "__main__":
    unittest.main()
