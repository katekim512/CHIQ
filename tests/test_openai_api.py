import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from generation.adhoc_query_rewriting import load_openai_generator, QuestionOnlySession


class OpenAIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env_file = Path(self.tmp.name) / ".env"
        self.env_file.write_text("OPENAI_API_KEY=test-fake-key\nOPENAI_MODEL=gpt-4.1-mini\n")
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def client_patch(self, handler):
        client = OpenAI(api_key="test-fake-key", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return patch("openai.OpenAI", return_value=client)

    @staticmethod
    def response(text, status="completed"):
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1,
            "model": "gpt-4.1-mini", "status": status,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        })

    def test_real_sdk_serializes_multiturn_requests_without_changing_prompts(self):
        outputs = iter([
            json.dumps({"topic":"old_topic",
                        "needs_clarification":False, "reason":"Line changed"}),
            "C2라인 불량률", '{"query":"C2라인 불량률"}',
        ])
        bodies = []
        def handler(request):
            self.assertEqual(request.url.path, "/v1/responses")
            bodies.append(json.loads(request.content))
            return self.response(next(outputs))
        with self.client_patch(handler):
            session = QuestionOnlySession(load_openai_generator(env_file=self.env_file))
            first = session.ask("C1라인 불량률")
            self.assertEqual(first["query"], "C1라인 불량률")
            self.assertEqual(bodies, [])
            result = session.ask("C2라인은?")
        self.assertEqual(result["query"], "C2라인 불량률")
        self.assertEqual(len(bodies), 3)
        for body in bodies:
            self.assertEqual(body["model"], "gpt-4.1-mini")
            self.assertIs(body["store"], False)
            self.assertNotIn("previous_response_id", body)
        self.assertIn("C1라인 불량률", bodies[0]["input"])
        self.assertNotIn("### Relation", bodies[1]["input"])

    def test_env_and_explicit_model_override_file(self):
        def explicit_handler(request):
            self.assertEqual(json.loads(request.content)["model"], "explicit-model")
            self.assertEqual(json.loads(request.content)["input"], "prompt")
            return self.response("ok")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-fake-key", "OPENAI_MODEL": "env-model"}):
            with self.client_patch(explicit_handler) as factory:
                generate = load_openai_generator(model="explicit-model", env_file=self.env_file)
                self.assertEqual(factory.call_args.kwargs["api_key"], "env-fake-key")
                generate("prompt")
        # Environment precedence is also checked in the serialized request.
        def handler(request):
            self.assertEqual(json.loads(request.content)["model"], "env-model")
            return self.response("ok")
        with patch.dict(os.environ, {"OPENAI_MODEL": "env-model"}), self.client_patch(handler):
            load_openai_generator(env_file=self.env_file)("prompt")

    def test_missing_key_fails_before_client_creation(self):
        self.env_file.write_text("OPENAI_API_KEY=\n")
        with patch("openai.OpenAI") as factory, self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            load_openai_generator(env_file=self.env_file)
        factory.assert_not_called()

    def test_error_body_is_not_exposed(self):
        for code in [401, 429, 500]:
            def handler(request):
                return httpx.Response(code, json={"error": {"message": "secret-value-in-server-error", "type": "api_error"}})
            with self.subTest(code=code), self.client_patch(handler):
                generate = load_openai_generator(env_file=self.env_file)
                with self.assertRaises(RuntimeError) as error:
                    generate("prompt")
                self.assertIn(str(code), str(error.exception))
                self.assertNotIn("secret-value", str(error.exception))

    def test_empty_or_incomplete_response_is_rejected(self):
        for text, status in [("", "completed"), ("partial", "incomplete")]:
            with self.subTest(status=status), self.client_patch(lambda request: self.response(text, status)):
                with self.assertRaises(RuntimeError):
                    load_openai_generator(env_file=self.env_file)("prompt")


if __name__ == "__main__":
    unittest.main()
