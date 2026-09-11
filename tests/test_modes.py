import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generation.adhoc_query_rewriting import ResultWriter, load_conversations, run_batch, run_interactive, main


class ModesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if "Classify it as either" in prompt:
            return json.dumps({"topic": "old_topic",
                               "needs_clarification": False, "reason": "대상 변경"})
        if "### Ambiguous Question" in prompt:
            return "C2라인 불량률"
        return '{"query": "재작성 결과"}'

    def test_batch_includes_each_turn_with_no_future_context_or_cross_conversation_leak(self):
        data = [{"id":"a", "questions":["C1라인 불량률", "C2라인은?"]},
                {"id":"b", "questions":["생산량"]}]
        writer = ResultWriter(self.root / "all.json")
        with contextlib.redirect_stdout(io.StringIO()):
            run_batch(data, self.generate, writer)
        rows = json.loads(writer.path.read_text())
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["turn"] for row in rows], [1, 2, 1])
        self.assertEqual(rows[1]["history_questions"], ["C1라인 불량률"])
        self.assertEqual(rows[2]["history_questions"], [])
        self.assertEqual(rows[0]["query"], "C1라인 불량률")
        self.assertEqual(rows[2]["query"], "생산량")
        self.assertEqual(len(self.prompts), 3)
        self.assertTrue(all("생산량" not in prompt for prompt in self.prompts))

    def test_interactive_reset_history_empty_lines_and_exit(self):
        writer = ResultWriter(self.root / "live.jsonl")
        entries = ["", "C1라인 불량률", "/history", "C2라인은?", "/reset", "생산량", "/exit"]
        with patch("builtins.input", side_effect=entries), contextlib.redirect_stdout(io.StringIO()) as screen:
            run_interactive(self.generate, writer)
        rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
        self.assertEqual([row["conversation_id"] for row in rows], ["1", "1", "2"])
        self.assertEqual([row["turn"] for row in rows], [1, 2, 1])
        self.assertEqual(rows[-1]["history_questions"], [])
        self.assertIn("Queryrewriting:", screen.getvalue())
        self.assertNotIn("최종 재작성:", screen.getvalue())
        self.assertNotIn("생략 복원:", screen.getvalue())
        self.assertEqual(rows[0]["query"], "C1라인 불량률")
        self.assertEqual(rows[-1]["query"], "생산량")
        self.assertEqual(len(self.prompts), 3)

    def test_interactive_retry_after_api_error(self):
        outputs = iter([RuntimeError("API error"), json.dumps({"topic": "old_topic", "needs_clarification": False, "reason": "대상 변경"}), "복원 질문", '{"query":"성공"}'])
        def generate(prompt):
            value = next(outputs)
            if isinstance(value, Exception):
                raise value
            return value
        writer = ResultWriter(self.root / "live.json")
        with patch("builtins.input", side_effect=["첫 질문", "실패할 질문", "다시 질문", "/exit"]), contextlib.redirect_stdout(io.StringIO()):
            run_interactive(generate, writer)
        rows = json.loads(writer.path.read_text())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["turn"], 2)
        self.assertEqual(rows[1]["history_questions"], ["첫 질문"])
        self.assertEqual(rows[1]["query"], "성공")

    def test_new_topic_starts_next_conversation_at_turn_one(self):
        outputs = iter([
            json.dumps({"topic": "old_topic", "needs_clarification": False, "reason": "조건 변경"}),
            "조건 복원", '{"query":"조건 재작성"}',
            json.dumps({"topic": "new_topic", "needs_clarification": False, "reason": "새 주제"}),
            '{"query":"UPH 질문"}',
            json.dumps({"topic": "old_topic", "needs_clarification": False, "reason": "대상 변경"}),
            "후속 복원", '{"query":"후속 재작성"}',
        ])
        writer = ResultWriter(self.root / "numbering.json")
        entries = ["첫 대화", "/reset", "병목 공정", "Palletizing 제외", "UPH 알려줘", "다른 법인은?", "/reset", "마지막 대화", "/exit"]
        with patch("builtins.input", side_effect=entries), contextlib.redirect_stdout(io.StringIO()) as screen:
            run_interactive(lambda prompt: next(outputs), writer)
        rows = json.loads(writer.path.read_text())
        self.assertEqual([(r["conversation_id"], r["turn"]) for r in rows],
                         [("1", 1), ("2", 1), ("2", 2), ("3", 1), ("3", 2), ("4", 1)])
        self.assertIn("[3 / 턴 1]", screen.getvalue())
        self.assertNotIn("[2 / 턴 3]", screen.getvalue())
        self.assertEqual(rows[4]["history_questions"], ["UPH 알려줘"])

    def test_input_formats_and_invalid_data(self):
        path = self.root / "input.json"
        for value in [["첫 질문", "후속 질문"], {"conversations":[{"id":"x", "questions":["질문"]}]}]:
            path.write_text(json.dumps(value))
            self.assertTrue(load_conversations(path))
        for value in [[], [""], [1], {"conversations": []}, {"questions": ["Q"]},
                      {"conversations": [{"id":"x", "questions":["Q"]}, {"id":"x", "questions":["Q"]}]}]:
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_conversations(path)

    def test_existing_results_are_not_overwritten(self):
        path = self.root / "output.json"
        path.write_text('["previous result"]')
        with self.assertRaises(FileExistsError):
            ResultWriter(path)
        self.assertEqual(json.loads(path.read_text()), ["previous result"])

    def test_cli_batch_uses_json_input_and_saves_all_results(self):
        source, target = self.root / "input.json", self.root / "result.json"
        source.write_text('["Q1", "Q2"]')
        with patch("sys.argv", ["rewrite", "--questions", str(source), "--output", str(target)]), \
             patch("generation.adhoc_query_rewriting.load_openai_generator", return_value=self.generate), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(len(json.loads(target.read_text())), 2)

    def test_interactive_eof_saves_valid_empty_json(self):
        writer = ResultWriter(self.root / "empty.json")
        with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            run_interactive(self.generate, writer)
        self.assertEqual(json.loads(writer.path.read_text()), [])


if __name__ == "__main__":
    unittest.main()
