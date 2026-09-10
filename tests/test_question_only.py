import json
import unittest

from generation.adhoc_query_rewriting import QuestionOnlySession, rewrite_question
from generation.enhancement_generation import RELATIONS, NEW_TOPIC, classify_relation


class FakeModel:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return next(self.outputs)


def ts(relation, blocked=False):
    return json.dumps({"relation": relation,
                       "needs_clarification": blocked, "reason": "Missing answer" if blocked else "Relation found"})


class QuestionOnlyTests(unittest.TestCase):
    def test_all_four_relations_feed_qd_and_rewriter(self):
        for relation in RELATIONS:
            if relation == NEW_TOPIC:
                continue
            with self.subTest(relation=relation):
                model = FakeModel(ts(relation), "명확한 질문", '{"query": "최종 질문"}')
                result = rewrite_question(["이전 질문 1", "이전 질문 2"], "새 질문", model)
                self.assertEqual(result["relation"], relation)
                self.assertEqual(result["query"], "최종 질문")
                self.assertEqual(len(model.prompts), 3)
                for prompt in model.prompts:
                    self.assertIn("이전 질문 1", prompt)
                    self.assertIn("이전 질문 2", prompt)
                    self.assertNotIn("question-and-answer pairs", prompt)
                self.assertIn(relation, model.prompts[1])
                self.assertIn("명확한 질문", model.prompts[2])

    def test_first_turn_preserves_exact_input_without_any_model_call(self):
        model = FakeModel()
        question = "  알람 Top3 알려줘!\n"
        result = rewrite_question([], question, model)
        self.assertEqual(result["query"], question)
        self.assertEqual(result["disambiguated_question"], question)
        self.assertEqual(result["relation"], NEW_TOPIC)
        self.assertEqual(len(model.prompts), 0)

    def test_missing_answer_stops_before_qd_and_rewrite(self):
        model = FakeModel(ts("Answer Exploration", True))
        result = rewrite_question(["불량률 상위 3개 제품은?"], "그중 첫 번째의 생산량은?", model)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIsNone(result["query"])
        self.assertEqual(len(model.prompts), 1)

    def test_new_topic_clears_previous_session_context(self):
        model = FakeModel(ts(NEW_TOPIC), '{"query":"내일 날씨"}',
                          ts("Constraint Refinement"), "모레 날씨", '{"query":"모레 날씨"}')
        session = QuestionOnlySession(model)
        session.ask("제품 불량률")
        session.ask("내일 날씨")
        self.assertEqual(session.history, ["내일 날씨"])
        self.assertNotIn("제품 불량률", model.prompts[1])
        session.ask("모레는?")
        self.assertNotIn("제품 불량률", model.prompts[2])
        self.assertEqual(session.history, ["내일 날씨", "모레는?"])

    def test_classifier_uses_exactly_five_labels_and_derives_topic(self):
        self.assertEqual(len(RELATIONS), 5)
        for relation in RELATIONS:
            with self.subTest(relation=relation):
                model = FakeModel(ts(relation))
                result = classify_relation(["이전 질문"], "현재 질문", model)
                self.assertEqual(result["relation"], relation)
                self.assertEqual(result["topic"], "new_topic" if relation == NEW_TOPIC else "old_topic")
                self.assertEqual(len(model.prompts), 1)
                self.assertIn("five categories", model.prompts[0])
                self.assertNotIn('"topic":', model.prompts[0])

    def test_new_topic_is_explicit_in_output_and_skips_qd(self):
        model = FakeModel(ts(NEW_TOPIC), '{"query":"내일 날씨"}')
        result = rewrite_question(["제품 불량률"], "내일 날씨", model)
        self.assertEqual(result["relation"], "New Topic")
        self.assertEqual(len(model.prompts), 2)
        self.assertNotIn("제품 불량률", model.prompts[-1])

    def test_null_relation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "five relation categories"):
            classify_relation(["이전 질문"], "새 질문", FakeModel(ts(None)))

    def test_blocked_turn_does_not_pollute_history(self):
        model = FakeModel(ts("Answer Exploration", True))
        session = QuestionOnlySession(model)
        session.ask("상위 제품")
        session.ask("첫 번째는?")
        self.assertEqual(session.history, ["상위 제품"])

    def test_invalid_inputs_rejected_before_model_call(self):
        for history, question in [([{"question": "Q", "answer": "A"}], "Q"), ("Q", "Q"), ([""], "Q"), ([], "")]:
            model = FakeModel()
            with self.assertRaises(ValueError):
                rewrite_question(history, question, model)
            self.assertEqual(model.prompts, [])

    def test_malformed_model_outputs_fail_explicitly(self):
        for output in ["old_topic", ts("unknown"), '{"topic":"old_topic"}',
                       '{"topic":"new_topic","relation":null,"needs_clarification":"false","reason":""}']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                rewrite_question(["Q1"], "Q2", FakeModel(output))
        with self.assertRaises(ValueError):
            rewrite_question(["Q1"], "Q2", FakeModel(ts("Participant Shift"), "Q2 clarified", '{"query":123}'))
        with self.assertRaises(ValueError):
            rewrite_question(["Q1"], "Q2", FakeModel(ts("Participant Shift"), ""))

    def test_three_turn_history_preserves_every_question(self):
        model = FakeModel(ts("Participant Shift"), "C2라인 불량률",
                          '{"query":"C2라인 불량률"}', ts("Constraint Refinement"), "C2라인 지난달 불량률",
                          '{"query":"C2라인 지난달 불량률"}')
        session = QuestionOnlySession(model)
        for question in ["C1라인 불량률", "C2라인은?", "지난달은?"]:
            session.ask(question)
        self.assertIn('"C1라인 불량률", "C2라인은?"', model.prompts[3])
        self.assertEqual(session.history, ["C1라인 불량률", "C2라인은?", "지난달은?"])


if __name__ == "__main__":
    unittest.main()
