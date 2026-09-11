"""Question-only TS -> QD -> ad-hoc rewriting, with an injectable LLM callable."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

try:
    from .enhancement_generation import (
        GROUNDING, PROMPT_DICT, PRESERVE_CONTEXT, NEW_TOPIC, classify_relation, context, disambiguate,
        parse_json, validate_questions,
    )
except ImportError:
    from enhancement_generation import (
        GROUNDING, PROMPT_DICT, PRESERVE_CONTEXT, NEW_TOPIC, classify_relation, context, disambiguate,
        parse_json, validate_questions,
    )


def rewrite_question(history, question, generate):
    """history contains previous questions only; question is the current turn."""
    validate_questions(history, question)
    classification = classify_relation(history, question, generate)
    result = {"question": question, **classification}
    if not history:
        return {**result, "status": "ok", "disambiguated_question": question, "query": question}
    if classification["needs_clarification"]:
        return {**result, "status": "needs_clarification", "disambiguated_question": None, "query": None}
    active_history = history if classification["topic"] == "old_topic" else []
    clarified = disambiguate(active_history, question, generate) if active_history else question
    prompt = PROMPT_DICT["search_query"] + GROUNDING + "\n\n" + PRESERVE_CONTEXT + context(active_history, question)
    prompt += "\n### Disambiguated Question\n" + clarified
    rewritten = parse_json(generate(prompt))
    if not isinstance(rewritten.get("query"), str) or not rewritten["query"].strip():
        raise ValueError("Rewriter must return a non-empty query string.")
    return {**result, "status": "ok", "disambiguated_question": clarified, "query": rewritten["query"].strip()}


class QuestionOnlySession:
    """State contains accepted user questions only, never generated answers."""
    def __init__(self, generate):
        self.generate = generate
        self.history = []

    def ask(self, question):
        result = rewrite_question(self.history, question, self.generate)
        if result["status"] == "ok":
            if result["topic"] == "new_topic":
                self.history.clear()
            self.history.append(question)
        return result


def load_openai_generator(model=None, env_file=None):
    """Load the project .env without overriding existing environment variables."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI, OpenAIError
    except ImportError as exc:
        raise RuntimeError("Install API dependencies: python -m pip install -r requirements-api.txt") from exc

    env_path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key in ("your_api_key_here", "sk-your-api-key-here"):
        raise ValueError(f"Set OPENAI_API_KEY in {env_path} or your environment before running.")
    model = model or os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4.1-mini"
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1", timeout=60.0, max_retries=2)

    def generate(prompt):
        try:
            response = client.responses.create(model=model, input=prompt, store=False)
        except OpenAIError as exc:
            # SDK error bodies may include credentials; surface only type/status.
            code = getattr(exc, "status_code", None)
            raise RuntimeError(
                f"OpenAI API request failed ({type(exc).__name__}, HTTP {code or 'unavailable'}). "
                "Check your API key, API billing/quota, model access, and network."
            ) from None
        if response.status != "completed":
            raise RuntimeError("OpenAI returned an incomplete response; no query was accepted.")
        if not response.output_text or not response.output_text.strip():
            raise RuntimeError("OpenAI returned no text (possibly a refusal); no query was accepted.")
        return response.output_text.strip()

    return generate


def load_conversations(path):
    """Accept a single question array or explicitly separated conversations."""
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        conversations = [{"id": "1", "questions": data}]
    elif isinstance(data, dict) and set(data) == {"conversations"}:
        conversations = data["conversations"]
    else:
        raise ValueError('Input must be a question array or {"conversations": [{"id": "...", "questions": [...]}]}.')
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("At least one conversation is required.")
    seen = set()
    for conversation in conversations:
        if not isinstance(conversation, dict) or set(conversation) != {"id", "questions"}:
            raise ValueError("Each conversation must contain only id and questions.")
        identifier = conversation["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Conversation IDs must be unique non-empty strings.")
        seen.add(identifier)
        validate_questions(conversation["questions"], "validate")
        if not conversation["questions"]:
            raise ValueError("Each conversation needs at least one question.")
    return conversations


class ResultWriter:
    """Persist every completed turn; .json is an array, .jsonl is one object per line."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix.lower() not in (".json", ".jsonl"):
            raise ValueError("Output must end in .json or .jsonl.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        # Avoid silently overwriting earlier experiment results.
        with self.path.open("x", encoding="utf-8") as stream:
            if self.path.suffix.lower() == ".json":
                stream.write("[]\n")

    def write(self, row):
        if self.path.suffix.lower() == ".jsonl":
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            self.rows.append(row)
            with self.path.open("w", encoding="utf-8") as stream:
                json.dump(self.rows, stream, ensure_ascii=False, indent=2)
                stream.write("\n")


def ask_turn(session, question, conversation_id, turn):
    history = list(session.history)
    result = session.ask(question)
    return {"conversation_id": conversation_id, "turn": turn, "history_questions": history, **result}


def display_result(row):
    print(f"\n[{row['conversation_id']} / 턴 {row['turn']}]")
    print(f"입력: {row['question']}")
    print(f"Topic: {row['topic']}")
    if row["status"] == "needs_clarification":
        print(f"확인 필요: {row['reason']}")
    else:
        print(f"Queryrewriting: {row['query']}")


def run_batch(conversations, generate, writer):
    for conversation in conversations:
        session = QuestionOnlySession(generate)
        for turn, question in enumerate(conversation["questions"], 1):
            row = ask_turn(session, question, conversation["id"], turn)
            writer.write(row)
            display_result(row)


def run_interactive(generate, writer):
    session = QuestionOnlySession(generate)
    conversation, turn = 1, 0
    print("질문을 입력하세요. /reset 새 대화 · /history 이전 질문 · /exit 종료")
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            session = QuestionOnlySession(generate)
            conversation, turn = conversation + 1, 0
            print("새 대화를 시작합니다.")
            continue
        if question == "/history":
            print(json.dumps(session.history, ensure_ascii=False, indent=2))
            continue
        if question.startswith("/"):
            print("사용 가능한 명령: /reset, /history, /exit")
            continue
        print("재작성 중...", flush=True)
        try:
            row = ask_turn(session, question, str(conversation), turn + 1)
        except (ValueError, RuntimeError) as exc:
            print(f"처리 실패: {exc}\n대화는 유지됩니다. 다시 입력하세요.")
            continue
        if turn > 0 and row["status"] == "ok" and row["topic"] == NEW_TOPIC:
            conversation += 1
            turn = 0
        turn += 1
        row["conversation_id"] = str(conversation)
        row["turn"] = turn
        writer.write(row)
        display_result(row)


def main():
    parser = argparse.ArgumentParser(description="Question-only TS / QD / query rewriting")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--interactive", action="store_true", help="Enter questions interactively")
    modes.add_argument("--questions", help="Batch input: JSON question array or conversations object")
    parser.add_argument("--output", help="New .json/.jsonl result file; default: timestamped file under results/")
    parser.add_argument("--model", help="OpenAI model ID; default: OPENAI_MODEL or gpt-4.1-mini")
    parser.add_argument("--env-file", help="Optional .env path; default: CHIQ/.env")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or f"results/{'live' if args.interactive else 'batch'}_{stamp}.json"
    try:
        conversations = None if args.interactive else load_conversations(args.questions)
        if args.questions and Path(args.questions).resolve() == Path(output).resolve():
            raise ValueError("--output must differ from --questions")
        if Path(output).exists():
            raise ValueError("Output already exists; choose a new --output filename.")
        generate = load_openai_generator(args.model, args.env_file)
        writer = ResultWriter(output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"결과 저장: {writer.path.resolve()}")
    try:
        if args.interactive:
            run_interactive(generate, writer)
        else:
            run_batch(conversations, generate, writer)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"처리 중단: {exc}\n완료된 턴은 {writer.path.resolve()}에 저장되어 있습니다.\n")
    except KeyboardInterrupt:
        parser.exit(130, f"\n중단되었습니다. 완료된 턴: {writer.path.resolve()}\n")
    print(f"저장 완료: {writer.path.resolve()}")


if __name__ == "__main__":
    main()
