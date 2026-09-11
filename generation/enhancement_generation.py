"""Question-only TS and QD. Original implementation: legacy/enhancement_generation.py."""
import json
import re


NEW_TOPIC = "new_topic"
OLD_TOPIC = "old_topic"

# QD keeps its original base instruction; final rewriting now preserves full questions.
# RE (explain_response), PR (single_response/multi_response), and HS
# (summarize_context) are disabled; their original prompts remain in legacy/.
PROMPT_DICT = {
    "explain_question": """
        You are given a set of previous questions and a new question that is ambiguous. Your goal is to re-write the
        question so it became clear. Just write the new question without any introduction.
    """,
    "search_query": """
        Given a series of previous questions as context, along with a new question, your task is to rewrite
        the new question as a complete, standalone natural-language question that preserves all applicable
        query conditions for downstream SQL generation. Do not shorten, summarize, or convert it into search keywords.
        Check the disambiguated question against the supplied question history and current question for missing
        or superseded conditions. If it is already complete and correct, keep its wording unchanged.
        The output should be placed in a JSON dictionary as follow: {"query": ""}
    """,
    "new_topic": """
        Given a series of previous questions, along with a new question, your task is to determine whether
        the new question continues the discussion on an existing topic or introduces a new topic.
        Classify it as either "old_topic" or "new_topic".
    """,
}
PROMPT_DICT = {k: re.sub(r"\s+", " ", v).strip() + "\n\n" for k, v in PROMPT_DICT.items()}

GROUNDING = (
    "Only use information in the supplied questions. No answers are available. "
    "Do not invent entities, values, results, schema names, or SQL. "
    "Preserve the current question's language and explicit changes; replace superseded constraints or entities."
)

PRESERVE_CONTEXT = (
    "Restore omitted information using the full supplied question history, not only the latest question. "
    "For a continuing question, carry forward every still-applicable condition unless it is explicitly "
    "changed or removed: date/time range, company, factory, line, process, entity, metric, aggregation "
    "(such as mean or quartile), grouping, ranking, units, filters, and exclusions. "
    "Silence about a condition does not remove it. Use the latest applicable value for each condition; "
    "do not reintroduce superseded values or carry a condition that is incompatible with the current request. "
    "When the question changes an entity, change only that entity at its stated scope (for example, the process), "
    "while preserving the other applicable conditions (such as date, line, metric, and aggregation). "
    "Keep identifiers and numeric values as supplied. Output a complete natural-language question, "
    "including the request wording; do not compress it into keywords or omit details for brevity.\n\n"
)


def validate_questions(history, question):
    if not isinstance(history, list) or any(not isinstance(q, str) or not q.strip() for q in history):
        raise ValueError("history must be a list of non-empty question strings (no Q&A pairs).")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")


def context(history, question):
    validate_questions(history, question)
    return "### Context\n" + json.dumps(history, ensure_ascii=False) + "\n\n### Question\n" + question


def parse_json(text):
    """Accept a JSON object, optionally enclosed in a single Markdown fence."""
    if not isinstance(text, str):
        raise ValueError("Model output must be text.")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Model must return a JSON object.")
    return value


def classify_relation(history, question, generate):
    validate_questions(history, question)
    if not history:
        return {"topic": "new_topic", "relation": NEW_TOPIC, "needs_clarification": False, "reason": "First question; no previous flow."}
    prompt = PROMPT_DICT["new_topic"] + GROUNDING + "\n" + (
        'Return only JSON: {"topic": "old_topic" or "new_topic", '
        '"needs_clarification": true or false, "reason": "short explanation"}. '
        'Use "old_topic" when the question continues or modifies the existing information need, '
        'including changes to conditions, properties, or entities, or references to previous results. '
        'Use "new_topic" for an unrelated information need. '
        "Use the full question history to resolve ellipsis. "
        "A reference that cannot be resolved does not by itself indicate a new topic. "
        "Set needs_clarification to true if a reference cannot be resolved from questions alone "
        "(for example 'the first one' in an unseen answer), or if the intended antecedent is ambiguous. "
        "An explicitly named entity can be resolved without an answer.\n\n"
    ) + context(history, question)
    result = parse_json(generate(prompt))
    if not {"topic", "needs_clarification", "reason"}.issubset(result):
        raise ValueError("TS output is missing required fields.")
    if result["topic"] not in (OLD_TOPIC, NEW_TOPIC):
        raise ValueError("TS must return old_topic or new_topic.")
    if type(result.get("needs_clarification")) is not bool or not isinstance(result.get("reason"), str):
        raise ValueError("TS must provide needs_clarification (boolean) and reason (string).")
    # Compatibility field only, not a second classification.
    return {"topic": result["topic"], "relation": result["topic"],
            "needs_clarification": result["needs_clarification"], "reason": result["reason"]}


def disambiguate(history, question, generate):
    prompt = PROMPT_DICT["explain_question"] + GROUNDING + "\n\n" + PRESERVE_CONTEXT
    prompt += context(history, question).replace("### Question\n", "### Ambiguous Question\n", 1)
    result = generate(prompt)
    if not isinstance(result, str) or not result.strip():
        raise ValueError("QD returned an empty question.")
    return result.strip()


if __name__ == "__main__":
    # Both historical entry points now execute the same TS -> QD -> rewrite flow.
    from adhoc_query_rewriting import main
    main()
