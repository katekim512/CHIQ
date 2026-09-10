# CHIQ — 질문 전용 멀티턴

답변 없이 **이전 질문 목록 + 현재 질문**을 받아 TS → QD → Ad-hoc Query Rewriting을 실행합니다.
출력은 SQL 생성기에 전달할 독립적인 질문입니다. SQL 생성·실행 자체는 포함하지 않습니다.

## 남긴 모듈

- **TS (5종 분류)**: `relation`을 아래 다섯 값 중 하나로 직접 분류합니다.
  - `Constraint Refinement`: 같은 종류의 대상, 다른 조건
  - `Topic Exploration`: 같은 대상, 다른 속성
  - `Participant Shift`: 같은 속성, 다른 대상
  - `Answer Exploration`: 이전 답변의 일부 또는 특정 개체를 후속 탐색
  - `New Topic`: **이전 흐름 아님**
  첫 질문도 이전 흐름이 없으므로 `New Topic`으로 설정합니다(첫 턴 TS 호출은 생략).
  더 이상 `relation: null`을 사용하지 않습니다. 별도 이진 분류를 요청하지 않으며,
  기존 `topic` 필드는 호환성을 위해 5종 결과에서 자동으로 계산합니다:
  `New Topic` → `new_topic`, 나머지 네 관계 → `old_topic`.
  `New Topic`이면 이전 맥락을 제외하고 재작성하며, 완료 후 세션 맥락을 초기화합니다.
  답변 참조가 불명확하다는 이유만으로 `New Topic`을 선택하지 않습니다.
  이 경우 기존처럼 `Answer Exploration`과 `needs_clarification: true`를 반환합니다.
- **QD**: 이전 질문들을 사용해 지시어·생략을 복원합니다. 답변 필드는 받지 않습니다.
- **최종 재작성**: `{"query": ""}` JSON 형식으로 완전한 자연어 질문을 출력합니다.
  검색 키워드로 압축하라는 원본 지시는 제거했습니다. QD와 최종 재작성에 공통으로
  날짜·법인·라인·공정·지표·집계 방식·필터·제외 조건 등 유효한 조건을 유지하도록 지시합니다.
  명시적으로 변경·제거한 조건은 갱신하고, 현재 요청과 양립하지 않는 조건은 이어받지 않습니다.
  Participant Shift는 지정된 범위의 대상만 바꾸도록 지시합니다.
  최종 단계는 QD 결과를 원문 맥락과 대조해 누락·이전 조건 잔존을 확인하고,
  이미 완전하면 문구를 그대로 반환하도록 지시합니다. 이는 프롬프트 지침이며 의미 정확성의 보증은 아닙니다.

RE·PR·HS를 포함한 기존 두 실행 파일은 `generation/legacy/`에 **전체 주석 처리**하여 보존했습니다.
온라인 실행 경로에는 이들을 로드하거나 결과를 합치는 코드가 없습니다.
`generation/generate_pseudo_query_*.py` 역시 주석 처리했습니다.
별도 `dense/` 학습·검색 연구 코드는 수정하지 않았으며 이 흐름에서 호출하지 않습니다.
원래 README는 `README.original.md`에 보존했습니다.

## 입력·출력

`examples/questions.json`처럼 한 대화를 질문 문자열 배열로 입력합니다. 홀수/짝수 위치를 Q/A로 나누지 않습니다.
한 질문 배열은 하나의 대화입니다. 여러 대화는 아래의 `conversations` 형식으로 구분합니다. Q/A 객체나 빈 질문은 오류 처리하며, 문자열이 질문인지는 호출자가 보장해야 합니다.

출력의 각 턴에는 `conversation_id`, `history_questions`(해당 턴 처리 직전의 질문 맥락), `turn`, `question`, `topic`, `relation`, `needs_clarification`, `reason`,
`status`, `disambiguated_question`, `query`가 들어갑니다.

Answer Exploration도 분류하지만 “그중 첫 번째 제품”처럼 답변에만 있는 개체를 질문만으로 복원할 수는 없습니다.
TS가 이를 감지하면 `status: "needs_clarification"`, `query: null`로 반환하고 QD·재작성을 중단합니다.
사용자는 구체적인 제품명 등을 넣어 다시 질문하면 됩니다. 확인이 필요한 턴은 세션 히스토리에 넣지 않습니다.
반면 “통계학과 교수의 평균 급여는?”처럼 대상이 명시되어 있으면 답변 없이도 처리할 수 있습니다.
이 판정과 재작성의 의미 정확성은 사용하는 LLM에 의존하며, 사실성의 기계적 보증은 아닙니다.

## 실행

Python 3.9 이상과 OpenAI API 키가 필요합니다. 로컬 모델이나 PyTorch는 필요 없습니다.
기본 모델은 `gpt-4.1-mini`이며 `--model` 또는 `.env`의 `OPENAI_MODEL`로 변경할 수 있습니다.
기존 TS·QD·재작성 프롬프트는 API 연결 과정에서 변경하지 않았습니다.

1. CHIQ 폴더에서 환경을 준비합니다.

```sh
cd /Users/yeeunkim/Documents/CHIQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
```

2. **CHIQ 폴더의 `.env` 파일**을 열고 `OPENAI_API_KEY=` 뒤에 발급받은 키를 넣어 저장합니다.
   이 작업 환경에는 빈 `.env` 파일을 준비했습니다. 새로 복제한 저장소라면 `.env.example`을 `.env`로 복사하세요.

```dotenv
OPENAI_API_KEY=여기에_실제_API_키
OPENAI_MODEL=gpt-4.1-mini
```

키 발급: [OpenAI API 키 관리](https://platform.openai.com/api-keys).
실제 키는 채팅이나 Python 코드에 넣지 마세요. `.env`는 Git에서 제외했습니다.
기존 환경변수 `OPENAI_API_KEY`가 있으면 `.env`보다 우선합니다.
`--env-file /path/to/.env`로 다른 파일을 지정할 수도 있습니다.

`.env` 값에 따옴표는 필요 없습니다. `OPENAI_API_KEY=sk-...`처럼 저장하면 됩니다.

## 1. 실시간 질문 입력

설치가 끝난 환경에서는 아래 두 줄로 활성화한 후 실행합니다.

```sh
cd /Users/yeeunkim/Documents/CHIQ
source .venv/bin/activate
python generation/adhoc_query_rewriting.py --interactive
```

`질문>`이 나타나면 한 줄씩 질문을 입력합니다. 이전 질문을 이어서 해석하고 다음을 표시합니다.

- 입력한 질문
- 관계 분류
- `Queryrewriting`: 최종 질문 (`query`)

화면에는 최종 질문 하나만 표시합니다. 내부 처리와 저장 파일의 `disambiguated_question`·`query` 필드는 유지합니다.

`/history`는 현재 맥락의 이전 질문을 표시하고, `/reset`은 새 대화를 시작합니다.
대화 도중 `New Topic`으로 처리 완료되면 대화 번호를 1 올리고 턴을 1로 초기화합니다. 예: `[2 / 턴 2]` 다음 새 주제는 `[3 / 턴 1]`, 그 후속 질문은 `[3 / 턴 2]`입니다. 화면과 저장 JSON의 번호에 동일하게 반영하며, 최초 질문과 `/reset` 직후에는 번호를 중복 증가시키지 않습니다.
`/exit` 또는 Ctrl+D로 종료합니다. Ctrl+C로 중단해도 완료된 턴은 파일에 남습니다.
`needs_clarification`이면 이유를 표시하므로 대상을 명시해 다시 질문하세요.
API 오류가 발생하면 히스토리를 유지하고 재입력을 받습니다. 실패한 호출은 결과 턴으로 저장하지 않습니다.

결과는 기본적으로 `results/live_날짜_시각.json`에 각 턴마다 자동 저장합니다.
저장 위치를 지정하려면 `--output results/my_live.json`을 추가하세요.

## 2. JSON 파일 일괄 처리

[examples/questions.json](examples/questions.json)처럼 질문을 순서대로 넣습니다.

```json
[
  "C1라인 창원에어컨 불량률을 알려줘",
  "베트남에어컨도 알려줘",
  "지난달은?"
]
```

```sh
python generation/adhoc_query_rewriting.py --questions examples/questions.json
```

첫 질문, 첫 질문+두 번째 질문, 첫 질문+두 번째 질문+세 번째 질문 순서로 처리합니다.
미래 질문은 앞선 턴에 제공하지 않습니다. 새 주제 초기화와 확인 필요 턴 제외 규칙은 실시간 모드와 동일합니다.
각 턴 결과를 화면에 표시하고 **모든 턴을 하나의 JSON 배열**로 `results/batch_날짜_시각.json`에 저장합니다.

여러 독립 대화는 [examples/conversations.json](examples/conversations.json) 형식을 사용하세요.

```json
{
  "conversations": [
    {"id": "alarm", "questions": ["C1라인 알람 Top3", "C2라인은?"]},
    {"id": "defect", "questions": ["창원에어컨 불량률", "베트남에어컨은?"]}
  ]
}
```

```sh
python generation/adhoc_query_rewriting.py \
  --questions examples/conversations.json \
  --output results/my_batch.json
```

대화마다 히스토리를 초기화하고 `turn`은 1부터 다시 시작합니다. `conversation_id`로 결과를 구분합니다.
입력 파일 전체를 먼저 검증한 후 API를 호출합니다.

두 모드 모두 `--output`을 생략하면 매번 새로운 파일을 만듭니다.
`.json`은 들여쓰기한 배열, `.jsonl`은 한 줄에 한 턴으로 저장합니다.
기존 결과 파일은 덮어쓰지 않으므로 이미 존재하는 파일을 지정하면 새 이름으로 변경하세요.
일괄 처리 중 API 오류가 나면 중단하고 완료된 턴까지 보존합니다. 재실행은 처음부터 API 요청을 다시 수행합니다.

`generation/enhancement_generation.py`도 동일한 옵션으로 실행할 수 있습니다.
OpenAI Responses API를 사용하며 각 요청에 `store=False`를 전달합니다.
실행하면 질문 맥락이 OpenAI API로 전송됩니다. 키에 연결된 API 사용량/결제 설정이 필요합니다.
키 누락은 호출 전 오류로 알립니다. API 실패 시 키나 서버 오류 본문을 출력하지 않습니다.

Python에서 직접 사용하는 예시:

```python
from generation.adhoc_query_rewriting import QuestionOnlySession, load_openai_generator

generate = load_openai_generator()  # CHIQ/.env 자동 로드
session = QuestionOnlySession(generate)
first = session.ask("C1라인 창원에어컨 불량률을 알려줘")
second = session.ask("베트남에어컨도 알려줘")
print(second)
```

새 대화의 첫 질문은 TS·QD·최종 재작성 API를 모두 생략합니다. `query`와 `disambiguated_question`에 입력 질문을 그대로 저장하므로 첫 턴의 모델 호출은 0회입니다. 실시간 시작·`/reset` 이후 및 JSON의 각 독립 대화에 동일하게 적용합니다. 대화 도중 TS가 `New Topic`으로 판단하는 경우는 기존처럼 TS와 최종 재작성을 수행합니다. 후속 질문은 TS → QD → 재작성 순서로 호출하고,
무관한 새 주제는 이전 맥락을 제외해 재작성하며 세션 히스토리를 초기화합니다.
히스토리 요약이나 자동 절단은 하지 않으므로 모델의 컨텍스트 한도는 호출자가 관리해야 합니다.

## 검증

```sh
python -m unittest discover -s tests -v
```

외부 모델 없이 입력 검증, 다섯 관계의 분류·전달, 질문 누락 방지, 새 주제 초기화,
답변 참조 중단 및 잘못된 모델 출력 처리를 검증합니다.
모의 LLM 기반 흐름 테스트이므로 실제 모델의 한국어 분류·재작성 정확도 평가는 별도로 필요합니다.

API 연결은 실제 SDK와 모의 HTTP 응답으로 검사합니다. 유효한 키를 넣은 실제 API 호출은 별도로 필요합니다.

공식 문서: [Python API 시작하기](https://developers.openai.com/api/docs/quickstart), [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
