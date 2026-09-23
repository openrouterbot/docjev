import json
from datetime import UTC, datetime
from email.utils import format_datetime

import httpx
import pytest

from jev_docs.classify import aclassify_document
from jev_docs.engines import jev as jev_module
from jev_docs.engines import make_engine
from jev_docs.engines.openrouter import OpenRouterJevEngine, retry_after_ms
from jev_docs.errors import ContextLimitError, ProviderError
from jev_docs.split import asplit_document

# The real body OpenRouter returned for a ~2 MB state during manual verification.
LIVE_OVERSIZE_BODY = {
    "error": {
        "message": 'HTTP 400: {"detail":{"error_type":"max_tokens_exceeded"}}',
        "code": 400,
    }
}


def decisions_payload(questions, *, cost=0.00021, split_at=frozenset({1})):
    answers = {}
    for key, question in questions.items():
        if question["type"] == "choice":
            answers[key] = {
                "type": "choice",
                "choice": "invoice",
                "confidence": 0.93,
                "probabilities": {
                    name: 0.97 if name == "invoice" else 0.03 / (len(question["criteria"]) - 1)
                    for name in question["criteria"]
                },
            }
        else:
            page = int(key.split("_")[1])
            answers[key] = {"type": "noul", "noul": 0.9 if page in split_at else 0.1}
    usage = {"input_tokens": 1200, "output_tokens": 4}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-dec-123",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": answers,
        "usage": usage,
    }


@pytest.fixture
async def engine_with():
    clients: list[httpx.AsyncClient] = []

    def build(handler, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return OpenRouterJevEngine(api_key="test-key", http_client=client, **kwargs)

    yield build
    for client in clients:
        await client.aclose()


@pytest.fixture
def no_sleep(monkeypatch):
    delays: list[float] = []

    async def record(delay):
        delays.append(delay)

    monkeypatch.setattr(jev_module.asyncio, "sleep", record)
    return delays


def answering(seen, **payload_kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        return httpx.Response(200, json=decisions_payload(body["questions"], **payload_kwargs))

    return handler


def failing_first(status, *, headers=None, body=None):
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(status, headers=headers, json=body or {"error": {"code": status}})
        payload = json.loads(request.content)
        return httpx.Response(200, json=decisions_payload(payload["questions"]))

    return handler, calls


def with_answer(update):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = decisions_payload(json.loads(request.content)["questions"])
        update(payload)
        return httpx.Response(200, json=payload)

    return handler


async def test_classify_sends_decisions_request_and_reports_billed_cost(
    document, rules, engine_with
):
    seen: list = []
    result = await aclassify_document(document, rules, engine=engine_with(answering(seen)))

    request, body = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["http-referer"] == "https://github.com/jerryjliu/docjev"
    assert request.headers["x-openrouter-title"] == "DocJev"
    assert body["model"] == "typesafe/jev-1.13"
    category = body["questions"]["category"]
    assert category["type"] == "choice"
    assert isinstance(category["instructions"], str) and category["instructions"]
    assert {"invoice", "purchase_order"} <= set(category["criteria"])
    assert body["state"]

    assert result.category == "invoice"
    assert result.engine == "openrouter"
    record = result.metrics.requests[0]
    assert record.provider == "openrouter"
    assert record.model == "typesafe/jev-1.13-20260917"
    assert record.request_id == "gen-dec-123"
    assert (record.input_tokens, record.output_tokens) == (1200, 4)
    assert record.cost_usd == pytest.approx(0.00021)
    assert record.cost_status == "reported"
    assert result.metrics.decision_cost_usd == pytest.approx(0.00021)


async def test_boundary_questions_omit_absent_criteria_instead_of_sending_null(
    document, rules, engine_with
):
    seen: list = []
    await asplit_document(document, rules, engine=engine_with(answering(seen)))

    boundary = seen[0][1]["questions"]["boundary_2"]
    assert boundary["type"] == "noul"
    assert isinstance(boundary["instructions"], str) and boundary["instructions"]
    assert "criteria" not in boundary or boundary["criteria"] is not None


async def test_missing_cost_is_unknown_rather_than_estimated(document, rules, engine_with):
    result = await aclassify_document(document, rules, engine=engine_with(answering([], cost=None)))
    assert result.metrics.requests[0].cost_usd is None
    assert result.metrics.requests[0].cost_status == "unknown"


async def test_split_maps_choice_and_noul_answers(document, rules, engine_with):
    engine = engine_with(answering([], split_at=frozenset({1, 3})))
    result = await asplit_document(document, rules, engine=engine)
    assert [segment.pages for segment in result.segments] == [[1, 2], [3, 4]]


async def test_rate_limit_is_retried_with_retry_after_seconds(
    document, rules, engine_with, no_sleep
):
    handler, calls = failing_first(429, headers={"retry-after": "2"})
    result = await aclassify_document(document, rules, engine=engine_with(handler))
    assert len(calls) == 2
    assert no_sleep == [2.0]
    assert [record.status for record in result.metrics.requests] == ["error", "ok"]
    assert result.metrics.requests[0].error_code == "429"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 524, 529])
async def test_transient_statuses_are_retried(document, rules, engine_with, no_sleep, status):
    handler, calls = failing_first(status)
    result = await aclassify_document(document, rules, engine=engine_with(handler))
    assert len(calls) == 2
    assert result.metrics.requests[0].error_code == str(status)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
async def test_terminal_statuses_fail_without_retry(document, rules, engine_with, no_sleep, status):
    handler, calls = failing_first(status)
    with pytest.raises(ProviderError) as caught:
        await aclassify_document(document, rules, engine=engine_with(handler))
    assert len(calls) == 1
    assert [record.error_code for record in caught.value.requests] == [str(status)]


async def test_retry_exhaustion_keeps_every_attempt(document, rules, engine_with, no_sleep):
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(529, json={"error": {"code": 529}})

    with pytest.raises(ProviderError) as caught:
        await aclassify_document(document, rules, engine=engine_with(handler, max_retries=2))
    assert len(calls) == 3
    assert [record.error_code for record in caught.value.requests] == ["529"] * 3


async def test_retry_after_beyond_cap_fails_instead_of_retrying_early(
    document, rules, engine_with, no_sleep
):
    handler, calls = failing_first(429, headers={"retry-after": "30"})
    with pytest.raises(ProviderError):
        await aclassify_document(document, rules, engine=engine_with(handler))
    assert len(calls) == 1
    assert no_sleep == []


def test_retry_after_accepts_seconds_and_http_dates():
    now = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC).timestamp()
    future = format_datetime(datetime(2026, 9, 23, 12, 0, 5, tzinfo=UTC), usegmt=True)
    past = format_datetime(datetime(2026, 9, 23, 11, 59, 0, tzinfo=UTC), usegmt=True)
    assert retry_after_ms("2") == 2000
    assert retry_after_ms(future, now=now) == pytest.approx(5000)
    assert retry_after_ms(past, now=now) == 0
    assert retry_after_ms("nan") == 0
    assert retry_after_ms("inf") == 0
    assert retry_after_ms("soon") == 0
    assert retry_after_ms(None) == 0


async def test_http_date_retry_after_sets_the_delay(document, rules, engine_with, no_sleep):
    later = datetime.now(UTC).timestamp() + 4
    header = format_datetime(datetime.fromtimestamp(later, UTC), usegmt=True)
    handler, calls = failing_first(429, headers={"retry-after": header})
    await aclassify_document(document, rules, engine=engine_with(handler))
    assert len(calls) == 2
    assert 2 < no_sleep[0] <= 4


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (413, {"error": {"code": 413, "message": "Payload exceeds limit"}}),
        (400, LIVE_OVERSIZE_BODY),
    ],
)
async def test_oversized_input_raises_context_error_without_truncating(
    document, rules, engine_with, status, body
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    with pytest.raises(ContextLimitError):
        await aclassify_document(document, rules, engine=engine_with(handler))


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (413, {"error": {"code": 413, "message": "Payload exceeds limit"}}),
        (400, LIVE_OVERSIZE_BODY),
    ],
)
async def test_split_recovers_from_size_rejection_with_smaller_windows(
    document, rules, engine_with, status, body
):
    targets: list[list[int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        pages = sorted(
            int(key.split("_")[1]) for key in payload["questions"] if key.startswith("category_")
        )
        targets.append(pages)
        if len(pages) > 2:
            return httpx.Response(status, json=body)
        return httpx.Response(
            200, json=decisions_payload(payload["questions"], split_at=frozenset({1, 3}))
        )

    result = await asplit_document(document, rules, engine=engine_with(handler))

    assert targets[0] == [1, 2, 3, 4]
    assert sorted(page for window in targets[1:] for page in window) == [1, 2, 3, 4]
    assert [segment.pages for segment in result.segments] == [[1, 2], [3, 4]]
    records = result.metrics.requests
    assert records[0].status == "error" and records[0].error_code == str(status)
    assert [record.status for record in records[1:]] == ["ok"] * (len(targets) - 1)


async def test_auth_failure_does_not_leak_provider_body(document, rules, engine_with):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 401, "message": "secret-detail"}})

    with pytest.raises(ProviderError) as caught:
        await aclassify_document(document, rules, engine=engine_with(handler))
    assert "401" in str(caught.value)
    assert "secret-detail" not in str(caught.value)


async def test_connection_errors_are_retryable(document, rules, engine_with, no_sleep):
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=request)
        body = json.loads(request.content)
        return httpx.Response(200, json=decisions_payload(body["questions"]))

    result = await aclassify_document(document, rules, engine=engine_with(handler))
    assert result.metrics.requests[0].error_code == "ConnectionError"


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {"category": {"type": "score", "score": 3}}, "model": "m", "usage": {}},
        {"model": "m", "usage": {}},
        ["not", "an", "object"],
    ],
)
async def test_unreadable_responses_fail_as_provider_errors(document, rules, engine_with, payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ProviderError):
        await aclassify_document(document, rules, engine=engine_with(handler))


@pytest.mark.parametrize(
    "update",
    [
        lambda p: p["usage"].update(cost="secret-detail"),
        lambda p: p["usage"].update(input_tokens="secret-detail"),
        lambda p: p["usage"].update(output_tokens=-1),
        lambda p: p.update(model={"secret-detail": 1}),
        lambda p: p.update(id=["secret-detail"]),
        lambda p: p["answers"]["category"].update(confidence="secret-detail"),
        lambda p: p["answers"]["category"].update(confidence=1.5),
        lambda p: p["answers"]["category"]["probabilities"].update(invoice="secret-detail"),
    ],
)
async def test_malformed_fields_are_redacted_and_recorded(document, rules, engine_with, update):
    with pytest.raises(ProviderError) as caught:
        await aclassify_document(document, rules, engine=engine_with(with_answer(update)))
    assert "secret-detail" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert [record.error_code for record in caught.value.requests] == ["MalformedDecisionsResponse"]


@pytest.mark.parametrize(
    "update",
    [
        lambda p: p["answers"]["category"].pop("probabilities"),
        lambda p: p["answers"]["category"].update(probabilities={}),
        lambda p: p["answers"]["category"].update(probabilities={"invoice": 0.97}),
    ],
)
async def test_classify_rejects_missing_or_partial_probabilities(
    document, rules, engine_with, update
):
    with pytest.raises(ProviderError, match="invalid category decision") as caught:
        await aclassify_document(document, rules, engine=engine_with(with_answer(update)))
    assert [record.status for record in caught.value.requests] == ["ok"]


@pytest.mark.parametrize(
    "update",
    [
        lambda p: p["answers"]["category_2"].pop("probabilities"),
        lambda p: p["answers"]["category_2"].update(probabilities={"purchase_order": 0.5}),
    ],
)
async def test_split_rejects_answers_without_the_chosen_probability(
    document, rules, engine_with, update
):
    with pytest.raises(ProviderError, match="invalid page decisions"):
        await asplit_document(document, rules, engine=engine_with(with_answer(update)))


async def test_caller_owned_client_stays_open_and_owned_client_closes():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        await OpenRouterJevEngine(api_key="k", http_client=client).aclose()
        assert not client.is_closed
    finally:
        await client.aclose()

    owned = OpenRouterJevEngine(api_key="k")
    await owned.aclose()
    assert owned.http.is_closed


def test_missing_key_explains_which_variable_to_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenRouterJevEngine()


async def test_factory_selects_openrouter_with_pinned_default_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    pinned = make_engine("openrouter")
    latest = make_engine("openrouter", "~typesafe/jev-latest")
    try:
        assert isinstance(pinned, OpenRouterJevEngine)
        assert pinned.model == "typesafe/jev-1.13"
        assert latest.model == "~typesafe/jev-latest"
    finally:
        await pinned.aclose()
        await latest.aclose()
