"""Jev decisions through OpenRouter's Decisions API, sharing the direct engine's contract."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

from ..errors import ProviderError
from ..schemas import RequestRecord
from .jev import JevEngine

if TYPE_CHECKING:
    import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/jerryjliu/docjev",
    "X-OpenRouter-Title": "DocJev",
}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class DecisionsResponse:
    id: str | None
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    answers: dict[str, ChoiceAnswer | NoulAnswer]

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}


class DecisionsRequestError(Exception):
    """Carries only the fields JevEngine reads to classify, retry, or redact a failure."""

    def __init__(self, status: int, *, body: str = "", retry_after_ms: float = 0):
        super().__init__(f"OpenRouter Decisions request failed ({status})")
        self.status = status
        self.body = body
        self.retry_after_ms = retry_after_ms
        self.request_id = None


class MalformedDecisionsResponse(Exception):
    """A 200 response that does not match the documented schema; never echoes its content."""

    def __init__(self, request_id: str | None = None):
        super().__init__("OpenRouter returned a malformed Decisions response")
        self.request_id = request_id


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unit(value: Any) -> float:
    if not _is_number(value) or not 0 <= value <= 1:
        raise ValueError
    return float(value)


def _optional_count(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError
    return value


def _optional_cost(value: Any) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or value < 0:
        raise ValueError
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError
    return value


def _parse_answer(answer: Any) -> ChoiceAnswer | NoulAnswer:
    if not isinstance(answer, dict):
        raise ValueError
    if answer.get("type") == "choice":
        choice = answer.get("choice")
        probabilities = answer.get("probabilities") or {}
        if not isinstance(choice, str) or not isinstance(probabilities, dict):
            raise ValueError
        confidence = answer.get("confidence")
        return ChoiceAnswer(
            choice=choice,
            probabilities={_str(k): _unit(v) for k, v in probabilities.items()},
            confidence=None if confidence is None else _unit(confidence),
        )
    if answer.get("type") == "noul":
        return NoulAnswer(noul=_unit(answer.get("noul")))
    raise ValueError


def _str(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError
    return value


def parse_decisions(payload: Any) -> DecisionsResponse:
    """Validate every field downstream records rely on; reject rather than coerce."""
    request_id = payload.get("id") if isinstance(payload, dict) else None
    try:
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise ValueError
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError
        return DecisionsResponse(
            id=_optional_str(payload.get("id")),
            model=_str(payload.get("model")),
            input_tokens=_optional_count(usage.get("input_tokens")),
            output_tokens=_optional_count(usage.get("output_tokens")),
            cost_usd=_optional_cost(usage.get("cost")),
            answers={_str(k): _parse_answer(v) for k, v in payload["answers"].items()},
        )
    except ValueError:
        raise MalformedDecisionsResponse(
            request_id if isinstance(request_id, str) else None
        ) from None


def retry_after_ms(value: str | None, *, now: float | None = None) -> float:
    """Parse delay-seconds or an HTTP-date; unusable values fall back to normal backoff."""
    if not value:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return 0
        if moment.tzinfo is None:
            return 0
        seconds = moment.timestamp() - (time.time() if now is None else now)
    if not math.isfinite(seconds):
        return 0
    return max(seconds, 0) * 1000


class OpenRouterJevEngine(JevEngine):
    name = "openrouter"
    # NOTE: 524 (edge timeout) and 529 (provider overloaded) are documented as transient,
    # and 413 is a payload-size rejection by definition — see the Decisions API reference:
    # https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request
    retryable_statuses = JevEngine.retryable_statuses | {524, 529}
    context_limit_statuses = frozenset({413})

    def __init__(
        self,
        model: str = "typesafe/jev-1.13",
        *,
        api_key: str | None = None,
        timeout: float = 30,
        max_retries: int = 1,
        window_size: int = 8,
        context_recovery: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ProviderError(
                "Install docjev[openrouter] to use Jev through OpenRouter."
            ) from None
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError(
                "Set OPENROUTER_API_KEY to use Jev through OpenRouter. "
                "Create a key at openrouter.ai/settings/keys."
            )
        self.model = model
        self.max_retries = max_retries
        self.window_size = window_size
        self.context_recovery = context_recovery
        self.headers = {"Authorization": f"Bearer {key}", **ATTRIBUTION_HEADERS}
        # NOTE: a caller-supplied client stays owned by the caller; aclose() leaves it open.
        self.owns_http = http_client is None
        self.http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self.owns_http:
            await self.http.aclose()

    async def _decide(self, state: dict, questions: dict) -> DecisionsResponse:
        import httpx

        body = {
            "model": self.model,
            "state": state,
            "questions": {key: question.model_dump() for key, question in questions.items()},
        }
        try:
            response = await self.http.post(DECISIONS_URL, json=body, headers=self.headers)
        except httpx.TimeoutException:
            raise TimeoutError("OpenRouter request timed out") from None
        except httpx.TransportError:
            raise ConnectionError("OpenRouter connection failed") from None
        if response.status_code != 200:
            raise DecisionsRequestError(
                response.status_code,
                body=response.text,
                retry_after_ms=retry_after_ms(response.headers.get("retry-after")),
            )
        try:
            payload = response.json()
        except ValueError:
            raise MalformedDecisionsResponse() from None
        return parse_decisions(payload)

    def _success_record(
        self, response: DecisionsResponse, task: str, attempt: int, elapsed_ms: float
    ) -> RequestRecord:
        return RequestRecord(
            provider=self.name,
            model=response.model,
            task=task,
            attempt=attempt,
            elapsed_ms=elapsed_ms,
            request_id=response.id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            cost_status="unknown" if response.cost_usd is None else "reported",
        )
