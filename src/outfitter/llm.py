"""LLM-provider interfaces owned by the standalone Outfitter package."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai import RateLimitError

from .provider_schemas import poe_tool_schemas


class RequestRateLimiter:
    """Evenly pace request starts according to an API-level RPM limit."""

    def __init__(
        self,
        requests_per_minute: int | None,
        *,
        clock=time.monotonic,
        sleeper=time.sleep,
    ):
        self.interval_seconds = (
            None
            if requests_per_minute is None
            else 60.0 / requests_per_minute + 0.1
        )
        self._clock = clock
        self._sleeper = sleeper
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self.interval_seconds is not None and self._last_request_at is not None:
            delay = self.interval_seconds - (now - self._last_request_at)
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
        self._last_request_at = now


@dataclass(slots=True)
class ModelResponse:
    assistant_message: dict[str, Any]
    raw: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class ModelProvider(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        temperature: float,
        max_output_tokens: int,
        timeout: float,
        parallel_tool_calls: bool,
        thinking_type: str | None,
        reasoning_effort: str | None,
        max_retries: int,
        request_rate_limit_per_minute: int | None,
        rate_limit_retry_seconds: float,
        provider_extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self._poe_tool_schema_envelope = (
            urlsplit(base_url or "").hostname == "api.poe.com"
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.parallel_tool_calls = parallel_tool_calls
        self.thinking_type = thinking_type
        self.reasoning_effort = reasoning_effort
        self.provider_extra_body = dict(provider_extra_body or {})
        conflicts = []
        if thinking_type is not None and "thinking" in self.provider_extra_body:
            conflicts.append("thinking")
        if (
            reasoning_effort is not None
            and "reasoning_effort" in self.provider_extra_body
        ):
            conflicts.append("reasoning_effort")
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise ValueError(
                "provider_extra_body conflicts with explicit provider settings: "
                f"{joined}"
            )
        self.max_retries = max_retries
        self.rate_limit_retry_seconds = rate_limit_retry_seconds
        self.rate_limiter = RequestRateLimiter(request_rate_limit_per_minute)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": (
                poe_tool_schemas(tools)
                if self._poe_tool_schema_envelope
                else tools
            ),
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        extra_body = dict(self.provider_extra_body)
        if self.thinking_type is not None:
            extra_body["thinking"] = {"type": self.thinking_type}
        if self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self._create_with_retries(kwargs)
        choice = response.choices[0]
        message = choice.message
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls or []
        ]
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            model_extra = getattr(message, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning = model_extra.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            assistant["reasoning_content"] = reasoning
        usage = response.usage
        return ModelResponse(
            assistant_message=assistant,
            raw=response.model_dump(mode="json"),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _create_with_retries(self, kwargs: dict[str, Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.client.chat.completions.create(**kwargs)
                choices = getattr(response, "choices", None)
                if choices and getattr(choices[0], "message", None) is not None:
                    return response
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "provider returned an empty or malformed chat completion "
                        f"after {attempt + 1} attempt(s)"
                    )
                time.sleep(self._transient_retry_delay(attempt))
            except RateLimitError as error:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self._rate_limit_retry_delay(error))
            except (APIConnectionError, APITimeoutError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(self._transient_retry_delay(attempt))
            except APIStatusError as error:
                retryable = error.status_code in {408, 409} or error.status_code >= 500
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(self._transient_retry_delay(attempt))
        raise RuntimeError("unreachable retry state")

    def _rate_limit_retry_delay(self, error: RateLimitError) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                parsed = float(headers.get("retry-after"))
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                return parsed
        return self.rate_limit_retry_seconds

    @staticmethod
    def _transient_retry_delay(attempt: int) -> float:
        return min(0.5 * (2**attempt), 8.0)


class ScriptedProvider:
    """Deterministic provider used by tests and local harness validation."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        if not self.responses:
            raise RuntimeError("scripted provider exhausted")
        self.requests.append({"messages": messages, "tools": tools})
        assistant = self.responses.pop(0)
        return ModelResponse(
            assistant_message=assistant,
            raw={"scripted": True, "message": assistant},
            input_tokens=max(1, len(json.dumps(messages)) // 4),
            output_tokens=max(1, len(json.dumps(assistant)) // 4),
        )
