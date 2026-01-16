import asyncio
import json
import logging
import time
from typing import Iterable, Optional, Type, TypeVar, List

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _enforce_strict_schema(schema: object) -> None:
    if isinstance(schema, dict):
        schema_type = schema.get("type")
        if schema_type == "object":
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False
            props = schema.get("properties")
            if isinstance(props, dict):
                # Strict json_schema mode requires every property to be listed in required.
                schema["required"] = sorted(props.keys())
        for value in schema.values():
            _enforce_strict_schema(value)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_strict_schema(item)


def _is_json_validate_failed(exc: Exception) -> bool:
    msg = str(exc)
    return "json_validate_failed" in msg or "Failed to validate JSON" in msg


def parse_jsonl(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])


class LLMFilterResult(BaseModel):
    type: str = Field(description="event type label, e.g. earnings/macro/garbage")
    actionable: bool = Field(description="false if macro/analyst/garbage")
    reason: str = ""
    risk_flags: list[str] = Field(default_factory=list)


class LLMDirection(BaseModel):
    sentiment: str = Field(description="bullish, bearish, or neutral")
    action: str = Field(description="buy/sell/hold/skip")
    magnitude: float = Field(description="0-1 magnitude bucket")
    confidence: float = Field(description="0-1 confidence score")
    volatility_factor: float = Field(default=1.0, description=">=0.5 low vol, ~1 normal, >1.5 high vol")
    tickers_out: List[str] = Field(default_factory=list, description="best single ticker guess (or echo input)")
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("sentiment")
    @classmethod
    def _sentiment(cls, v: str) -> str:
        v = v.lower()
        if v not in {"bullish", "bearish", "neutral"}:
            return "neutral"
        return v

    @field_validator("action")
    @classmethod
    def _action(cls, v: str) -> str:
        v = v.lower()
        if v not in {"buy", "sell", "hold", "skip"}:
            return "skip"
        return v

    @field_validator("magnitude", "confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("volatility_factor")
    @classmethod
    def _vol(cls, v: float) -> float:
        try:
            val = float(v)
        except Exception:
            return 1.0
        return max(0.1, min(5.0, val))

    @field_validator("tickers_out")
    @classmethod
    def _tickers(cls, v: List[str]) -> List[str]:
        out: List[str] = []
        for t in v:
            if not t:
                continue
            sym = str(t).upper().strip()
            if sym in {"UNK", "UNKNOWN", "N/A"}:
                continue
            out.append(sym)
        return out


class LLMRiskPlan(BaseModel):
    approve: bool = False
    atr_trail_mult: float = Field(default=0.0, description="ATR multiple for trailing stop distance")
    trail_extra_pct: float = Field(default=0.0, description="extra percent buffer for trailing distance")
    reason: str = ""
    risk_flags: list[str] = Field(default_factory=list)


FILTER_PROMPT = (
    '''

    '''
)

DIRECTION_PROMPT = (
 '''

 '''
)




RISK_PROMPT = (
'''


'''
)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        timeout: float = 10,
        max_tokens: int = 80,
        concurrency: int = 8,
        fallback_model: Optional[str] = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(concurrency)
        self.fallback_model = fallback_model

    async def _call(
        self,
        system_prompt: str,
        user_content: str,
        model_cls: Type[T],
        fallback: T,
        _using_fallback: bool = False,
        _model_override: Optional[str] = None,
    ) -> T:
        model_to_use = _model_override or self.model
        try:
            async with self._sem:
                t0 = time.perf_counter()
                sp = system_prompt.replace("\n", " ")
                uc = user_content.replace("\n", " ")
                logger.info(
                    "LLM request model=%s system_prompt=%s user_content=%s",
                    model_to_use,
                    sp[:500] + ("..." if len(sp) > 500 else ""),
                    uc[:500] + ("..." if len(uc) > 500 else ""),
                )
                schema = model_cls.model_json_schema()
                _enforce_strict_schema(schema)
                try:
                    resp = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=model_to_use,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": model_cls.__name__,
                                    "strict": True,
                                    "schema": schema,
                                },
                            },
                            max_completion_tokens=self.max_tokens,
                            temperature=0,
                        ),
                        timeout=self.timeout,
                    )
                    raw = resp.choices[0].message.content or "{}"
                except BadRequestError as exc:
                    if not _is_json_validate_failed(exc):
                        raise
                    logger.warning("LLM strict json_schema failed; retrying with json_object mode")
                    try:
                        resp = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=model_to_use,
                                messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            response_format={"type": "json_object"},
                            max_completion_tokens=self.max_tokens,
                            temperature=0,
                        ),
                        timeout=self.timeout,
                    )
                        raw = resp.choices[0].message.content or "{}"
                    except BadRequestError as exc2:
                        if not _is_json_validate_failed(exc2):
                            raise
                        logger.warning("LLM json_object failed; retrying without response_format")
                        resp = await asyncio.wait_for(
                            self.client.chat.completions.create(
                                model=model_to_use,
                                messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            max_completion_tokens=self.max_tokens,
                            temperature=0,
                        ),
                        timeout=self.timeout,
                    )
                        raw = resp.choices[0].message.content or "{}"
                latency = (time.perf_counter() - t0) * 1000
            logger.info("LLM %s raw: %s", model_cls.__name__, (raw[:400] + ("..." if len(raw) > 400 else "")))
            logger.info("LLM %s latency_ms=%.1f", model_cls.__name__, latency)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    data = parse_jsonl(raw)
                except Exception as exc:
                    logger.warning("LLM JSON decode failed: %s raw=%r", exc, raw[:500])
                    return fallback
            return model_cls.model_validate(data)
        except (asyncio.TimeoutError, Exception, ValidationError) as exc:
            logger.warning("LLM call failed (%s): %r", type(exc).__name__, exc)
            if self.fallback_model and not _using_fallback:
                logger.info("Retrying with fallback model %s", self.fallback_model)
                return await self._call(
                    system_prompt,
                    user_content,
                    model_cls,
                    fallback,
                    _using_fallback=True,
                    _model_override=self.fallback_model,
                )
            return fallback

    async def filter_news(self, headline: str, tickers: Iterable[str], holdings: Iterable[str]) -> LLMFilterResult:
        user_content = (
            f"Ticker(s): {','.join(tickers) or 'UNK'}\n"
            f"Holdings: {','.join(holdings) or 'NONE'}\n"
            f"Headline: {headline}"
        )
        fallback = LLMFilterResult(
            type="unknown",
            actionable=False,
            reason="fallback",
            risk_flags=["fallback"],
        )
        return await self._call(FILTER_PROMPT, user_content, LLMFilterResult, fallback)

    async def decide_direction(
        self,
        headline: str,
        tickers: Iterable[str],
        event_type: str,
    ) -> LLMDirection:
        user_content = (
            f"Type: {event_type}\nTicker(s): {','.join(tickers) or 'UNK'}\nHeadline: {headline}"
        )
        fallback = LLMDirection(
            sentiment="neutral",
            action="skip",
            magnitude=0.0,
            confidence=0.0,
            risk_flags=["fallback"],
        )
        return await self._call(DIRECTION_PROMPT, user_content, LLMDirection, fallback)

    async def risk_plan(
        self,
        headline: str,
        tickers: Iterable[str],
        event_type: str,
        action_hint: Optional[str] = None,
        magnitude_hint: Optional[float] = None,
        volatility_factor: Optional[float] = None,
        confidence_hint: Optional[float] = None,
        risk_flags: Optional[Iterable[str]] = None,
    ) -> LLMRiskPlan:
        user_content = (
            f"Type: {event_type}\nAction: {action_hint or 'unknown'}\n"
            f"Magnitude: {magnitude_hint if magnitude_hint is not None else 'unknown'}\n"
            f"Confidence: {confidence_hint if confidence_hint is not None else 'unknown'}\n"
            f"Volatility_factor: {volatility_factor if volatility_factor is not None else '1.0'}\n"
            f"Risk_flags: {','.join(risk_flags or []) or 'NONE'}\n"
            f"Ticker(s): {','.join(tickers) or 'UNK'}\nHeadline: {headline}"
        )
        fallback = LLMRiskPlan(
            approve=False,
            atr_trail_mult=0.0,
            trail_extra_pct=0.0,
            reason="fallback",
            risk_flags=["fallback"],
        )
        return await self._call(RISK_PROMPT, user_content, LLMRiskPlan, fallback)
