import asyncio
import contextlib
import json
import logging
import math
import time
import signal
import sqlite3
from typing import Any, Iterable, List, Optional

import websockets

from config import Settings
from dedupe import Dedupe
from llm import LLMClient, LLMDirection, LLMFilterResult, LLMRiskPlan
from ibkr_executor import IBKRConfig, IBKRExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pipeline")


def _build_ws_url(base_url: str, api_key: str) -> str:
    if "token=" in base_url:
        return base_url
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}token={api_key}"


def _extract_headline(payload: dict[str, Any]) -> str:
    for key in ("headline", "title", "text"):
        if payload.get(key):
            return str(payload[key])
    return ""


def _extract_tickers(payload: dict[str, Any]) -> List[str]:
    return []


class MetricsLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._use_db = self.path.endswith(".db")

    async def log(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        await self._queue.put((kind, payload))

    async def start(self) -> None:
        if not self.enabled or self._task:
            return
        if self._use_db:
            await asyncio.to_thread(self._init_db)
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _worker(self) -> None:
        while True:
            kind, payload = await self._queue.get()
            entry = {
                "ts": time.time(),
                "kind": kind,
                "payload": payload,
            }
            if self._use_db:
                await asyncio.to_thread(self._append_db, entry)
            else:
                line = json.dumps(entry, ensure_ascii=False)
                await asyncio.to_thread(self._append_line, line)

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, payload TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    def _append_db(self, entry: dict[str, Any]) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
                (entry["ts"], entry["kind"], json.dumps(entry["payload"], ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def _append_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def size_position(
    equity: float,
    price: float,
    atr: float,
    confidence: float,
    magnitude: float,
    volatility_factor: float,
    atr_trail_mult: float,
    trail_extra_pct: float,
    max_risk_pct: float = 0.01,
    max_position_pct: float = 0.10,
) -> tuple[int, float, float]:
    """
    Equity/price/atr based position sizing with volatility + signal modulation.
    Returns (shares, position_pct, stop_distance).
    """
    # trail_extra_pct is percent (e.g., 0.2 means 0.2%)
    stop_distance = atr * atr_trail_mult + price * (trail_extra_pct / 100.0)
    if stop_distance <= 0:
        return 0, 0.0, stop_distance

    risk_budget = equity * max_risk_pct
    max_shares_by_risk = risk_budget / stop_distance

    signal_score = confidence * magnitude
    vol_penalty = 1.0 / max(volatility_factor, 0.5)
    effective_scale = signal_score * vol_penalty

    shares = math.floor(max_shares_by_risk * effective_scale)
    if shares <= 0:
        return 0, 0.0, stop_distance

    position_value = shares * price
    position_pct = position_value / equity
    if position_pct > max_position_pct:
        shares = math.floor((equity * max_position_pct) / price)
        position_value = shares * price
        position_pct = position_value / equity

    return shares, position_pct * 100.0, stop_distance


async def handle_news(
    deduper: Optional[Dedupe],
    llm: LLMClient,
    executor: IBKRExecutor,
    settings: Settings,
    item: dict[str, Any],
    metrics: MetricsLogger,
) -> None:
    headline = _extract_headline(item)
    if not headline:
        return
    tickers = _extract_tickers(item)
    body = ""
    if "body" in item and item["body"]:
        body = str(item["body"])
    full_text = headline if not body else f"{headline}\n\n{body}"

    if deduper:
        ok, reason = await deduper.accept(headline, tickers)
        if not ok:
            logger.info("DROPPED by dedupe reason=%s headline=%s", reason, headline)
            await metrics.log(
                "dropped_dedupe",
                {"reason": reason, "headline": headline, "tickers": tickers},
            )
            return

    # Fetch holdings for filter context
    try:
        holdings = await executor.get_positions_symbols()
    except Exception as exc:
        logger.warning("Failed to fetch holdings: %s", exc)
        holdings = []

    logger.info("NEWS payload headline=%s body_len=%s", headline, len(body))
    await metrics.log("news_in", {"headline": headline, "tickers": tickers, "body": bool(body)})

    logger.info("LLM FILTER input: %s", full_text[:500])
    filter_result: LLMFilterResult = await llm.filter_news(full_text, tickers, holdings)
    logger.info("FILTER result: %s", filter_result.model_dump())
    await metrics.log("llm_filter", {"result": filter_result.model_dump(), "input": full_text[:400]})
    if not filter_result.actionable:
        logger.info(
            "DROPPED by filter type=%s reason=%s ticker=%s headline=%s",
            filter_result.type,
            filter_result.reason,
            ",".join(tickers),
            headline,
        )
        await metrics.log(
            "dropped_filter",
            {
                "type": filter_result.type,
                "reason": filter_result.reason,
                "tickers": tickers,
                "headline": headline,
            },
        )
        return

    logger.info("LLM DIRECTION input: %s", full_text[:500])
    direction: LLMDirection = await llm.decide_direction(full_text, tickers, filter_result.type)
    logger.info("DIRECTION result: %s", direction.model_dump())
    await metrics.log("llm_direction", {"result": direction.model_dump(), "input": full_text[:400]})
    # Trust LLM-provided ticker guess.
    tickers = [t for t in (direction.tickers_out or []) if t and t.upper() not in {"UNK", "UNKNOWN", "N/A"}]
    if tickers and holdings and any(t in holdings for t in tickers):
        logger.info(
            "DROPPED in code-layer holding check ticker=%s holdings=%s headline=%s",
            ",".join(tickers),
            ",".join(holdings),
            headline,
        )
        await metrics.log("dropped_holdings", {"tickers": tickers, "holdings": holdings, "headline": headline})
        return
    # Prefetch cached ATRs as soon as tickers are known.
    cached_atr_tasks: dict[str, asyncio.Task] = {}
    for symbol in tickers:
        cached_atr_tasks[symbol] = asyncio.create_task(executor.get_cached_atr(symbol))
    logger.info("LLM RISK input: %s", full_text[:500])
    risk: LLMRiskPlan = await llm.risk_plan(
        full_text,
        tickers,
        filter_result.type,
        direction.action,
        direction.magnitude,
        direction.volatility_factor,
        direction.confidence,
        direction.risk_flags,
    )
    logger.info("RISK result: %s", risk.model_dump())
    await metrics.log("llm_risk", {"result": risk.model_dump(), "input": full_text[:400]})

    if direction.action in {"skip", "hold"}:
        logger.info(
            "SKIP action=%s type=%s sentiment=%s mag=%.2f conf=%.2f ticker=%s headline=%s",
            direction.action,
            filter_result.type,
            direction.sentiment,
            direction.magnitude,
            direction.confidence,
            ",".join(tickers),
            headline,
        )
        await metrics.log(
            "dropped_action",
            {
                "action": direction.action,
                "headline": headline,
                "tickers": tickers,
                "reason": "llm_skip_or_hold",
            },
        )
        return
    if direction.action == "sell":
        logger.info("SKIP sell action (shorting disabled) ticker=%s headline=%s", ",".join(tickers), headline)
        await metrics.log(
            "dropped_action",
            {"action": direction.action, "headline": headline, "tickers": tickers, "reason": "short_disabled"},
        )
        return

    if not risk.approve:
        logger.info(
            "RISK REJECT action=%s type=%s reason=%s ticker=%s headline=%s",
            direction.action,
            filter_result.type,
            risk.reason or "unknown",
            ",".join(tickers),
            headline,
        )
        await metrics.log(
            "dropped_risk",
            {
                "reason": risk.reason or "unknown",
                "headline": headline,
                "tickers": tickers,
                "action": direction.action,
                "type": filter_result.type,
            },
        )
        return

    if not tickers:
        logger.info("No ticker provided or inferred, skipping trade for headline=%s", headline)
        await metrics.log("dropped_noticker", {"headline": headline})
        return

    for symbol in tickers:
        try:
            cached_atr = None
            task = cached_atr_tasks.get(symbol)
            if task:
                try:
                    cached_atr = await task
                except Exception:
                    cached_atr = None
            price_atr = await executor.get_price_and_atr_cached(symbol, cached_atr)
            if not price_atr:
                logger.info("Missing price/ATR, skipping trade ticker=%s headline=%s", symbol, headline)
                await metrics.log(
                    "dropped_price_atr", {"headline": headline, "ticker": symbol, "reason": "missing_price_atr"}
                )
                continue
            last_price, atr_val = price_atr

            equity = await executor.get_equity()
            qty, pos_pct, trail_dist = size_position(
                equity=equity or executor.cfg.default_notional,
                price=last_price,
                atr=atr_val,
                confidence=direction.confidence,
                magnitude=direction.magnitude,
                volatility_factor=direction.volatility_factor,
                atr_trail_mult=risk.atr_trail_mult,
                trail_extra_pct=risk.trail_extra_pct,
                max_risk_pct=settings.max_risk_pct,
                max_position_pct=settings.max_position_pct,
            )

            if qty <= 0:
                equity_used = equity or executor.cfg.default_notional
                risk_budget = equity_used * settings.max_risk_pct
                signal_score = direction.confidence * direction.magnitude
                vol_penalty = 1.0 / max(direction.volatility_factor, 0.5)
                max_shares_by_risk = risk_budget / max(trail_dist, 1e-9)
                eff_shares = max_shares_by_risk * signal_score * vol_penalty
                logger.info(
                    "Qty is zero after sizing, skipping trade ticker=%s headline=%s "
                    "[equity_used=%.2f stop_dist=%.4f risk_budget=%.2f max_shares_by_risk=%.4f "
                    "signal_score=%.3f vol_penalty=%.3f eff_shares=%.4f]",
                    symbol,
                    headline,
                    equity_used,
                    trail_dist,
                    risk_budget,
                    max_shares_by_risk,
                    signal_score,
                    vol_penalty,
                    eff_shares,
                )
                await metrics.log(
                    "dropped_qty", {"headline": headline, "ticker": symbol, "reason": "qty_zero_after_sizing"}
                )
                continue
            if trail_dist <= 0:
                logger.info("Invalid trail distance, skipping trade ticker=%s headline=%s", symbol, headline)
                await metrics.log(
                    "dropped_trail", {"headline": headline, "ticker": symbol, "reason": "invalid_trail_distance"}
                )
                continue

            equity_used = equity or executor.cfg.default_notional
            logger.info(
                "SIZING ok ticker=%s equity=%.2f price=%.4f atr=%.4f stop_dist=%.4f qty=%s pos_pct=%.2f "
                "risk_budget=%.2f max_risk_pct=%.4f max_pos_pct=%.4f",
                symbol,
                equity_used,
                last_price,
                atr_val,
                trail_dist,
                qty,
                pos_pct,
                equity_used * settings.max_risk_pct,
                settings.max_risk_pct,
                settings.max_position_pct,
            )

            logger.info(
                "TRADE action=%s type=%s sentiment=%s mag=%.2f conf=%.2f "
                "pos=%.2f%% risk[atr_trail=%.2f trail_extra=%.2f%%] price=%.2f atr=%.2f qty=%s ticker=%s headline=%s",
                direction.action,
                filter_result.type,
                direction.sentiment,
                direction.magnitude,
                direction.confidence,
                pos_pct,
                risk.atr_trail_mult,
                risk.trail_extra_pct,
                last_price,
                atr_val,
                qty,
                symbol,
                headline,
            )
            logger.info(
                "TRADE candidate (no order placement) ticker=%s qty=%s stop_dist=%.4f price=%.4f atr=%.4f",
                symbol,
                qty,
                trail_dist,
                last_price,
                atr_val,
            )
            await metrics.log(
                "trade_candidate",
                {
                    "symbol": symbol,
                    "action": direction.action,
                    "qty": qty,
                    "last_price": last_price,
                    "atr": atr_val,
                    "stop_distance": trail_dist,
                    "pos_pct": pos_pct,
                    "risk": {
                        "atr_trail_mult": risk.atr_trail_mult,
                        "trail_extra_pct": risk.trail_extra_pct,
                    },
                    "headline": headline,
                },
            )
        except Exception as exc:
            logger.warning("Trade evaluation failed for %s: %s", symbol, exc)
            await metrics.log(
                "dropped_trade",
                {
                    "headline": headline,
                    "ticker": symbol,
                    "reason": f"trade_eval_exception: {exc}",
                },
            )


def parse_message(raw: str) -> Iterable[dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to decode message: %s", raw)
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []


async def consume_ws(settings: Settings, deduper: Optional[Dedupe], llm: LLMClient, executor: IBKRExecutor, metrics: MetricsLogger) -> None:
    url = _build_ws_url(settings.benzinga_ws_url, settings.benzinga_api_key)
    backoff = 1
    logger.info("Starting Benzinga WS: %s", settings.benzinga_ws_url)
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                logger.info("Connected to Benzinga WS")
                backoff = 1
                async for message in ws:
                    for item in parse_message(message):
                        await handle_news(deduper, llm, executor, settings, item, metrics)
        except Exception as exc:
            logger.warning("WS connection failed: %s; retrying in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def consume_tws_news(settings: Settings, deduper: Dedupe, llm: LLMClient, executor: IBKRExecutor, metrics: MetricsLogger) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    startup_ts = time.time()
    warmup_secs = 5.0  # drop ticks in the first few seconds if no timestamp present

    # Discover available providers using the main IB connection
    providers_cfg = [p.strip() for p in settings.tws_broad_providers.split(",") if p.strip()]
    try:
        await executor.ensure_connected()
        available = await executor.ib.reqNewsProvidersAsync()
        available_codes = {p.code for p in available}
        logger.info("IB available news providers: %s", ", ".join(sorted(available_codes)))
    except Exception as exc:
        logger.warning("Failed to fetch IB news providers: %s", exc)
        available_codes = set()

    if providers_cfg:
        providers = [p for p in providers_cfg if not available_codes or p in available_codes]
        missing = [p for p in providers_cfg if available_codes and p not in available_codes]
        if missing:
            logger.warning("Configured news providers not available: %s", ",".join(missing))
    else:
        providers = sorted(available_codes)

    if not providers:
        logger.warning("No TWS providers to subscribe; skipping IB news.")
        return

    # Subscribe on the main connection to avoid competing live session
    await executor.subscribe_broad_news(providers, queue, fetch_body=True)

    while True:
        news = await queue.get()
        # warmup drop if no timestamp present
        if "pub_ts" not in news and time.time() - startup_ts < warmup_secs:
            logger.info(
                "IB news drop (warmup) provider=%s id=%s headline=%s",
                news.get("provider"),
                news.get("articleId"),
                news.get("headline"),
            )
            continue
        if "pub_ts" in news and news["pub_ts"] < startup_ts:
            logger.info(
                "IB news drop (old_ts) provider=%s id=%s ts=%.0f startup=%.0f headline=%s",
                news.get("provider"),
                news.get("articleId"),
                news["pub_ts"],
                startup_ts,
                news.get("headline"),
            )
            continue

        item = {
            "headline": news.get("headline", ""),
            "tickers": [],
            "source": news.get("provider", "tws"),
            "extra": news.get("extra"),
            "articleId": news.get("articleId"),
            "body": news.get("body"),
        }
        await handle_news(deduper, llm, executor, settings, item, metrics)


async def consume_self_ws(settings: Settings, deduper: Dedupe, llm: LLMClient, executor: IBKRExecutor, metrics: MetricsLogger) -> None:
    url = settings.self_ws_url
    backoff = 1
    logger.info("Starting self WS: %s", url)
    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("Connected to self WS")
                backoff = 1
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if isinstance(data, dict):
                            await handle_news(deduper, llm, executor, settings, data, metrics)
                            continue
                    except json.JSONDecodeError:
                        pass
                    await handle_news(
                        deduper,
                        llm,
                        executor,
                        settings,
                        {"headline": str(message), "tickers": [], "source": "self"},
                        metrics,
                    )
        except Exception as exc:
            logger.warning("Self WS connection failed: %s; retrying in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def main() -> None:
    settings = Settings.from_env()
    if settings.accept_news_benzinga and not settings.benzinga_api_key:
        logger.error("BENZINGA_API_KEY is not set")
        return
    if not settings.llm_api_key:
        logger.error("LLM_API_KEY is not set")
        return
    if settings.dedupe_enabled and not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set (required for embeddings dedupe)")
        return

    llm = LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.openai_model,
        timeout=settings.llm_timeout,
        max_tokens=settings.llm_max_tokens,
        concurrency=settings.llm_concurrency,
        fallback_model=settings.openai_fallback_model,
    )
    metrics = MetricsLogger(settings.metrics_path, enabled=settings.metrics_enabled)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await metrics.start()

    executor = IBKRExecutor(
        IBKRConfig(
            host=settings.ib_host,
            port=settings.ib_port,
            client_id=settings.ib_client_id,
            account=settings.ib_account or None,
            default_notional=settings.ib_default_notional,
            atr_duration=settings.atr_duration,
            atr_bar_size=settings.atr_bar_size,
            atr_use_rth=settings.atr_use_rth,
        )
    )

    deduper: Optional[Dedupe] = None
    if settings.dedupe_enabled:
        deduper = Dedupe(
            api_key=settings.openai_api_key,
            ttl=settings.dedupe_ttl,
            sim_threshold=settings.dedupe_similarity_threshold,
        )

    tasks: list[asyncio.Task] = []
    logger.info(
        "News sources enabled: benzinga=%s tws=%s self_ws=%s providers=%s",
        settings.accept_news_benzinga,
        settings.accept_news_tws,
        settings.accept_news_selfws,
        settings.tws_broad_providers,
    )
    if settings.accept_news_benzinga:
        tasks.append(asyncio.create_task(consume_ws(settings, deduper, llm, executor, metrics)))
    if settings.accept_news_tws:
        tasks.append(asyncio.create_task(consume_tws_news(settings, deduper, llm, executor, metrics)))
    if settings.accept_news_selfws:
        tasks.append(asyncio.create_task(consume_self_ws(settings, deduper, llm, executor, metrics)))

    if not tasks:
        logger.error("No news sources enabled (set ACCEPT_NEWS_BENZINGA or ACCEPT_NEWS_TWS).")
        return

    await stop_event.wait()
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t
    await metrics.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
