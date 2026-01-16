import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

from ib_insync import IB, Contract, Stock, NewsTick


logger = logging.getLogger(__name__)


@dataclass
class IBKRConfig:
    host: str
    port: int
    client_id: int
    account: Optional[str]
    default_notional: float
    atr_duration: str
    atr_bar_size: str
    atr_use_rth: bool


class IBKRExecutor:
    """
    Thin wrapper around ib_insync for market data, positions, and news.
    Uses blocking ib_insync calls inside threads to avoid stalling asyncio loop.
    """

    def __init__(self, config: IBKRConfig) -> None:
        self.cfg = config
        self.ib = IB()
        self.ib.RequestTimeout = 30  # seconds for sync requests
        self._lock = asyncio.Lock()
        self._hb_task: Optional[asyncio.Task] = None
        self._hb_interval = 15
        self._news_providers: list[str] = []
        self.ib.errorEvent += self._on_error
        self._seen_articles: dict[str, float] = {}
        self._seen_store = Path("news_seen_ids.jsonl")
        self._atr_db = None
        try:
            from atrdb.atr_db import ATRDatabase
            self._atr_db = ATRDatabase()
        except Exception as exc:
            logger.warning("ATR cache disabled: %s", exc)

    async def ensure_connected(self) -> None:
        async with self._lock:
            if self.ib.isConnected():
                return
            await self.ib.connectAsync(
                self.cfg.host,
                self.cfg.port,
                clientId=self.cfg.client_id,
                readonly=False,
                account=self.cfg.account,
            )
            logger.info("Connected to IBKR TWS/Gateway %s:%s", self.cfg.host, self.cfg.port)
            # Use delayed market data if real-time subscriptions are unavailable.
            try:
                self.ib.reqMarketDataType(3)
            except Exception as exc:
                logger.warning("Failed to set delayed market data type: %s", exc)
            # start heartbeat
        if self._hb_task is None:
            self._hb_task = asyncio.create_task(self._heartbeat())
        # load seen news ids on first connect
        if not self._seen_articles:
            self._load_seen_articles()
        # if we had prior news providers, resubscribe after reconnect
        if self._news_providers:
            try:
                await self.subscribe_broad_news(self._news_providers, asyncio.Queue(), fetch_body=False)
                logger.info("Re-subscribed news providers after reconnect: %s", self._news_providers)
            except Exception as exc:
                logger.warning("Failed to resubscribe news after reconnect: %s", exc)

    async def _heartbeat(self) -> None:
        backoff = 1
        while True:
            await asyncio.sleep(self._hb_interval)
            try:
                if not self.ib.isConnected():
                    raise RuntimeError("IB disconnected")
                await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), timeout=5)
                backoff = 1
            except Exception as exc:
                logger.warning("IB heartbeat failed: %s; reconnecting in %ss", exc, backoff)
                with contextlib.suppress(Exception):
                    self.ib.disconnect()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                try:
                    await self.ensure_connected()
                except Exception as exc2:
                    logger.warning("IB reconnect attempt failed: %s", exc2)

    def _load_seen_articles(self) -> None:
        if not self._seen_store.exists():
            return
        try:
            with self._seen_store.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._seen_articles[line] = time.time()
            logger.info("Loaded %s seen news ids from %s", len(self._seen_articles), self._seen_store)
        except Exception as exc:
            logger.warning("Failed to load seen news ids: %s", exc)

    def _persist_seen_article(self, article_id: str) -> None:
        try:
            with self._seen_store.open("a", encoding="utf-8") as f:
                f.write(article_id + "\n")
        except Exception as exc:
            logger.warning("Failed to persist seen news id: %s", exc)

    async def _qualify(self, symbol: str) -> Contract:
        await self.ensure_connected()
        contract = Stock(symbol, "SMART", "USD")
        qualified = await self.ib.qualifyContractsAsync(contract)
        return qualified[0]

    async def get_price_and_atr(
        self,
        symbol: str,
        atr_period: int = 14,
    ) -> Optional[Tuple[float, float]]:
        """
        Fetch last price (market data) and compute daily ATR (EMA). Falls back to 2% if insufficient data.
        """
        contract = await self._qualify(symbol)
        # daily bars for ATR and close price
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=self.cfg.atr_duration,
            barSizeSetting=self.cfg.atr_bar_size,
            whatToShow="TRADES",
            useRTH=self.cfg.atr_use_rth,
            formatDate=1,
        )
        if not bars or len(bars) < atr_period + 1:
            logger.warning("%s: not enough bars for ATR (got %s), using default 2%%", symbol, len(bars) if bars else 0)
            last_price = bars[-1].close if bars else None
            if not last_price or last_price <= 0:
                return None
            default_atr = last_price * 0.02
            return last_price, default_atr

        last_price = bars[-1].close
        if not last_price or last_price <= 0:
            logger.error("%s: cannot get valid last price from bars", symbol)
            return None

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        trs = []
        for i in range(1, len(bars)):
            prev_close = closes[i - 1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
            trs.append(tr)
        if len(trs) < atr_period:
            logger.warning("%s: TR data insufficient (trs=%s), using default 2%%", symbol, len(trs))
            default_atr = last_price * 0.02
            return last_price, default_atr

        # EMA ATR
        atr = trs[-atr_period]
        alpha = 2 / (atr_period + 1)
        for tr in trs[-atr_period + 1:]:
            atr = alpha * tr + (1 - alpha) * atr

        atr_pct = atr / last_price
        if atr_pct < 0.005:
            logger.warning("%s: ATR too small %.2f%%, using default 2%%", symbol, atr_pct * 100)
            atr = last_price * 0.02
        elif atr_pct > 0.15:
            logger.warning("%s: ATR too large %.2f%%, capping at 10%%", symbol, atr_pct * 100)
            atr = last_price * 0.10

        logger.info("%s price=%.2f atr=%.2f (%.2f%%) bars=%s", symbol, last_price, atr, (atr/last_price)*100, len(bars))
        return last_price, atr

    async def get_cached_atr(self, symbol: str) -> Optional[float]:
        if not self._atr_db:
            return None
        try:
            rec = await asyncio.to_thread(self._atr_db.get, symbol.upper())
        except Exception as exc:
            logger.warning("ATR cache read failed for %s: %s", symbol, exc)
            return None
        if not rec:
            return None
        try:
            atr = float(rec.get("atr"))
        except Exception:
            return None
        return atr if atr > 0 else None

    async def get_price_and_atr_cached(
        self,
        symbol: str,
        cached_atr: Optional[float],
        atr_period: int = 14,
    ) -> Optional[Tuple[float, float]]:
        if cached_atr and cached_atr > 0:
            last_price = await self.get_last_price_fast(symbol)
            if last_price:
                return last_price, cached_atr
        return await self.get_price_and_atr(symbol, atr_period=atr_period)

    async def get_last_price_fast(self, symbol: str) -> Optional[float]:
        """
        Fetch a fast snapshot price via market data (last/close/marketPrice).
        """
        await self.ensure_connected()
        contract = await self._qualify(symbol)
        try:
            tickers = await asyncio.wait_for(self.ib.reqTickersAsync(contract), timeout=5)
        except Exception as exc:
            logger.warning("Snapshot price failed for %s: %s", symbol, exc)
            return None
        if not tickers:
            return None
        ticker = tickers[0]
        price = ticker.marketPrice()
        if not price or price <= 0:
            for candidate in (ticker.last, ticker.close, ticker.bid, ticker.ask):
                if candidate and candidate > 0:
                    price = candidate
                    break
        return price if price and price > 0 else None

    def calc_qty(self, price: float, position_pct: float) -> int:
        notional = max(0.0, position_pct / 100.0) * self.cfg.default_notional
        if notional <= 0:
            return 0
        qty = int(notional // price)
        return max(qty, 0)

    async def get_equity(self) -> Optional[float]:
        """
        Fetch NetLiquidation (account equity) if available; fallback to default_notional.
        """
        await self.ensure_connected()
        try:
            summary = await self.ib.accountSummaryAsync()
            for tag in summary:
                if tag.tag in ("NetLiquidationByCurrency", "NetLiquidation"):
                    try:
                        return float(tag.value)
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("Failed to fetch equity: %s", exc)
        return None

    async def get_positions_symbols(self) -> list[str]:
        """
        Fetch current positions and return a list of unique symbols.
        """
        await self.ensure_connected()
        # ib_insync has a sync positions(); use thread to avoid blocking
        positions = await asyncio.to_thread(self.ib.positions)
        symbols: list[str] = []
        for p in positions:
            sym = getattr(p.contract, "symbol", None)
            if sym:
                symbols.append(sym)
        return sorted(set(symbols))

    async def subscribe_broad_news(
        self,
        providers: list[str],
        queue: asyncio.Queue,
        fetch_body: bool = True,
    ) -> None:
        """
        Subscribe to BroadTape news (tickNewsEvent) for given providers (e.g., FLY, BRFG).
        Mirrors IBnews_streamV2.py behavior.
        """
        await self.ensure_connected()
        # remember for auto-resubscribe
        self._news_providers = providers

        for p in providers:
            p = p.strip()
            if not p:
                continue
            contract = Contract()
            contract.secType = "NEWS"
            contract.exchange = p
            contract.symbol = f"{p}:{p}_ALL"
            self.ib.reqMktData(
                contract,
                genericTickList="mdoff,292",
                snapshot=False,
                regulatorySnapshot=False,
            )
            logger.info("Subscribed to IB broad news provider=%s", p)

        async def _fetch_article(tick: NewsTick) -> Optional[str]:
            try:
                art = await self.ib.reqNewsArticleAsync(
                    providerCode=tick.providerCode, articleId=tick.articleId
                )
                if art and getattr(art, "articleText", None):
                    return art.articleText
                if art and getattr(art, "rawMessage", None):
                    return art.rawMessage
            except Exception as exc:
                logger.warning("Failed to fetch article %s:%s: %s", tick.providerCode, tick.articleId, exc)
            return None

        async def _handle_tick(tick: NewsTick) -> None:
            # drop duplicates by articleId (persistent)
            if tick.articleId in self._seen_articles:
                return
            self._seen_articles[tick.articleId] = time.time()
            self._persist_seen_article(tick.articleId)
            payload = {
                "provider": tick.providerCode,
                "articleId": tick.articleId,
                "headline": tick.headline,
                "extra": tick.extraData,
            }
            logger.info("IB NEWS TICK provider=%s id=%s headline=%s", tick.providerCode, tick.articleId, tick.headline)
            if fetch_body:
                body = await _fetch_article(tick)
                if body:
                    payload["body"] = body
            queue.put_nowait(payload)

        def _handler(tick: NewsTick) -> None:
            asyncio.create_task(_handle_tick(tick))

        self.ib.tickNewsEvent += _handler
        logger.info("Subscribed to IB broad news provider list: %s", providers)

    def _on_error(self, reqId: int, code: int, msg: str, misc: Any = None) -> None:
        # Auto re-subscribe on competing live session errors
        if code == 10197 and self._news_providers:
            logger.warning("IB error %s (%s): attempting news re-subscribe", code, msg)
            asyncio.create_task(self.subscribe_broad_news(self._news_providers, asyncio.Queue(), fetch_body=False))
