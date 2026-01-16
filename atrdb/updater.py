#!/usr/bin/env python3
"""
ATR batch updater without IB.
Sources: yfinance (primary) with stooq fallback.
Supports resume via a progress file.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from atr_db import ATRDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SP500_CSV = Path(__file__).with_name("sp500_tickers.csv")
PROGRESS_FILE = Path(__file__).with_name("atr_update_progress.json")


def _parse_duration_days(value: str) -> int:
    raw = str(value or "").strip().upper()
    if not raw:
        return 30
    parts = raw.split()
    try:
        num = int(parts[0])
    except Exception:
        return 30
    unit = parts[1] if len(parts) > 1 else "D"
    if unit.startswith("W"):
        return num * 7
    if unit.startswith("M"):
        return num * 30
    if unit.startswith("Y"):
        return num * 365
    return num


def _yfinance_period(days: int) -> str:
    if days <= 31:
        return "1mo"
    if days <= 93:
        return "3mo"
    if days <= 186:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"


def _yfinance_interval(bar_size: str) -> str:
    raw = str(bar_size or "").strip().lower()
    if raw in {"1 hour", "1h", "60m", "60min", "60 mins", "1 hr"}:
        return "60m"
    if raw in {"1 day", "1d", "day"}:
        return "1d"
    return "1d"


def _read_sp500(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing S&P 500 list: {path}")
    tickers: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            sym = row[0].strip().upper()
            if not sym or sym == "SYMBOL":
                continue
            tickers.append(sym)
    return tickers


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed": [], "failed": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        completed = data.get("completed") or []
        failed = data.get("failed") or []
        return {"completed": list(completed), "failed": list(failed)}
    except Exception:
        return {"completed": [], "failed": []}


def _save_progress(path: Path, completed: Iterable[str], failed: Iterable[str]) -> None:
    payload = {
        "completed": sorted(set(completed)),
        "failed": sorted(set(failed)),
        "updated_at": datetime.utcnow().isoformat(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_ema_atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    if len(highs) != len(lows) or len(highs) != len(closes):
        return None
    if len(closes) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = trs[-period]
    alpha = 2 / (period + 1)
    for tr in trs[-period + 1:]:
        atr = alpha * tr + (1 - alpha) * atr
    return atr


def _fetch_yfinance(
    symbol: str,
    days: int,
    interval: str,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    try:
        import yfinance as yf
        import pandas as pd
    except Exception:
        return None
    fetch_symbol = symbol.replace(".", "-")
    period = _yfinance_period(days)
    try:
        df = yf.download(fetch_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    try:
        if isinstance(df.columns, pd.MultiIndex):
            for level in (0, -1):
                lvl = df.columns.get_level_values(level)
                if {"Open", "High", "Low", "Close"}.issubset(set(lvl)):
                    df = df.copy()
                    df.columns = lvl
                    break

        def _col(name: str):
            if name in df.columns:
                return df[name]
            lower_map = {str(c).lower(): c for c in df.columns}
            key = name.lower()
            if key in lower_map:
                return df[lower_map[key]]
            return None

        high = _col("High")
        low = _col("Low")
        close = _col("Close")
        if high is None or low is None or close is None:
            return None

        def _to_list(obj) -> Optional[List[float]]:
            if isinstance(obj, pd.DataFrame):
                if obj.empty:
                    return None
                obj = obj.iloc[:, 0]
            if isinstance(obj, pd.Series):
                obj = pd.to_numeric(obj, errors="coerce").dropna()
                if obj.empty:
                    return None
                return obj.astype(float).tolist()
            try:
                return [float(x) for x in obj]
            except Exception:
                return None

        highs = _to_list(high)
        lows = _to_list(low)
        closes = _to_list(close)
        if not highs or not lows or not closes:
            return None
        min_len = min(len(highs), len(lows), len(closes))
        if min_len == 0:
            return None
        return highs[-min_len:], lows[-min_len:], closes[-min_len:]
    except Exception:
        return None


def _fetch_stooq(symbol: str, interval: str) -> Optional[Tuple[List[float], List[float], List[float]]]:
    if interval != "1d":
        return None
    fetch_symbol = symbol.replace(".", "-").lower()
    url = f"https://stooq.com/q/d/l/?s={fetch_symbol}.us&i=d"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return None
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return None
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    for row in rows:
        try:
            highs.append(float(row["High"]))
            lows.append(float(row["Low"]))
            closes.append(float(row["Close"]))
        except Exception:
            continue
    if not closes:
        return None
    return highs, lows, closes


class ATRUpdater:
    def __init__(
        self,
        atr_duration: str = "30 D",
        atr_bar_size: str = "1 hour",
        atr_use_rth: bool = True,
        atr_period: int = 14,
        sp500_path: Path = SP500_CSV,
        use_yfinance: bool = True,
        use_stooq: bool = True,
    ) -> None:
        self.db = ATRDatabase()
        self.atr_period = atr_period
        self.duration_days = _parse_duration_days(atr_duration)
        self.bar_interval = _yfinance_interval(atr_bar_size)
        self.atr_use_rth = atr_use_rth
        self.sp500_path = sp500_path
        self.use_yfinance = use_yfinance
        self.use_stooq = use_stooq
        # yfinance is not thread-safe; serialize requests to avoid cross-talk.
        self._yf_lock = asyncio.Lock()

    def get_ticker_universe(self) -> List[str]:
        return _read_sp500(self.sp500_path)

    def _calculate_atr_sync(self, ticker: str) -> Optional[float]:
        sources = []
        if self.use_yfinance:
            sources.append("yfinance")
        if self.use_stooq:
            sources.append("stooq")
        for source in sources:
            if source == "yfinance":
                ohlc = _fetch_yfinance(ticker, self.duration_days, self.bar_interval)
            else:
                ohlc = _fetch_stooq(ticker, self.bar_interval)
            if not ohlc:
                continue
            highs, lows, closes = ohlc
            if len(closes) < self.atr_period + 1:
                continue
            atr = _compute_ema_atr(highs, lows, closes, self.atr_period)
            if atr is None:
                continue
            last_price = closes[-1]
            atr_pct = atr / last_price if last_price > 0 else 0
            if atr_pct < 0.005:
                atr = last_price * 0.02
            elif atr_pct > 0.15:
                atr = last_price * 0.10
            return atr
        return None

    async def update_ticker(self, ticker: str) -> Tuple[str, bool]:
        try:
            if self.use_yfinance:
                async with self._yf_lock:
                    atr = await asyncio.to_thread(self._calculate_atr_sync, ticker)
                    await asyncio.sleep(0.2)
            else:
                atr = await asyncio.to_thread(self._calculate_atr_sync, ticker)
        except Exception as exc:
            logger.warning("✗ %s: error %s", ticker, exc)
            return ticker, False
        if atr is None:
            logger.warning("✗ %s: no data", ticker)
            return ticker, False
        self.db.upsert(ticker, atr)
        logger.info("✓ %s: ATR=%.6f", ticker, atr)
        return ticker, True

    async def update_all(self, tickers: List[str], batch_size: int = 25, resume: bool = True) -> None:
        progress = _load_progress(PROGRESS_FILE)
        completed = set(progress["completed"]) if resume else set()
        failed = set(progress["failed"]) if resume else set()

        pending = [t for t in tickers if t not in completed]
        total = len(pending)
        if total == 0:
            logger.info("All tickers already completed.")
            return

        logger.info("Starting update for %s tickers (pending).", total)

        for i in range(0, total, batch_size):
            batch = pending[i:i + batch_size]
            logger.info("Batch %s: %s-%s/%s", i // batch_size + 1, i + 1, min(i + batch_size, total), total)
            results = await asyncio.gather(*(self.update_ticker(t) for t in batch), return_exceptions=True)
            for item in results:
                if isinstance(item, Exception):
                    continue
                sym, ok = item
                if ok:
                    completed.add(sym)
                    failed.discard(sym)
                else:
                    failed.add(sym)
            _save_progress(PROGRESS_FILE, completed, failed)
            await asyncio.sleep(1)

        logger.info("Update complete: %s success, %s failed", len(completed), len(failed))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Update ATR cache (S&P 500 only)")
    parser.add_argument("--limit", type=int, help="Limit number of tickers")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size")
    parser.add_argument("--atr-duration", default=os.environ.get("ATR_DURATION", "30 D"), help="ATR duration")
    parser.add_argument("--atr-bar-size", default=os.environ.get("ATR_BAR_SIZE", "1 hour"), help="ATR bar size")
    parser.add_argument("--atr-use-rth", default=os.environ.get("ATR_USE_RTH", "1"), help="Use RTH for ATR bars")
    parser.add_argument("--atr-period", type=int, default=int(os.environ.get("ATR_PERIOD", "14")), help="ATR period")
    parser.add_argument("--no-yfinance", action="store_true", help="Disable yfinance source")
    parser.add_argument("--no-stooq", action="store_true", help="Disable stooq source")
    parser.add_argument("--reset-progress", action="store_true", help="Reset progress file")

    args = parser.parse_args()

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    updater = ATRUpdater(
        atr_duration=args.atr_duration,
        atr_bar_size=args.atr_bar_size,
        atr_use_rth=str(args.atr_use_rth).lower() in {"1", "true", "yes", "y", "on"},
        atr_period=args.atr_period,
        use_yfinance=not args.no_yfinance,
        use_stooq=not args.no_stooq,
    )

    tickers = updater.get_ticker_universe()
    if args.limit:
        tickers = tickers[:args.limit]
    await updater.update_all(tickers, batch_size=args.batch_size, resume=not args.reset_progress)


if __name__ == "__main__":
    asyncio.run(main())
