from ib_insync import *
import csv
import time
from string import ascii_uppercase
from collections import Counter

# ========================
# IB connection settings
# ========================
IB_HOST = "192.168.0.109"
IB_PORT = 4002
CLIENT_ID = 77

# ========================
# Output
# ========================
OUTPUT_CSV = "universe.csv"

# ========================
# Filters / Controls
# ========================
ALLOWED_PRIMARY_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}

SLEEP = 0.15               # 0.1~0.3 之间调；太快会被 IB 限流
PRINT_EVERY_ADDED = True   # True = 每加入一个 ticker 都打印
MAX_ADDED_PER_LETTER = 0   # 0 = 不限制；比如 200 表示每个字母最多收 200 个

# 你想强制在日志里看到的 ticker（即使它不一定会被 samples 命中，也会最后单独 qualify 一次）
WATCHLIST = {"AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA"}

def log(msg: str):
    print(msg, flush=True)

def qualify_stock(ib: IB, symbol: str):
    """
    Resolve a stock contract via SMART routing.
    Returns qualified contract or None.
    """
    c = Stock(symbol=symbol, exchange="SMART", currency="USD")
    try:
        cds = ib.qualifyContracts(c)
    except Exception:
        return None
    return cds[0] if cds else None

def main():
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    log(f"Connected to IB @ {IB_HOST}:{IB_PORT} clientId={CLIENT_ID}")

    universe = {}
    reasons = Counter()

    total_samples = 0
    total_added = 0

    # A-Z seed search
    for letter in ascii_uppercase:
        try:
            samples = ib.reqMatchingSymbols(letter)
        except Exception as e:
            log(f"[{letter}] ERROR reqMatchingSymbols: {e}")
            reasons["reqMatchingSymbols_error"] += 1
            continue

        added_this_letter = 0
        log(f"[{letter}] samples={len(samples)}")

        for s in samples:
            total_samples += 1
            c = s.contract

            # ---- stage 1: basic filters ----
            if c.secType != "STK":
                reasons["not_STK"] += 1
                continue
            if c.currency != "USD":
                reasons["not_USD"] += 1
                continue

            # NOTE: do NOT trust c.exchange here (often SMART/ARCA/ISLAND/etc)
            symbol = c.symbol

            # de-dupe early
            if symbol in universe:
                reasons["dup_symbol"] += 1
                continue

            # ---- stage 2: qualify to get the real contract ----
            cd = qualify_stock(ib, symbol)
            if cd is None:
                reasons["qualify_failed"] += 1
                continue

            # ---- stage 3: primaryExchange filter ----
            if cd.primaryExchange not in ALLOWED_PRIMARY_EXCHANGES:
                reasons[f"primary_not_allowed:{cd.primaryExchange or 'UNKNOWN'}"] += 1
                continue

            # success
            universe[cd.symbol] = {
                "symbol": cd.symbol,
                "conId": cd.conId,
                "primaryExchange": cd.primaryExchange,
                "currency": cd.currency
            }
            total_added += 1
            added_this_letter += 1

            if PRINT_EVERY_ADDED:
                log(f"  ✓ {cd.symbol:6s} conId={cd.conId:<10d} primary={cd.primaryExchange}")

            if MAX_ADDED_PER_LETTER and added_this_letter >= MAX_ADDED_PER_LETTER:
                reasons["max_per_letter_cap_hit"] += 1
                log(f"[{letter}] reached cap MAX_ADDED_PER_LETTER={MAX_ADDED_PER_LETTER}, skipping rest")
                break

            time.sleep(SLEEP)

        log(f"[{letter}] added={added_this_letter}")

    # ---- ensure watchlist is visible (and included if passes filters) ----
    log("\n[WATCHLIST] ensuring key tickers are resolved:")
    for sym in sorted(WATCHLIST):
        if sym in universe:
            log(f"  ✓ {sym} already in universe")
            continue

        cd = qualify_stock(ib, sym)
        if cd is None:
            log(f"  ✗ {sym} qualify_failed")
            reasons["watchlist_qualify_failed"] += 1
            continue

        if cd.primaryExchange not in ALLOWED_PRIMARY_EXCHANGES:
            log(f"  ✗ {sym} primary={cd.primaryExchange} not allowed")
            reasons[f"watchlist_primary_not_allowed:{cd.primaryExchange or 'UNKNOWN'}"] += 1
            continue

        universe[cd.symbol] = {
            "symbol": cd.symbol,
            "conId": cd.conId,
            "primaryExchange": cd.primaryExchange,
            "currency": cd.currency
        }
        total_added += 1
        log(f"  ✓ {cd.symbol} added via watchlist conId={cd.conId} primary={cd.primaryExchange}")

    ib.disconnect()

    # ---- summary ----
    log("\n========== SUMMARY ==========")
    log(f"Total samples seen: {total_samples}")
    log(f"Universe size:      {len(universe)}")
    log(f"Total added:        {total_added}")
    log("\nTop filter reasons:")
    for k, v in reasons.most_common(20):
        log(f"  {k:32s} {v}")

    # ---- write CSV ----
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "conId", "primaryExchange", "currency"]
        )
        writer.writeheader()
        for row in sorted(universe.values(), key=lambda x: x["symbol"]):
            writer.writerow(row)

    log(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
