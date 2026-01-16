"""
Simple read-only metrics API.
Exposes:
    GET /metrics?limit=500
Reads from METRICS_PATH (default metrics.db). Supports:
- SQLite: events(ts REAL, kind TEXT, payload TEXT)
- JSONL: fallback if path endswith .jsonl
"""

import json
import os
import sqlite3
from typing import List

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.db")

app = FastAPI(title="Metrics API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_db() -> None:
    if not METRICS_PATH.endswith(".db"):
        return
    con = sqlite3.connect(METRICS_PATH)
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, payload TEXT)"
        )
        con.commit()
    finally:
        con.close()


def _load_from_db(limit: int) -> List[dict]:
    _ensure_db()
    con = sqlite3.connect(METRICS_PATH)
    try:
        rows = con.execute(
            "SELECT ts, kind, payload FROM events ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()
    out = []
    for ts, kind, payload in rows:
        try:
            obj = json.loads(payload)
        except Exception:
            obj = payload
        out.append({"ts": ts, "kind": kind, "payload": obj})
    return out


def _load_from_jsonl(limit: int) -> List[dict]:
    out = []
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(obj)
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except FileNotFoundError:
        return []
    return out


@app.get("/metrics")
def metrics(limit: int = Query(500, ge=1, le=5000)) -> List[dict]:
    if METRICS_PATH.endswith(".db"):
        return _load_from_db(limit)
    return _load_from_jsonl(limit)
