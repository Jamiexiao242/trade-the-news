import os
from dataclasses import dataclass


@dataclass
class Settings:
    benzinga_ws_url: str
    benzinga_api_key: str
    openai_api_key: str
    llm_api_key: str
    llm_base_url: str = ""
    openai_model: str = ""
    openai_fallback_model: str = ""
    llm_timeout: float = 10.0
    llm_max_tokens: int = 120
    llm_concurrency: int = 8
    dedupe_enabled: bool = False
    accept_news_benzinga: bool = False
    accept_news_tws: bool = False
    tws_broad_providers: str = ""
    accept_news_selfws: bool = False
    self_ws_url: str = ""
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # paper default
    ib_client_id: int = 1
    ib_account: str = ""
    ib_default_notional: float = 10000.0
    max_risk_pct: float = 0.01
    max_position_pct: float = 0.10
    atr_duration: str = ""
    atr_bar_size: str = ""
    atr_use_rth: bool = True
    dedupe_ttl: float = 86400.0  # embedding TTL default 1 day
    dedupe_similarity_threshold: float = 0.9
    metrics_enabled: bool = True
    metrics_path: str = "metrics.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        def _b(name: str, default: str) -> bool:
            return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}

        def _f(name: str, default: float) -> float:
            val = os.environ.get(name, "")
            try:
                return float(val)
            except Exception:
                return default

        def _i(name: str, default: int) -> int:
            val = os.environ.get(name, "")
            try:
                return int(val)
            except Exception:
                return default

        return cls(
            benzinga_ws_url=os.environ.get("BENZINGA_WS_URL", ""),
            benzinga_api_key=os.environ.get("BENZINGA_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_api_key=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            llm_base_url=os.environ.get("LLM_BASE_URL", ""),
            openai_model=os.environ.get("OPENAI_MODEL", ""),
            openai_fallback_model=os.environ.get("OPENAI_FALLBACK_MODEL", ""),
            llm_timeout=_f("LLM_TIMEOUT_SECONDS", 10.0),
            llm_max_tokens=_i("LLM_MAX_TOKENS", 120),
            llm_concurrency=_i("LLM_CONCURRENCY", 8),
            dedupe_enabled=_b("DEDUPE_ENABLED", "false"),
            accept_news_benzinga=_b("ACCEPT_NEWS_BENZINGA", "false"),
            accept_news_tws=_b("ACCEPT_NEWS_TWS", "false"),
            tws_broad_providers=os.environ.get("TWS_BROAD_PROVIDERS", ""),  # BRFG, BZ, DJ, DJNL, FLY, DJTOP
            accept_news_selfws=_b("ACCEPT_NEWS_SELFWS", "false"),
            self_ws_url=os.environ.get("SELF_WS_URL", ""),
            ib_host=os.environ.get("IB_HOST", "127.0.0.1"),
            ib_port=_i("IB_PORT", 7497),
            ib_client_id=_i("IB_CLIENT_ID", 1),
            ib_account=os.environ.get("IB_ACCOUNT", ""),
            ib_default_notional=_f("IB_DEFAULT_NOTIONAL", 10000.0),
            max_risk_pct=_f("MAX_RISK_PCT", 0.01),
            max_position_pct=_f("MAX_POSITION_PCT", 0.10),
            atr_duration=os.environ.get("ATR_DURATION", ""),
            atr_bar_size=os.environ.get("ATR_BAR_SIZE", ""),
            atr_use_rth=_b("ATR_USE_RTH", "true"),
            dedupe_ttl=_f("DEDUP_TTL_SECONDS", 86400.0),
            dedupe_similarity_threshold=_f("DEDUP_SIM_THRESHOLD", 0.9),
            metrics_enabled=_b("METRICS_ENABLED", "true"),
            metrics_path=os.environ.get("METRICS_PATH", "metrics.jsonl"),
        )
