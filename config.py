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
    llm_timeout: float = 
    llm_max_tokens: int = 
    llm_concurrency: int = 
    dedupe_enabled: bool = 
    accept_news_benzinga: bool = 
    accept_news_tws: bool = 
    tws_broad_providers: str = 
    accept_news_selfws: bool = 
    self_ws_url: str = 
    ib_host: str = 
    ib_port: int =  # paper default
    ib_client_id: int = 
    ib_account: str = 
    ib_default_notional: float = 
    max_risk_pct: float = 
    max_position_pct: float = 
    atr_duration: str = ""
    atr_bar_size: str = ""
    atr_use_rth: bool = 
    dedupe_ttl: float =  # embedding TTL default 1 day
    dedupe_similarity_threshold: float = 
    metrics_enabled: bool = 
    metrics_path: str = 

    @classmethod
    def from_env(cls) -> "Settings":
        def _b(name: str, default: str) -> bool:
            return os.environ.get(name, default).lower() in {"1", "true", "yes", "y", "on"}

        return cls(
            benzinga_ws_url=os.environ.get("BENZINGA_WS_URL", ""),
            benzinga_api_key=os.environ.get("BENZINGA_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            llm_api_key=os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            llm_base_url=os.environ.get("LLM_BASE_URL", ""),
            openai_model=os.environ.get("OPENAI_MODEL", ""),
            openai_fallback_model=os.environ.get("OPENAI_FALLBACK_MODEL", ""),
            llm_timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "")),
            llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "")),
            llm_concurrency=int(os.environ.get("LLM_CONCURRENCY", "")),
            dedupe_enabled=_b("DEDUPE_ENABLED", ""),
            accept_news_benzinga=_b("ACCEPT_NEWS_BENZINGA", ""),
            accept_news_tws=_b("ACCEPT_NEWS_TWS", ""),
            tws_broad_providers=os.environ.get("TWS_BROAD_PROVIDERS", ""),# BRFG, BZ, DJ, DJNL, FLY, DJTOP
            accept_news_selfws=_b("ACCEPT_NEWS_SELFWS", ""),
            self_ws_url=os.environ.get("SELF_WS_URL", ""),
            ib_host=os.environ.get("IB_HOST", ""),
            ib_port=int(os.environ.get("IB_PORT", "")),
            ib_client_id=int(os.environ.get("IB_CLIENT_ID", "")),
            ib_account=os.environ.get("IB_ACCOUNT", ""),
            ib_default_notional=float(os.environ.get("IB_DEFAULT_NOTIONAL", "")),
            max_risk_pct=float(os.environ.get("MAX_RISK_PCT", "")),
            max_position_pct=float(os.environ.get("MAX_POSITION_PCT", "")),
            atr_duration=os.environ.get("ATR_DURATION", ""),
            atr_bar_size=os.environ.get("ATR_BAR_SIZE", ""),
            atr_use_rth=_b("ATR_USE_RTH", ""),
            dedupe_ttl=float(os.environ.get("DEDUP_TTL_SECONDS", "")),
            dedupe_similarity_threshold=float(os.environ.get("DEDUP_SIM_THRESHOLD", "")),
            metrics_enabled=_b("METRICS_ENABLED", ""),
            metrics_path=os.environ.get("METRICS_PATH", ""),
        )
