from __future__ import annotations

import os
from dataclasses import dataclass


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") in {"1", "true", "TRUE", "yes", "YES"}


@dataclass(frozen=True)
class AppSettings:
    debug: bool = env_bool("FLASK_DEBUG", False)
    host: str = os.getenv("TRADEWISE_HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", os.getenv("TRADEWISE_PORT", "5000")))
    open_browser: bool = env_bool("TRADEWISE_OPEN_BROWSER", True)
    market_status_ttl: int = int(os.getenv("TRADEWISE_MARKET_CACHE_TTL", "60"))
    analysis_ttl: int = int(os.getenv("TRADEWISE_ANALYSIS_CACHE_TTL", "180"))
    news_ttl: int = int(os.getenv("TRADEWISE_NEWS_CACHE_TTL", "120"))
    diagnostics_window: int = int(os.getenv("TRADEWISE_DIAGNOSTICS_WINDOW", "25"))


settings = AppSettings()
