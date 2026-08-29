from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGORA_", extra="ignore")

    database_url: str = "postgresql://agora:agora@127.0.0.1:5433/agora"
    redis_url: str = "redis://127.0.0.1:6379/0"
    small_model: str = "gpt-5.6-luna"
    big_model: str = "gpt-5.6-terra"

    # Phase 4c: Jobs authenticate to /runtime/* as the cluster host.
    # Empty means cluster auth is off (dev default: in-process DirectWorld).
    cluster_token: str = ""
    k8s_enabled: bool = False
    k8s_namespace: str = "agora"
    k8s_image: str = "agora:local"
    k8s_server_url: str = "http://agora.agora.svc.cluster.local:8000"
    k8s_api_url: str = ""
    k8s_token: str = ""
    k8s_ca_path: str = ""
    k8s_insecure: bool = False
    k8s_job_timeout_s: float = 180.0
    k8s_poll_s: float = 0.5
    k8s_ttl_s: int = 120
    k8s_secret_name: str = ""

    # Stall sweep (server/stall.py): proactive wake for quiet rooms with
    # work owed. Arithmetic eligibility, per-stall decline cap.
    stall_min_s: float = 20.0
    stall_max_s: float = 3600.0
    stall_max_nudges: int = 3
    stall_interval_s: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
