"""Application settings, read from the environment or a `.env` file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Auth. Two people share one password; the username scopes their progress
    # and their saved drafts, and nothing else.
    app_users: str = "michu,enderboi"
    app_password: str = "mochimochi"
    session_secret: str = "dev-only-insecure-secret"
    session_days: int = 30
    cookie_secure: bool = False

    # Paths
    data_dir: Path = REPO_ROOT / "data"
    content_dir: Path = REPO_ROOT / "content"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    # The PDF book and its manifest, written by frontend/scripts/build-book.mjs
    # on the deploying machine and mounted into the container.
    book_dir: Path = REPO_ROOT / "book"
    # Models and clips a lab loads through `speechlab.asset_path`, exposed to
    # the runner as SPEECH_ASSETS. The GPU worker bakes them in at /assets;
    # here the directory is Git-ignored and usually empty, so a lab that
    # needs one checks `speechlab.has_asset` and falls back to something small.
    assets_dir: Path = REPO_ROOT / "assets"

    # Execution
    runner: str = "local"
    local_timeout_ceiling: int = 180
    # Zero leaves RLIMIT_AS off and lets the container's cgroup bound memory.
    # See `runner/local.py` for why an address-space limit is the wrong tool.
    local_memory_mb: int = 0
    # Two at once fits the VM, which has under 2 GB of memory to give.
    local_workers: int = 2

    # RunPod, used when a lesson asks for a GPU and an endpoint is configured.
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""
    runpod_gpu: str = "A100-80GB"
    runpod_timeout: int = 900

    git_sha: str = "unknown"

    @property
    def users(self) -> list[str]:
        return [name.strip() for name in self.app_users.split(",") if name.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def gpu_available(self) -> bool:
        return bool(self.runpod_api_key and self.runpod_endpoint_id)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
