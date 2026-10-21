"""The API's settings, read from the environment with pydantic-settings.

Every setting is an environment variable named `RECOMMENDITOS_<FIELD>`, so the
Compose file and the VM's environment file configure the service without a
config file in the image. Locally, `recommenditos.config` loads the repository's
`.env` at import time, which puts its values into the environment first.

The API key is a `SecretStr`: its repr, a log line and a traceback show
`**********`, never the key.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from recommenditos.config import MODELS_DIR, PROJ_ROOT

ENV_PREFIX = "RECOMMENDITOS_"

#: The variable that names the bundle to serve.
MODEL_DIR_ENV_VAR = f"{ENV_PREFIX}MODEL_DIR"

#: The variable that holds `/v1/feedback`'s key (FR-10). It is a secret (NFR-15):
#: `.env.template` lists it without a value.
API_KEY_ENV_VAR = f"{ENV_PREFIX}API_KEY"

#: The variable that names the SQLite file of the prediction log and the feedback.
PREDICTION_LOG_ENV_VAR = f"{ENV_PREFIX}PREDICTION_LOG"

#: The shortest key the service accepts, so that a placeholder such as `test` or
#: `changeme` stops the service at startup instead of guarding the endpoint.
MIN_API_KEY_LENGTH = 16

#: NFR-04 caps the prediction log and the feedback together at 2 GB.
DEFAULT_LOG_MAX_BYTES = 2_000_000_000


class Settings(BaseSettings):
    """What the service is started with."""

    # `hide_input_in_errors`: a refused value, such as a key that is too short,
    # is not repeated in the error, which the service log would show (NFR-15).
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, frozen=True, hide_input_in_errors=True
    )

    model_dir: Path = Field(
        default=MODELS_DIR / "lgbm-basic",
        description="The bundle to serve (EDN-08, FR-12).",
    )
    api_key: Annotated[SecretStr, Field(min_length=MIN_API_KEY_LENGTH)] | None = Field(
        default=None,
        description="The key `/v1/feedback` requires in `X-API-Key` (FR-10). Unset or empty, "
        "the endpoint refuses every request.",
    )
    prediction_log: Path = Field(
        default=PROJ_ROOT / "logs" / "prediction-log.sqlite3",
        description="The SQLite file that holds the prediction log and the feedback (EDN-86).",
    )
    prediction_log_max_bytes: int = Field(
        default=DEFAULT_LOG_MAX_BYTES,
        gt=0,
        description="The size the log and the feedback are pruned to, oldest first (NFR-04).",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_no_key(cls, value: object) -> object:
        """`RECOMMENDITOS_API_KEY=` copied from the template means no key, not the empty one."""
        if isinstance(value, str) and not value.strip():
            return None
        return value
