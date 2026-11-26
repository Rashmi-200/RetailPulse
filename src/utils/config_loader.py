import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import yaml


class ConfigurationError(ValueError):
    """Raised when required environment variables or production secrets are invalid."""


_INSECURE_DEFAULTS = {
    "your_secure_password_here",
    "change_me_docker_db_pass",
    "change_me_airflow_db_pass",
    "change_me_api_key",
    "airflow_docker_pass",
    "retailpulse_docker_pass",
    "replace_with_generated_fernet_key",
    "46bkjoqylppoexq0ohdznilnepkff87wfwlbfzqddho=",
    "password",
    "admin",
    "secret",
    "changeme",
    "default",
    "postgres",
}

_API_PRODUCTION_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "RETAILPULSE_API_KEY",
)

_AIRFLOW_PRODUCTION_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "AIRFLOW_DB_NAME",
    "AIRFLOW_DB_USER",
    "AIRFLOW_DB_PASSWORD",
    "AIRFLOW_FERNET_KEY",
)

_REQUIRED_PRODUCTION_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DOCKER_DB_NAME",
    "DOCKER_DB_USER",
    "DOCKER_DB_PASSWORD",
    "AIRFLOW_DB_NAME",
    "AIRFLOW_DB_USER",
    "AIRFLOW_DB_PASSWORD",
    "AIRFLOW_FERNET_KEY",
    "RETAILPULSE_API_KEY",
)

_SECRET_LIKE_KEYS = {
    "DB_PASSWORD",
    "DOCKER_DB_PASSWORD",
    "AIRFLOW_DB_PASSWORD",
    "AIRFLOW_FERNET_KEY",
    "RETAILPULSE_API_KEY",
    "AZURE_STORAGE_ACCOUNT_KEY",
}


def load_yaml_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _is_production_env(env_map: Mapping[str, str], app_env: Optional[str] = None) -> bool:
    raw_env = (app_env if app_env is not None else env_map.get("APP_ENV", "development")).strip().lower()
    return raw_env in {"prod", "production"}


def is_insecure_secret(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in _INSECURE_DEFAULTS:
        return True
    if normalized.startswith(("change_me", "replace_with", "your_")):
        return True
    return False


def _validate_port(port_value: str) -> None:
    if not re.fullmatch(r"\d{1,5}", port_value.strip()):
        raise ConfigurationError("Invalid DB_PORT: must be a numeric port between 1 and 65535.")
    port_num = int(port_value.strip())
    if not (1 <= port_num <= 65535):
        raise ConfigurationError("Invalid DB_PORT: must be between 1 and 65535.")


def validate_runtime_environment(
    env: Optional[Mapping[str, str]] = None,
    *,
    strict_production: Optional[bool] = None,
    app_env: Optional[str] = None,
    required_keys: Optional[Iterable[str]] = None,
    service: str = "all",
) -> Dict[str, str]:
    """
    Validate environment variables and secrets without ever exposing secret values.
    """
    source_env: Mapping[str, str] = env if env is not None else os.environ
    is_prod = strict_production if strict_production is not None else _is_production_env(source_env, app_env=app_env)

    db_port = source_env.get("DB_PORT", "")
    if db_port and db_port.strip():
        _validate_port(db_port)

    if not is_prod:
        return {k: v for k, v in source_env.items() if k not in _SECRET_LIKE_KEYS}

    if required_keys is None:
        if service == "api":
            effective_keys: Iterable[str] = _API_PRODUCTION_KEYS
        elif service == "airflow":
            effective_keys = _AIRFLOW_PRODUCTION_KEYS
        else:
            effective_keys = _REQUIRED_PRODUCTION_KEYS
    else:
        effective_keys = required_keys

    errors = []
    for key in effective_keys:
        val = source_env.get(key)
        if val is None or not str(val).strip():
            errors.append(f"Missing required production environment variable: {key}")
            continue
        if key in _SECRET_LIKE_KEYS and is_insecure_secret(str(val)):
            errors.append(f"Insecure or default secret detected for production variable: {key}")

    if "AIRFLOW_FERNET_KEY" in set(effective_keys):
        fernet_key = source_env.get("AIRFLOW_FERNET_KEY", "").strip()
        if fernet_key and not is_insecure_secret(fernet_key) and len(fernet_key) < 32:
            errors.append("AIRFLOW_FERNET_KEY appears too short to be a valid Fernet key.")

    if errors:
        raise ConfigurationError("; ".join(errors))

    return {k: v for k, v in source_env.items() if k not in _SECRET_LIKE_KEYS}
