from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a SentinelFlow configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration loaded from YAML."""

    raw: dict[str, Any]
    root: Path

    @property
    def dataset_path(self) -> Path:
        return self._resolve(self.raw["paths"]["dataset"])

    @property
    def results_path(self) -> Path:
        return self._resolve(self.raw["paths"]["results"])

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def target_column(self) -> str:
        return str(self.raw["data"]["target_column"])

    @property
    def cnp_column(self) -> str:
        return str(self.raw["data"]["cnp_column"])

    @property
    def cnp_values(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["data"]["cnp_values"])

    @property
    def batch_size(self) -> int:
        return int(self.raw["data"]["batch_size"])

    @property
    def warmup_rows(self) -> int:
        return int(self.raw["data"]["warmup_rows"])

    @property
    def label_delay_batches(self) -> int:
        return int(self.raw["stream"]["label_delay_batches"])

    def section(self, name: str) -> dict[str, Any]:
        section = self.raw.get(name)

        if not isinstance(section, dict):
            raise ConfigurationError(f"Configuration section '{name}' is missing or invalid.")

        return section

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (self.root / path).resolve()


_REQUIRED_SECTIONS = {
    "project",
    "paths",
    "data",
    "stream",
    "drift",
    "models",
    "evaluation",
}


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a SentinelFlow YAML configuration."""

    config_path = Path(path).expanduser().resolve()

    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a YAML mapping.")

    missing_sections = sorted(_REQUIRED_SECTIONS.difference(raw))

    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ConfigurationError(f"Configuration is missing required sections: {missing}")

    config = ProjectConfig(
        raw=raw,
        root=config_path.parent.parent,
    )

    _validate_config(config)

    return config


def _validate_config(config: ProjectConfig) -> None:
    if config.seed < 0:
        raise ConfigurationError("project.random_seed must be nonnegative.")

    if config.batch_size <= 0:
        raise ConfigurationError("data.batch_size must be greater than zero.")

    if config.warmup_rows <= 0:
        raise ConfigurationError("data.warmup_rows must be greater than zero.")

    if config.label_delay_batches < 0:
        raise ConfigurationError("stream.label_delay_batches must be nonnegative.")

    if not config.target_column:
        raise ConfigurationError("data.target_column cannot be empty.")

    if not config.cnp_column:
        raise ConfigurationError("data.cnp_column cannot be empty.")

    if not config.cnp_values:
        raise ConfigurationError("data.cnp_values cannot be empty.")

    drift = config.section("drift")

    start_fraction = float(drift["start_fraction"])
    end_fraction = float(drift["end_fraction"])
    max_intensity = float(drift["max_intensity"])

    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ConfigurationError(
            "Drift fractions must satisfy 0 <= start_fraction < end_fraction <= 1."
        )

    if not 0.0 <= max_intensity <= 1.0:
        raise ConfigurationError("drift.max_intensity must be between zero and one.")
