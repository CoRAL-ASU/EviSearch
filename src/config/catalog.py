"""
Inference catalog loader and validator.

src/config/catalog.yaml lists every endpoint, model, role, local vLLM server, preset and option.
src/config/config.py selects which of them a run uses; Catalog.resolve() checks that selection
against the catalog and returns the Selection the inference layer reads.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")

Capability = Literal["tools", "json_schema", "images", "pdf"]
ModelKind = Literal["chat", "embedding", "reranker"]
GpuAssignment = Union[str, List[int]]


class ConfigError(ValueError):
    """The catalog, or the selection made in config.py, is invalid."""


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Capabilities(_Spec):
    tools: bool = False
    json_schema: bool = False
    images: bool = False
    pdf: bool = False

    def missing(self, required: Iterable[str]) -> List[str]:
        return [name for name in required if not getattr(self, name)]


class Price(_Spec):
    input: float = 0.0
    output: float = 0.0


class ImageTokens(_Spec):
    """How many prompt tokens a model spends on an image: one per pixels_per_token square after scaling the image down
    to fit max_side, plus one break token per row when row_break is set (Pixtral). The default (32 px squares, no
    downscaling) is an upper bound for Gemini and OpenAI, which tile images more cheaply."""

    pixels_per_token: int = Field(32, ge=1)
    max_side: Optional[int] = Field(None, ge=1)
    row_break: bool = False

    def count(self, width: int, height: int) -> int:
        ratio = max(width, height) / self.max_side if self.max_side else 1.0
        if ratio > 1:
            width, height = round(width / ratio), round(height / ratio)
        columns = math.ceil(width / self.pixels_per_token)
        rows = math.ceil(height / self.pixels_per_token)
        return rows * (columns + (1 if self.row_break else 0))


class EndpointSpec(_Spec):
    type: Literal["openai_compatible", "gemini", "vllm_rerank"]
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    server: Optional[str] = None
    api_key_env: Optional[str] = None
    project_env: Optional[str] = None
    location_env: Optional[str] = None
    default_location: str = "us-central1"
    timeout_s: float = 600.0
    max_retries: int = 2


class ModelSpec(_Spec):
    kind: ModelKind
    endpoint: str
    name: str
    capabilities: Capabilities = Capabilities()
    context_tokens: Optional[int] = None
    thinking: Optional[bool] = None
    query_instruction: Optional[str] = None
    image_tokens: ImageTokens = ImageTokens()
    price_per_1k: Price = Price()


class RoleSpec(_Spec):
    kind: ModelKind
    requires: List[Capability] = []
    optional: bool = False
    description: str = ""


class ServerSpec(_Spec):
    model_path: str
    host: str = "127.0.0.1"
    port: int
    tensor_parallel: int = Field(1, ge=1)
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1)
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    args: List[str] = []
    env: Dict[str, str] = {}  # extra environment variables for the server process


class Catalog(_Spec):
    endpoints: Dict[str, EndpointSpec]
    models: Dict[str, ModelSpec]
    roles: Dict[str, RoleSpec]
    servers: Dict[str, ServerSpec] = {}
    presets: Dict[str, Dict[str, Optional[str]]]
    options: Dict[str, List[str]] = {}

    # ---- lookups -----------------------------------------------------------------------------

    def endpoint_for(self, model_key: str) -> EndpointSpec:
        return self.endpoints[self.models[model_key].endpoint]

    def server_for_model(self, model_key: str) -> Optional[str]:
        return self.endpoint_for(model_key).server

    def models_on_server(self, server_key: str) -> List[str]:
        return [key for key in self.models if self.server_for_model(key) == server_key]

    def models_for_role(self, role: str) -> List[str]:
        return [key for key in self.models if self.role_model_error(role, key) is None]

    def role_model_error(self, role: str, model_key: Optional[str]) -> Optional[str]:
        spec = self.roles[role]
        if model_key is None:
            return None if spec.optional else f"role '{role}' needs a model. Valid: {self._valid(role)}"
        model = self.models.get(model_key)
        if model is None:
            return f"role '{role}': unknown model '{model_key}'. Valid: {self._valid(role)}"
        if model.kind != spec.kind:
            return (
                f"role '{role}' needs a {spec.kind} model, but '{model_key}' has kind={model.kind}. "
                f"Valid: {self._valid(role)}"
            )
        missing = model.capabilities.missing(spec.requires)
        if missing:
            return (
                f"role '{role}' needs {', '.join(missing)}; '{model_key}' does not support it. "
                f"Valid: {self._valid(role)}"
            )
        return None

    def check_model_for_role(self, role: str, model_key: Optional[str]) -> None:
        if role not in self.roles:
            raise ConfigError(f"Unknown role '{role}'. Roles: {', '.join(self.roles)}")
        error = self.role_model_error(role, model_key)
        if error:
            raise ConfigError(error)

    def _valid(self, role: str) -> str:
        spec = self.roles[role]
        valid = [
            key
            for key, model in self.models.items()
            if model.kind == spec.kind and not model.capabilities.missing(spec.requires)
        ]
        return ", ".join(valid) or "(none)"

    # ---- validation --------------------------------------------------------------------------

    def validate_references(self) -> None:
        errors: List[str] = []
        for key, endpoint in self.endpoints.items():
            if endpoint.server and endpoint.server not in self.servers:
                errors.append(f"endpoints.{key}: unknown server '{endpoint.server}'")
            if endpoint.type != "gemini" and not (endpoint.base_url or endpoint.server or endpoint.base_url_env):
                errors.append(f"endpoints.{key}: needs base_url, base_url_env or server")
        for key, model in self.models.items():
            endpoint = self.endpoints.get(model.endpoint)
            if endpoint is None:
                errors.append(f"models.{key}: unknown endpoint '{model.endpoint}'")
                continue
            if (model.kind == "reranker") != (endpoint.type == "vllm_rerank"):
                errors.append(f"models.{key}: reranker models need a vllm_rerank endpoint and vice versa")
            if model.kind == "embedding" and endpoint.type != "openai_compatible":
                errors.append(f"models.{key}: embedding models need an openai_compatible endpoint")
        if errors:
            raise ConfigError(_format_errors("Invalid catalog (src/config/catalog.yaml)", errors))

        for key in self.servers:
            served = self.models_on_server(key)
            if len(served) != 1:
                errors.append(f"servers.{key}: must be served by exactly one model (found: {served or 'none'})")
        for name, values in self.options.items():
            if not values:
                errors.append(f"options.{name}: needs at least one value")
        for preset, assignment in self.presets.items():
            for role in assignment:
                if role not in self.roles:
                    errors.append(f"presets.{preset}: unknown role '{role}'")
            for role, spec in self.roles.items():
                if role not in assignment:
                    if not spec.optional:
                        errors.append(f"presets.{preset}: missing role '{role}'")
                    continue
                error = self.role_model_error(role, assignment[role])
                if error:
                    errors.append(f"presets.{preset}: {error}")
        if errors:
            raise ConfigError(_format_errors("Invalid catalog (src/config/catalog.yaml)", errors))

    # ---- selection ---------------------------------------------------------------------------

    def resolve(
        self,
        preset: str,
        role_overrides: Optional[Mapping[str, Optional[str]]] = None,
        options: Optional[Mapping[str, str]] = None,
        gpus: Optional[Mapping[str, GpuAssignment]] = None,
        gpu_pool: Iterable[int] = (),
    ) -> "Selection":
        """Validate a selection made in config.py and return it. Raises ConfigError listing every problem."""
        if preset not in self.presets:
            raise ConfigError(f"Unknown preset '{preset}'. Presets: {', '.join(self.presets)}")

        errors: List[str] = []
        roles: Dict[str, Optional[str]] = {role: self.presets[preset].get(role) for role in self.roles}
        for role, model_key in (role_overrides or {}).items():
            if role not in self.roles:
                errors.append(f"ROLE_OVERRIDES: unknown role '{role}'. Roles: {', '.join(self.roles)}")
                continue
            error = self.role_model_error(role, model_key)
            if error:
                errors.append(f"ROLE_OVERRIDES: {error}")
                continue
            roles[role] = model_key

        chosen: Dict[str, str] = {name: values[0] for name, values in self.options.items()}
        for name, value in (options or {}).items():
            if name not in self.options:
                errors.append(f"OPTIONS: unknown option '{name}'. Options: {', '.join(self.options)}")
            elif value not in self.options[name]:
                errors.append(f"OPTIONS['{name}']={value!r}: choose one of {' | '.join(self.options[name])}")
            else:
                chosen[name] = value

        pdf_model = roles.get("pdf_query")
        if chosen.get("pdf_query_input") == "markdown_images" and pdf_model in self.models:
            if not self.models[pdf_model].capabilities.images:
                readers = [key for key in self.models_for_role("pdf_query") if self.models[key].capabilities.images]
                errors.append(
                    f"OPTIONS['pdf_query_input']='markdown_images' but '{pdf_model}' cannot read images. "
                    f"Use 'markdown' or pick: {', '.join(readers)}"
                )

        pool = list(gpu_pool)
        if any(not isinstance(g, int) or isinstance(g, bool) or g < 0 for g in pool) or len(set(pool)) != len(pool):
            errors.append(f"GPU_POOL={pool}: must be unique non-negative GPU indices")
        resolved_gpus: Dict[str, GpuAssignment] = {}
        for server, assignment in (gpus or {}).items():
            if server not in self.servers:
                errors.append(f"GPUS: unknown server '{server}'. Servers: {', '.join(self.servers)}")
                continue
            if assignment == "auto":
                resolved_gpus[server] = "auto"
                continue
            if not isinstance(assignment, list) or not all(isinstance(g, int) and not isinstance(g, bool) for g in assignment):
                errors.append(f"GPUS['{server}']={assignment!r}: use \"auto\" or a list of GPU indices")
                continue
            outside = [g for g in assignment if g not in pool]
            if outside:
                errors.append(f"GPUS['{server}'] uses GPU {outside}, which is not in GPU_POOL {pool}")
            needed = self.servers[server].tensor_parallel
            if len(assignment) != needed:
                errors.append(
                    f"GPUS['{server}'] lists {len(assignment)} GPU(s) but the server needs tensor_parallel={needed}"
                )
            resolved_gpus[server] = list(assignment)
        for server in self.servers:
            resolved_gpus.setdefault(server, "auto")

        if errors:
            raise ConfigError(_format_errors(f"Invalid selection in src/config/config.py (preset '{preset}')", errors))
        return Selection(catalog=self, preset=preset, roles=roles, options=chosen, gpus=resolved_gpus, gpu_pool=pool)


@dataclass(frozen=True)
class Selection:
    """What this run uses: role -> model key, option values and GPU placement for local servers."""

    catalog: Catalog
    preset: str
    roles: Dict[str, Optional[str]]
    options: Dict[str, str]
    gpus: Dict[str, GpuAssignment]
    gpu_pool: List[int]

    def model_key(self, role: str) -> Optional[str]:
        if role not in self.catalog.roles:
            raise ConfigError(f"Unknown role '{role}'. Roles: {', '.join(self.catalog.roles)}")
        return self.roles.get(role)

    def model(self, role: str) -> Optional[ModelSpec]:
        key = self.model_key(role)
        return self.catalog.models[key] if key else None

    def option(self, name: str) -> str:
        if name not in self.options:
            raise ConfigError(f"Unknown option '{name}'. Options: {', '.join(self.options)}")
        return self.options[name]

    def servers_needed(self) -> List[str]:
        """Local servers behind the models this selection uses, in catalog order."""
        used = {self.catalog.server_for_model(key) for key in self.roles.values() if key}
        return [server for server in self.catalog.servers if server in used]


def _format_errors(title: str, errors: List[str]) -> str:
    return title + ":\n  - " + "\n  - ".join(errors)


@lru_cache(maxsize=None)
def _load_catalog_cached(path: str) -> Catalog:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read inference catalog {path}: {exc}") from exc
    try:
        catalog = Catalog.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid catalog ({path}):\n{exc}") from exc
    catalog.validate_references()
    return catalog


def load_catalog(path: Optional[Path] = None) -> Catalog:
    return _load_catalog_cached(str(path or CATALOG_PATH))


def env(name: str, default: Any) -> Any:
    """Read an override from the environment, parsed to the type of `default`.

    list defaults take "0,1,2"; dict defaults (GPU placement) take "server=0,1;other=auto".
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    raw = raw.strip()
    try:
        if isinstance(default, bool):
            return raw.lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, list):
            return [int(part) for part in raw.split(",") if part.strip()]
        if isinstance(default, dict):
            merged = dict(default)
            for item in raw.split(";"):
                if not item.strip():
                    continue
                key, sep, value = item.partition("=")
                if not sep:
                    raise ValueError(f"expected server=gpus, got {item!r}")
                value = value.strip()
                merged[key.strip()] = "auto" if value == "auto" else [int(g) for g in value.split(",") if g.strip()]
            return merged
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}={raw!r} is invalid: {exc}") from exc
    return raw


def role_overrides_from_env(prefix: str = "EVISEARCH_ROLE_") -> Dict[str, Optional[str]]:
    """EVISEARCH_ROLE_SEARCH_AGENT=gemini-2.5-flash -> {"search_agent": "gemini-2.5-flash"}; "none" clears a role."""
    overrides: Dict[str, Optional[str]] = {}
    for name, value in os.environ.items():
        if name.startswith(prefix) and value.strip():
            model = value.strip()
            overrides[name[len(prefix):].lower()] = None if model.lower() in {"none", "null"} else model
    return overrides
