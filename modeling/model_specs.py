"""Default model presets and config resolution helpers."""

from __future__ import annotations


DEFAULT_LOCATION_CONFIGS = {
    "satclip": {
        "model_id": "microsoft/SatCLIP-ViT16-L40",
        "checkpoint_filename": "satclip-vit16-l40.ckpt",
    },
    "geoclip": {"model_id": "geoclip", "checkpoint_filename": None},
}

DEFAULT_TEXT_CONFIGS = {
    "geoclip": {"model_id": "geoclip", "pretrained": "openai"},
    "gritlm": {"model_id": "GritLM/GritLM-7B", "pretrained": None},
    "openclip": {"model_id": "ViT-B-32", "pretrained": "openai"},
}


def resolve_location_config(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    checkpoint_filename: str | None = None,
    legacy_type: str | None = None,
    legacy_model: str | None = None,
    legacy_filename: str | None = None,
) -> dict:
    backend = backend or legacy_type or "satclip"
    if backend not in DEFAULT_LOCATION_CONFIGS:
        raise ValueError(
            f"Unsupported location backend={backend!r}. "
            f"Use one of {list(DEFAULT_LOCATION_CONFIGS)}."
        )

    defaults = DEFAULT_LOCATION_CONFIGS[backend]
    resolved = {
        "backend": backend,
        "model_id": model_id or legacy_model or defaults["model_id"],
        "checkpoint_filename": (
            checkpoint_filename
            if checkpoint_filename is not None
            else (legacy_filename if legacy_filename is not None else defaults["checkpoint_filename"])
        ),
    }

    if backend == "satclip":
        if not resolved["model_id"]:
            raise ValueError("satclip requires model_id.")
        if not resolved["checkpoint_filename"]:
            raise ValueError("satclip requires checkpoint_filename.")
    return resolved


def resolve_text_config(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    pretrained: str | None = None,
    legacy_type: str | None = None,
    legacy_model: str | None = None,
    legacy_vocabulary: str | None = None,
) -> dict:
    backend_guess = backend or legacy_type
    candidate_model = model_id or legacy_model
    if backend_guess is None and candidate_model == "ViT-B-32":
        backend_guess = "openclip"
    backend = backend_guess or "geoclip"

    if backend not in DEFAULT_TEXT_CONFIGS:
        raise ValueError(
            f"Unsupported text backend={backend!r}. "
            f"Use one of {list(DEFAULT_TEXT_CONFIGS)}."
        )

    defaults = DEFAULT_TEXT_CONFIGS[backend]
    resolved = {
        "backend": backend,
        "model_id": candidate_model or defaults["model_id"],
        "pretrained": (
            pretrained
            if pretrained is not None
            else (legacy_vocabulary if legacy_vocabulary is not None else defaults["pretrained"])
        ),
    }

    if backend == "geoclip":
        resolved["model_id"] = "geoclip"
        if resolved["pretrained"] in {"", None}:
            resolved["pretrained"] = "openai"
        if resolved["pretrained"] != "openai":
            raise ValueError(
                "geoclip backend expects pretrained='openai' (or omitted). "
                f"Got {resolved['pretrained']!r}."
            )
    elif backend == "openclip":
        if not resolved["model_id"]:
            raise ValueError("openclip requires model_id (e.g. ViT-B-32).")
        if not resolved["pretrained"]:
            raise ValueError("openclip requires pretrained (e.g. openai).")
    elif backend == "gritlm":
        if resolved["model_id"] in {"", None, "gritlm"}:
            resolved["model_id"] = defaults["model_id"]
    return resolved


def format_default_model_specs() -> str:
    return (
        "Default model presets:\n"
        "- location.satclip: model_id='microsoft/SatCLIP-ViT16-L40', checkpoint='satclip-vit16-l40.ckpt'\n"
        "- location.geoclip: model_id='geoclip'\n"
        "- text.geoclip: model_id='geoclip', pretrained='openai'\n"
        "- text.gritlm: model_id='GritLM/GritLM-7B'\n"
        "- text.openclip: model_id='ViT-B-32', pretrained='openai'"
    )
