"""Satellite-focused prompts for VLMs; optional ensembling via prompt merging."""

from __future__ import annotations

from typing import List, Sequence


def default_satellite_prompts() -> List[str]:
    """Standard prompts for Earth-observation captioning."""
    return [
        "Describe this satellite image in detail: terrain, land cover, water bodies, and human structures.",
        "List observable features: vegetation, bare soil, snow or ice, clouds, shadows, and roads or buildings.",
        "Provide a concise geographic scene description suitable for a remote-sensing catalog.",
    ]


def detailed_caption_prompts() -> List[str]:
    """Longer instructions when the model supports extended outputs."""
    return [
        (
            "You are an Earth observation analyst. Describe this satellite image: "
            "dominant land cover, topography cues, hydrology, and any linear features or settlements. "
            "Note uncertainty if the view is cloudy or ambiguous."
        ),
    ]


def merge_prompts(prompts: Sequence[str], separator: str = "\n\n") -> str:
    """Concatenate multiple prompts into one instruction (single forward pass)."""
    parts = [p.strip() for p in prompts if p and str(p).strip()]
    return separator.join(parts)


def get_prompts(
    style: str = "default",
    *,
    merge: bool = False,
    extra: Sequence[str] | None = None,
) -> List[str] | str:
    """
    Args:
        style: ``default`` or ``detailed``.
        merge: If True, return a single merged string; else a list of strings.
        extra: Additional prompt strings to append before optional merge.
    """
    if style == "detailed":
        base = detailed_caption_prompts()
    else:
        base = default_satellite_prompts()
    if extra:
        base = list(base) + list(extra)
    if merge:
        return merge_prompts(base)
    return base
