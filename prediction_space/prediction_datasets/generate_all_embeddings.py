"""Generate SatCLIP / GeoCLIP / AEF embedding caches for every prediction task."""

from __future__ import annotations

import argparse

from . import air_temp, biome, countries, ecoregion, elevation, nightlights, pop_density, treecover

# (folder / cache key, module) — must match each module's ``get_task_embeddings(..., "<name>", ...)``.
REGRESSION_TASKS: list[tuple[str, object]] = [
    ("elevation", elevation),
    ("pop_density", pop_density),
    ("nightlights", nightlights),
    ("treecover", treecover),
    ("air_temp", air_temp),
]

CLASSIFICATION_TASKS: list[tuple[str, object]] = [
    ("biome", biome),
    ("ecoregion", ecoregion),
    ("countries", countries),
]

ALL_TASKS: list[tuple[str, object]] = REGRESSION_TASKS + CLASSIFICATION_TASKS

BACKENDS = ("satclip", "geoclip", "aef")


def generate_all_embeddings(
    *,
    device: str = "cuda",
    force: bool = False,
    backends: tuple[str, ...] | None = BACKENDS,
    tasks: list[tuple[str, object]] | None = ALL_TASKS,
) -> None:
    """
    Call :func:`load_embeddings_and_values` for each task and backend so
    ``~/data/prediction_tasks/<task>/embeddings_<backend>.pt`` are created or refreshed.
    """

    for task_key, mod in tasks:
        load_fn = getattr(mod, "load_embeddings_and_values", None)
        if load_fn is None:
            raise AttributeError(f"{task_key}: module has no load_embeddings_and_values")
        for backend in backends:
            print(f"{task_key} / {backend} …", flush=True)
            load_fn(backend=backend, device=device, force=force)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate embedding .pt caches for all prediction tasks.")
    p.add_argument("--device", default="cuda", help="Device passed to embedding generators (default: cuda).")
    p.add_argument("--force", action="store_true", help="Regenerate even if cache exists.")
    p.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help=f"Comma-separated backends (default: {','.join(BACKENDS)}).",
    )
    p.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task keys (e.g. elevation,biome) or 'all' (default).",
    )
    args = p.parse_args()
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    if args.tasks.strip().lower() == "all":
        tasks = ALL_TASKS
    else:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        by_key = dict(ALL_TASKS)
        missing = set(task_list) - by_key.keys()
        if missing:
            raise SystemExit(f"Unknown task keys: {sorted(missing)}. Known: {sorted(by_key.keys())}")
        tasks = [(k, by_key[k]) for k in task_list]
    generate_all_embeddings(device=args.device, force=args.force, backends=backends, tasks=tasks)


if __name__ == "__main__":
    main()
