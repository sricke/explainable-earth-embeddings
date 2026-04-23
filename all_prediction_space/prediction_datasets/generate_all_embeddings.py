"""Generate SatCLIP / GeoCLIP / AEF embedding caches for every prediction task."""

from __future__ import annotations

import argparse

from .datasets import ALL_TASKS, load_dataset

BACKENDS = ("satclip", "geoclip", "aef")


def generate_all_embeddings(
    *,
    device: str = "cuda",
    force: bool = False,
    backends: tuple[str, ...] = BACKENDS,
    tasks: list[str] | None = None,
) -> None:
    """Write ``~/data/prediction_tasks/<task>/embeddings_<backend>.pt`` for each task/backend."""
    for task in (tasks or ALL_TASKS):
        for backend in backends:
            print(f"{task} / {backend} …", flush=True)
            load_dataset(task, backend, device=device, force=force)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate embedding .pt caches for all prediction tasks.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--backends", default=",".join(BACKENDS))
    p.add_argument("--tasks", default="all", help="Comma-separated task keys or 'all'.")
    args = p.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    tasks = None if args.tasks.strip().lower() == "all" else [
        t.strip() for t in args.tasks.split(",") if t.strip()
    ]
    if tasks:
        unknown = set(tasks) - set(ALL_TASKS)
        if unknown:
            raise SystemExit(f"Unknown tasks: {sorted(unknown)}. Known: {sorted(ALL_TASKS)}")

    generate_all_embeddings(device=args.device, force=args.force, backends=backends, tasks=tasks)


if __name__ == "__main__":
    main()
