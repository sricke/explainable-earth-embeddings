import logging
from pathlib import Path


def setup_logging(run_name: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to {log_file}")


def build_run_name(args, dataset_name: str) -> tuple[str, str]:
    subsample = getattr(args, "train_subsample_size", None)
    p = [
        f"txt-{args.text_encoder}",
        f"loc-{args.location_encoder}",
        f"tproj-{args.text_projection}",
        f"lproj-{args.location_projection}",
        f"tft-{args.text_finetune_mode}",
        f"lft-{args.loc_finetune_mode}",
        *([ f"lora_r-{args.lora_rank}"] if getattr(args, "lora_rank", 0) and (args.text_finetune_mode == "lora" or args.loc_finetune_mode == "lora") else []),
        f"loss-{args.train_loss}",
        f"lr-{args.lr}",
        *([ f"sub-{subsample}"] if subsample is not None else []),
    ]
    run_tag = "__".join(str(x).strip().replace(" ", "-").replace("/", "_").replace("\\", "_") for x in p if x)
    return f"{dataset_name}__{run_tag}".strip().replace(" ", "-").replace("/", "_").replace("\\", "_"), run_tag
