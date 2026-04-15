import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from dataset import GeoTextDataset
from models.model import TextLocationModel
from models.utils import make_text_encoder, make_location_encoder
from loss import make_loss
from train import train_epoch
from eval import val_epoch
from utils import EarlyStopping


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    p = [f"txt-{args.text_encoder}", f"loc-{args.location_encoder}", f"tproj-{args.text_projection}", f"lproj-{args.location_projection}", f"loss-{args.train_loss}", f"lr-{args.lr}"]
    run_tag = "__".join(str(x).strip().replace(" ", "-").replace("/", "_").replace("\\", "_") for x in p if x)
    return f"{dataset_name}__{run_tag}".strip().replace(" ", "-").replace("/", "_").replace("\\", "_"), run_tag


def get_args():
    # Check for --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        return argparse.Namespace(**cfg)

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--precomputed_text_embeddings', type=bool, default=True)
    parser.add_argument('--precomputed_location_embeddings', type=bool, default=True)
    # Encoders
    parser.add_argument('--text_encoder', type=str, required=True, help="'open_clip' | 'geoclip'")
    parser.add_argument('--location_encoder', type=str, required=True, help="'satclip' | 'geoclip'")
    parser.add_argument('--text_finetune_mode', type=str, default='only_proj', help="'all' | 'only_proj'")
    parser.add_argument('--loc_finetune_mode', type=str, default='only_proj', help="'only_proj'")
    # Projections
    parser.add_argument('--text_projection', type=str, default='linear', help="'linear' | 'mlp'")
    parser.add_argument('--text_proj_hidden_layers', type=int, default=1)
    parser.add_argument('--text_proj_hidden_features', type=int, default=512)
    parser.add_argument('--location_projection', type=str, default='linear', help="'none' | 'linear' | 'mlp'")
    parser.add_argument('--loc_proj_hidden_layers', type=int, default=1)
    parser.add_argument('--loc_proj_hidden_features', type=int, default=512)
    parser.add_argument('--shared_dim', type=int, default=256)
    # Loss
    parser.add_argument('--train_loss', type=str, required=True, help="'clip' | 'mse'")
    parser.add_argument('--lambda_alignment', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--logit_scale_temp', type=float, default=0.07)
    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--text_nonlinearity', type=str, default=None, help="'relu' | 'sine'")
    parser.add_argument('--loc_nonlinearity', type=str, default=None, help="'relu' | 'sine'")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    # Checkpointing
    parser.add_argument(
        '--model_save_path', type=str, default=None,
        help='Base directory for checkpoints (default: ~/outputs/explainable-earth-embeddings/<dataset_name>); '
             'checkpoints go in <base>/<run-subdir>/best.pt. File logs use ./logs in the cwd.',
    )
    parser.add_argument('--resume_from', type=str)
    # Logging
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B team or username (optional; defaults to logged-in default).')
    parser.add_argument('--wandb_project', type=str, default='explainable-earth-embeddings')
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def build_dataloaders(args):
    train_dataset = GeoTextDataset(
        root=args.dataset_path,
        split='train',
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
    )
    val_dataset = GeoTextDataset(
        root=args.dataset_path,
        split='val',
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    worker_init = lambda wid: set_seed(args.seed + wid)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=worker_init)
    logger.info(f"Loaded dataset from {args.dataset_path}")
    logger.info(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}")
    return train_dataloader, val_dataloader


def build_model(args, device):
    text_encoder = make_text_encoder(
        args.text_encoder, args.text_projection, args.shared_dim, args.text_finetune_mode,
        num_hidden_layers=args.text_proj_hidden_layers, num_hidden_features=args.text_proj_hidden_features,
        nonlinearity=args.text_nonlinearity,
        precomputed=args.precomputed_text_embeddings,
    )
    location_encoder = make_location_encoder(
        args.location_encoder, args.location_projection, args.shared_dim, args.loc_finetune_mode,
        num_hidden_layers=args.loc_proj_hidden_layers, num_hidden_features=args.loc_proj_hidden_features,
        nonlinearity=args.loc_nonlinearity,
        precomputed=args.precomputed_location_embeddings,
    )
    logger.info(f"Using text encoder={args.text_encoder}, location encoder={args.location_encoder}")
    return TextLocationModel(text_encoder=text_encoder, location_encoder=location_encoder).to(device)


def build_optimizer(args, model, logit_scale):
    return torch.optim.AdamW(
        list(model.parameters()) + [logit_scale],
        lr=args.lr, weight_decay=args.weight_decay,
    )


def save_checkpoint(path, epoch, model, optimizer, logit_scale, best_val_loss, args):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'logit_scale': logit_scale.data,
        'best_val_loss': best_val_loss,
        'args': vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, logit_scale, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    logit_scale.data = ckpt['logit_scale'].to(device)
    model.to(device)
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)
    logger.info(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf'))


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset_name = Path(args.dataset_path).expanduser().resolve().name
    default_wandb_name, default_run_tag = build_run_name(args, dataset_name)
    desired_wandb_name = str(args.wandb_run_name).strip().replace(" ", "-").replace("/", "_").replace("\\", "_") if getattr(args, "wandb_run_name", None) else default_wandb_name

    # Skip if this run already completed all epochs.
    _early_checkpoint_base = (Path(args.model_save_path).expanduser().resolve() if getattr(args, "model_save_path", None) else (Path.home() / "outputs" / "explainable-earth-embeddings" / dataset_name).resolve())
    if (_early_checkpoint_base / default_run_tag / "done").exists():
        logger.info(f"Run {default_run_tag} already completed. Skipping.")
        return

    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=desired_wandb_name,
            config=vars(args),
        )
        # W&B may alter the final name; keep filesystem tags safe.
        wandb_name = wandb.run.name if wandb.run is not None else desired_wandb_name
        run_tag = wandb_name.replace(f"{dataset_name}__", "", 1).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")
    else:
        run_tag = default_run_tag

    run_tag = str(run_tag).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")
    setup_logging(f"{dataset_name}__{run_tag}".strip().replace(" ", "-").replace("/", "_").replace("\\", "_"), Path("logs"))

    if getattr(args, "model_save_path", None):
        checkpoint_base = Path(args.model_save_path).expanduser().resolve()
    else:
        checkpoint_base = (Path.home() / "outputs" / "explainable-earth-embeddings" / dataset_name).resolve()
    checkpoint_dir = checkpoint_base / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best.pt"
    config_save_path = checkpoint_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Checkpoints directory: {checkpoint_dir}")

    args.precomputed_text_embeddings = args.precomputed_text_embeddings and args.text_finetune_mode != "all"
    args.precomputed_location_embeddings = args.precomputed_location_embeddings and args.loc_finetune_mode != "all"

    train_dataloader, val_dataloader = build_dataloaders(args)
    model = build_model(args, device)

    logit_scale = torch.nn.Parameter(torch.tensor(1.0 / args.logit_scale_temp, device=device).log())
    criterion = make_loss(args.train_loss, logit_scale, args.lambda_alignment, args.sigma)
    optimizer = build_optimizer(args, model, logit_scale)
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    logger.info(f"Loss: {args.train_loss}")
    logger.info(f"LR={args.lr}, weight_decay={args.weight_decay}")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume_from:
        start_epoch, best_val_loss = load_checkpoint(args.resume_from, model, optimizer, logit_scale, device)
        early_stopping.load_best(best_val_loss)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, logger)
        val_loss = val_epoch(val_dataloader, model, criterion, epoch, device, logger)

        logger.info(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_path, epoch, model, optimizer, logit_scale, best_val_loss, args)
            logger.info(f"Saved checkpoint (val={val_loss:.4f})")

        if early_stopping(val_loss):
            print("Early Stopping")
            (checkpoint_dir / "done").touch()
            logger.info("Training stopped early. Wrote 'done' marker.")
            break
    else:
        # Loop exited without break → all epochs completed.
        (checkpoint_dir / "done").touch()
        logger.info("Training completed all epochs. Wrote 'done' marker.")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
