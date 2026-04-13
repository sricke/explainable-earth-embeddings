import argparse
from pathlib import Path

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
    # Encoders
    parser.add_argument('--text_encoder', type=str, required=True, help="'open_clip' | 'geoclip'")
    parser.add_argument('--location_encoder', type=str, required=True, help="'satclip' | 'geoclip'")
    parser.add_argument('--finetune_mode', type=str, default='only_proj', help="'all' | 'only_proj' | 'none'")
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
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    # Checkpointing
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Checkpoint path. Auto-derived from run config if not set.')
    parser.add_argument('--resume_from', type=str)
    # Logging
    parser.add_argument('--wandb_project', type=str, default='explainable-earth-embeddings')
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def build_dataloaders(args):
    train_dataset = GeoTextDataset(root=args.dataset_path, split='train')
    val_dataset = GeoTextDataset(root=args.dataset_path, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataloader


def build_model(args, device):
    text_encoder = make_text_encoder(
        args.text_encoder, args.text_projection, args.shared_dim, args.finetune_mode,
        num_hidden_layers=args.text_proj_hidden_layers, num_hidden_features=args.text_proj_hidden_features,
    )
    location_encoder = make_location_encoder(
        args.location_encoder, args.location_projection, args.shared_dim,
        num_hidden_layers=args.loc_proj_hidden_layers, num_hidden_features=args.loc_proj_hidden_features,
    )
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
    logit_scale.data = ckpt['logit_scale']
    print(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf'))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    train_dataloader, val_dataloader = build_dataloaders(args)
    model = build_model(args, device)

    logit_scale = torch.nn.Parameter(torch.tensor(1.0 / args.logit_scale_temp).log())
    criterion = make_loss(args.train_loss, logit_scale, args.lambda_alignment, args.sigma)
    optimizer = build_optimizer(args, model, logit_scale)

    start_epoch = 0
    best_val_loss = float('inf')
    if args.model_save_path:
        save_path = Path(args.model_save_path)
    elif not args.no_wandb:
        save_path = Path(f"checkpoints/{wandb.run.name}/best.pt")
    else:
        ds = Path(args.dataset_path).name
        save_path = Path(f"checkpoints/{ds}__{args.train_loss}__loc-{args.location_projection}__text-{args.text_projection}/best.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        start_epoch, best_val_loss = load_checkpoint(args.resume_from, model, optimizer, logit_scale, device)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, epoch, device)
        val_loss = val_epoch(val_dataloader, model, criterion, epoch, device)

        print(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_path, epoch, model, optimizer, logit_scale, best_val_loss, args)
            print(f"  Saved checkpoint (val={val_loss:.4f})")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
