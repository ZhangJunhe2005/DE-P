import os
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from policy.dep_trainer import DATASET_MODES, FREEZE_POLICIES, DepTrainer
from policy.backbone_variant import BACKBONE_VARIANTS, resolve_backbone_variant



def configure_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    parser.add_argument("--backbone-variant", choices=BACKBONE_VARIANTS, default=None,
                        help="override config/traj_opt.yaml backbone_variant")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit legacy/corrected checkpoint path")
    parser.add_argument("--dataset-mode", choices=DATASET_MODES, default="static")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to run; intentionally defaults to one")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--dynamic-data-root", type=Path, default=None)
    parser.add_argument("--dynamic-context-source", choices=("ground_truth", "estimated"),
                        default=None)
    parser.add_argument("--dynamic-loss-enabled", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--dynamic-loss-weight", type=float, default=None)
    parser.add_argument("--freeze-policy", choices=FREEZE_POLICIES, default="none")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="0 disables clipping; default comes from traj_opt.yaml")
    parser.add_argument("--resume", type=Path, default=None,
                        help="full training checkpoint; incompatible with --checkpoint")
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def resolve_checkpoint_path(pretrained, trial, epoch, log_dir, backbone_variant=None):
    """Resolve an explicitly requested DEP checkpoint or fail loudly."""
    if not pretrained:
        return None
    variant = resolve_backbone_variant(backbone_variant)
    directory = f"DEP_{trial}" if variant == "legacy" else f"DEP_corrected_{trial}"
    checkpoint = (Path(log_dir) / directory / f"epoch{epoch}.pth").resolve()
    print("Requested checkpoint:", checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Requested pretrained checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def resolve_explicit_checkpoint_path(checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    print("Requested checkpoint:", path)
    if not path.is_file():
        raise FileNotFoundError(f"Requested checkpoint does not exist: {path}")
    return str(path)


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(args.random_seed)
    backbone_variant = resolve_backbone_variant(args.backbone_variant)
    print(f"Backbone variant: {backbone_variant}")

    # save the configuration and other files
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    if args.checkpoint is not None and args.pretrained:
        raise ValueError("Use either --checkpoint or --pretrained/--trial/--epoch, not both")
    if args.resume is not None and (args.checkpoint is not None or args.pretrained):
        raise ValueError("Use --resume by itself; it restores weights and optimizer state")
    checkpoint_path = (
        resolve_explicit_checkpoint_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_checkpoint_path(
            args.pretrained, args.trial, args.epoch, log_dir, backbone_variant
        )
    )

    trainer = DepTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss_weight=[1.0, 1.0],
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
        backbone_variant=backbone_variant,
        dataset_mode=args.dataset_mode,
        dynamic_data_root=args.dynamic_data_root,
        dynamic_context_source=args.dynamic_context_source,
        dynamic_loss_enabled=args.dynamic_loss_enabled,
        dynamic_loss_weight=args.dynamic_loss_weight,
        freeze_policy=args.freeze_policy,
        max_grad_norm=args.max_grad_norm,
        resume=args.resume,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
    )

    trainer.train(epoch=args.epochs, save_interval=args.save_interval)

    print("Run DE-P Finish!")
