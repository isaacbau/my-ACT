#!/usr/bin/env python

import argparse
import os
from datetime import datetime
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ACT import ACT
from configuration_act import ACTConfig
from act_types import FeatureType, PolicyFeature

try:
    import wandb
except Exception:
    wandb = None


def build_config(state_dim: int, action_dim: int, img_h: int, img_w: int, chunk_size: int) -> ACTConfig:
    return ACTConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
            "observation.images.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
            "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_h, img_w)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        pretrained_backbone_weights=None,
    )


def make_batch(batch_size: int, state_dim: int, action_dim: int, img_h: int, img_w: int, chunk_size: int):
    batch = {
        "observation.state": torch.randn(batch_size, state_dim),
        "observation.images.main": torch.randn(batch_size, 3, img_h, img_w),
        "observation.images.left_wrist": torch.randn(batch_size, 3, img_h, img_w),
        "observation.images.right_wrist": torch.randn(batch_size, 3, img_h, img_w),
        "action": torch.randn(batch_size, chunk_size, action_dim),
        "action_is_pad": torch.zeros(batch_size, chunk_size, dtype=torch.bool),
    }
    batch["observation.images"] = [
        batch["observation.images.main"],
        batch["observation.images.left_wrist"],
        batch["observation.images.right_wrist"],
    ]
    return batch


def compute_loss(config: ACTConfig, batch, actions, mu, log_sigma_x2):
    l1_loss = (torch.abs(batch["action"] - actions) * ~batch["action_is_pad"].unsqueeze(-1)).mean()

    if mu is None or log_sigma_x2 is None:
        mean_kld = torch.tensor(0.0, device=actions.device)
    else:
        mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()

    loss = l1_loss + mean_kld * config.kl_weight

    return loss, l1_loss, mean_kld


def evaluate(model, config, args):
    model.eval()
    with torch.no_grad():
        batch = make_batch(
            args.batch_size,
            args.state_dim,
            args.action_dim,
            args.img_h,
            args.img_w,
            args.chunk_size,
        )
        actions, (mu, log_sigma_x2) = model(batch)
        loss, l1_loss, mean_kld = compute_loss(config, batch, actions, mu, log_sigma_x2)
    model.train()
    return loss.item(), l1_loss.item(), mean_kld.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--img-h", type=int, default=224)
    parser.add_argument("--img-w", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-dir", type=str, default="evals")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="act")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d-%H%M")

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    args.eval_dir = os.path.join(args.eval_dir, args.run_name)

    config = build_config(args.state_dim, args.action_dim, args.img_h, args.img_w, args.chunk_size)

    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("wandb not installed; continuing without logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config={
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_size": args.batch_size,
                "chunk_size": args.chunk_size,
                "state_dim": args.state_dim,
                "action_dim": args.action_dim,
                "img_h": args.img_h,
                "img_w": args.img_w,
                "lr": args.lr,
                "save_every": args.save_every,
                "eval_every": args.eval_every,
                "dim_model": config.dim_model,
                "n_encoder_layers": config.n_encoder_layers,
                "n_decoder_layers": config.n_decoder_layers,
                "n_vae_encoder_layers": config.n_vae_encoder_layers,
                "latent_dim": config.latent_dim,
                "vision_backbone": config.vision_backbone,
                "pretrained_backbone_weights": config.pretrained_backbone_weights,
                },
            )

    print("ACTConfig")
    print(f"  chunk_size={config.chunk_size}")
    print(f"  n_action_steps={config.n_action_steps}")
    print(f"  dim_model={config.dim_model}")
    print(f"  n_encoder_layers={config.n_encoder_layers}")
    print(f"  n_decoder_layers={config.n_decoder_layers}")
    print(f"  n_vae_encoder_layers={config.n_vae_encoder_layers}")
    print(f"  latent_dim={config.latent_dim}")
    print(f"  vision_backbone={config.vision_backbone}")
    print(f"  pretrained_backbone_weights={config.pretrained_backbone_weights}")

    model = ACT(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\nModel params")
    print(f"  total_params={sum(p.numel() for p in model.parameters())}")

    print("\nTraining")
    print(f"  epochs={args.epochs}")
    print(f"  steps_per_epoch={args.steps_per_epoch}")
    print(f"  save_every={args.save_every}")
    print(f"  eval_every={args.eval_every}")

    if args.save_every > 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.eval_every > 0:
        os.makedirs(args.eval_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            batch = make_batch(
                args.batch_size,
                args.state_dim,
                args.action_dim,
                args.img_h,
                args.img_w,
                args.chunk_size,
            )

            # Model predictions
            actions, (mu, log_sigma_x2) = model(batch)

            # Loss calculation
            loss, l1_loss, mean_kld = compute_loss(config, batch, actions, mu, log_sigma_x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(
                f"  epoch={epoch} step={step + 1}/{args.steps_per_epoch} "
                f"loss={loss.item():.4f} l1={l1_loss.item():.4f} kld={mean_kld.item():.4f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/l1": l1_loss.item(),
                        "train/kld": mean_kld.item(),
                        "epoch": epoch,
                        "step": (epoch - 1) * args.steps_per_epoch + step + 1,
                    }
                )

        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"  epoch={epoch} avg_loss={avg_loss:.4f}")
        if wandb_run is not None:
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch})

        # Save checkpoints
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = f"{args.checkpoint_dir}/act_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_path)
            print(f"  saved checkpoint: {ckpt_path}")

        # Save evals
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_loss, eval_l1, eval_kld = evaluate(model, config, args)
            eval_path = f"{args.eval_dir}/eval_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                    "eval_l1": eval_l1,
                    "eval_kld": eval_kld,
                },
                eval_path,
            )
            print(
                f"  eval epoch={epoch} loss={eval_loss:.4f} l1={eval_l1:.4f} kld={eval_kld:.4f}"
            )
            print(f"  saved eval: {eval_path}")
            if wandb_run is not None:
                wandb.log(
                    {
                        "eval/loss": eval_loss,
                        "eval/l1": eval_l1,
                        "eval/kld": eval_kld,
                        "epoch": epoch,
                    }
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
