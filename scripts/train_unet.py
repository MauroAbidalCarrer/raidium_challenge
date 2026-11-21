import argparse

import torch

import configs as cfg
from raidium_challenge import dataset, training, models, metrics, timing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", nargs="?", help="Weights & Biases run name")
    parser.add_argument("tags", nargs="*", help="Additional Weights & Biases run tags")
    parser.add_argument("--ckpt", help="Checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ckpt:
        chkpt = torch.load(args.ckpt, weights_only=False)
        train_cfg = cfg.TrainingConfig(**chkpt["train_cfg"])
        model_cfg = cfg.ModelConfig(**chkpt["model_cfg"])
        model = models.mk_model_from_cfg(model_cfg)
        model.load_state_dict(chkpt["model"])
        model.cfg = model_cfg
        omptimizer_cfg = cfg.OPTIM_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)
        optimizer.load_state_dict(chkpt["optimizer"])
    else:
        train_cfg = cfg.TRAIN_CONFIGS["unet_training"]
        model_cfg = cfg.MODELS_CFGS["unet"]
        model = models.mk_model_from_cfg(model_cfg)
        omptimizer_cfg = cfg.OPTIM_CFGS["unet"]
        optimizer = training.mk_optimizer(model, omptimizer_cfg)

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    criterion = metrics.SegmentationLoss(train_cfg)
    data_loaders = dataset.mk_segmentation_data_loaders(train_cfg)
    wandb_run = training.wandb_init(
        model_cfg,
        train_cfg,
        omptimizer_cfg,
        name=args.run_name,
        tags=cfg.WANDB_RUN_TAGS["unet"] + args.tags,
        group="manual_training",
    )
    trainer = training.Trainer(
        model,
        train_cfg, 
        optimizer,
        lr_scheduler,
        wandb_run,
    )
    if args.ckpt:
        print("setting epoch, step and training samples seen")
        n_btaches_per_epoch = len(data_loaders["train"])
        samples_seen = chkpt["training_samples_seen"]
        trainer.training_samples_seen = samples_seen
        n_samples_per_epoch = n_btaches_per_epoch * train_cfg.batch_size
        trainer.epoch = chkpt.get("epoch", int(samples_seen // n_samples_per_epoch))
        trainer.step = chkpt.get("step", int(samples_seen // train_cfg.batch_size))
    trainer.train_model(
        data_loaders,
        criterion,
        #"checkpoints/unet/{wandb_run_name}/unet_epoch_{epoch}.pt",
    )
    timing.print_time_dict()


if __name__ == "__main__":
    main()
