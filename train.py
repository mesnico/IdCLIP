import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import pytorch_lightning as pl
from pathlib import Path
import clip

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    # Skip if training already done
    ckpt_path = Path(cfg.run_dir) / "logs/checkpoints"
    
    if ckpt_path.exists():
        latest_ckpt = list(ckpt_path.glob("latest-epoch*.ckpt"))
        if len(latest_ckpt) != 0:
            # extract the number after the = sign in the filename
            epochs = max([int(c.stem.split("=")[1]) + 1 for c in latest_ckpt])

            if epochs >= cfg.trainer.max_epochs:
                logger.info(
                    f"Training already done. Latest checkpoint found: {latest_ckpt}"
                )
                return

    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = instantiate(cfg.model)
    transform = model.transform

    target_transform=lambda texts: clip.tokenize(texts[:5])

    logger.info("Loading the dataloaders")
    train_dataset = instantiate(cfg.data.train, transform=transform, target_transform=target_transform)
    val_dataset_general_retrieval = instantiate(cfg.data.val_general_retrieval, transform=transform, target_transform=target_transform)
    val_dataset_entities_retrieval = instantiate(cfg.data.val_entities_retrieval, transform=transform, target_transform=target_transform)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        shuffle=True,
    )

    val_dataloader_general_retrieval = instantiate(
        cfg.dataloader,
        dataset=val_dataset_general_retrieval,
        shuffle=False,
    )

    val_dataloader_entities_retrieval = instantiate(
        cfg.dataloader,
        dataset=val_dataset_entities_retrieval,
        shuffle=False,
    )

    val_dataloaders = [val_dataloader_general_retrieval, val_dataloader_entities_retrieval]

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloaders, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
