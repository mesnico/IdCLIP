import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import pytorch_lightning as pl
from pathlib import Path
import clip
from src.data.collate import CollateGeneralTextAndEntitiesText

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    
    # If dummy file "training_done" exists, skip training
    run_dir = Path(cfg.run_dir)
    if (run_dir / "training_done").exists():
        logger.info("Training already done. Skipping")
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

    # assert not (cfg.model.clip_model.shallow_visual_prompt_tokens != 0 and cfg.model.train_visual_encoder != 'shallow-vpt'), "shallow_visual_prompt_tokens != 0 is only compatible with train_visual_encoder='shallow_vpt'"

    logger.info("Loading the model")
    model = instantiate(cfg.model)
    transform = model.transform

    target_transform=lambda texts: clip.tokenize(texts)

    logger.info("Loading the dataloaders")
    train_dataset = instantiate(cfg.data.train.dataset, transform=transform, target_transform=target_transform)
    val_dataset_general_retrieval = instantiate(cfg.data.val_general_retrieval.dataset, transform=transform, target_transform=target_transform)
    val_dataset_entities_retrieval = instantiate(cfg.data.val_entities_retrieval.dataset, transform=transform, target_transform=target_transform)

    collator = CollateGeneralTextAndEntitiesText(cfg.data.train.default_collate)

    train_dataloader = instantiate(
        cfg.data.train.dataloader,
        batch_sampler={"dataset": train_dataset} if 'batch_sampler' in cfg.data.train.dataloader else None,
        dataset=train_dataset,
        collate_fn=collator,
    )

    val_dataloader_general_retrieval = instantiate(
        cfg.data.dataloader,
        dataset=val_dataset_general_retrieval,
        shuffle=False,
    )

    val_dataloader_entities_retrieval = instantiate(
        cfg.data.dataloader,
        dataset=val_dataset_entities_retrieval,
        shuffle=False,
    )

    val_dataloaders = [val_dataloader_general_retrieval, val_dataloader_entities_retrieval]

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloaders, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
