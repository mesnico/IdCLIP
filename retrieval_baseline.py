import os
from omegaconf import DictConfig
import logging
import hydra
import glob
import clip
import pytorch_lightning as pl
from hydra.utils import instantiate

from src.config import read_config

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="retrieval_baseline")
def retrieval(newcfg: DictConfig) -> None:
    model = instantiate(newcfg.model)
    transform = model.transform
    target_transform=lambda texts: clip.tokenize(texts)

    pl.seed_everything(newcfg.seed)

    test_dataset = instantiate(newcfg.data.test.dataset, transform=transform, target_transform=target_transform)

    test_dataloader = instantiate(
        newcfg.data.test.dataloader,
        dataset=test_dataset,
        shuffle=False,
    )

    trainer = instantiate(newcfg.trainer)
    trainer.test(
        model,
        dataloaders=test_dataloader,
        verbose=True
    )

if __name__ == "__main__":
    retrieval()