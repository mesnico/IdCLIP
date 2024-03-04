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

@hydra.main(version_base=None, config_path="configs", config_name="retrieval")
def retrieval(newcfg: DictConfig) -> None:
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt

    cfg = read_config(run_dir)
    model = instantiate(cfg.model)
    transform = model.transform
    target_transform=lambda texts: clip.tokenize(texts[:5])

    # load the checkpoint
    ckpt_path = glob.glob(os.path.join(run_dir, "logs/checkpoints", f"{ckpt_name}*.ckpt"))
    if len(ckpt_path) == 0:
        raise FileNotFoundError(f"Checkpoint not found in {run_dir}")
    ckpt_path = ckpt_path[0]
    # state_dict = torch.load(ckpt_path)["state_dict"]
    # model.load_state_dict(state_dict)
    # logger.info("Model loaded")

    pl.seed_everything(cfg.seed)

    test_dataset = instantiate(newcfg.data.test.dataset, transform=transform, target_transform=target_transform)

    test_dataloader = instantiate(
        cfg.data.test.dataloader,
        dataset=test_dataset,
        shuffle=False,
    )

    trainer = instantiate(newcfg.trainer)
    trainer.test(
        model, 
        ckpt_path=ckpt_path,
        dataloaders=test_dataloader,
        verbose=True
    )

if __name__ == "__main__":
    retrieval()