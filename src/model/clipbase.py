from pathlib import Path
from typing import List, Dict, Optional
from .evaluation import only_tok_metrics, recall_at_k
from torch import Tensor

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class CLIPBaseline(LightningModule):
    r"""CLIP Baseline model (only for inference).

    Args:
        clip_model: original CLIP model (required).
    """

    def __init__(
        self,
        clip_model,
    ) -> None:

        super().__init__()

        # adding the CLIP model
        self.clip_model, self.transform = clip_model

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_t_entity_latents = []
        self.validation_step_v_latents = []
        self.text_to_image_map = []
        self.image_to_text_map = []

        # if len(self.losses_weights) == 1 and 'with_entities' in self.losses_weights:
        #     self.losses_weights = {'with_entities': torch.tensor(1.0).cuda()}
    
    # Forward: image, text, facial_features -> text_features, image_features
    def forward(
        self,
        images: Tensor,
        texts: Tensor,
        entities: Optional[List] = None,
        single_caption: bool = True,
    ) -> List[Tensor]:

        bs, nc, np, _ = texts.shape
        texts = torch.flatten(texts, start_dim=0, end_dim=2)

        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts, normal_beh=True)

        # mean pooling over different prompts
        text_features = text_features.view(bs * nc, np, -1)
        text_features = text_features.mean(dim=1)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # TODO: ask about single_caption
        if not single_caption:
            image_features = image_features.unsqueeze(1).expand(-1,5,-1)
            image_features = torch.flatten(image_features, start_dim=0, end_dim=1)

        return text_features, image_features
    
    def _collate_general_retrieval_metrics(self, split="val"):
        assert split in ["val", "test"]
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        v_latents = torch.cat(self.validation_step_v_latents)

        text_to_image_map = torch.LongTensor(self.text_to_image_map).to(t_latents.device)
        image_to_text_map = torch.LongTensor(self.image_to_text_map).to(v_latents.device)

        # Compute the recall@k metrics
        k_vals = [1, 5, 10, 50]
        contrastive_metrics = recall_at_k(
            v_latents, t_latents, image_to_text_map, text_to_image_map, k_vals
        )
        contrastive_metrics['contrastive_sum'] = sum(contrastive_metrics.values())
        contrastive_metrics['contrastive_t2i_sum'] = sum([v for k, v in contrastive_metrics.items() if 't2i' in k])

        return contrastive_metrics
    
    def _collate_entity_retrieval_metrics(self, split="val"):
        assert split in ["val", "test"]
        # Compute entity metrics on the whole batch
        t_entity_latents = torch.cat(self.validation_step_t_entity_latents)
        v_latents = torch.cat(self.validation_step_v_latents)

        # Get the entities dataset
        k_vals = [1, 5, 10, 50]
        
        dataloaders = self.trainer.val_dataloaders if split == "val" else self.trainer.test_dataloaders
        dataloader = dataloaders[1] if isinstance(dataloaders, list) and len(dataloaders) == 2 else dataloaders
        dataset = dataloader.dataset

        entities_metrics = only_tok_metrics(
            v_latents, t_entity_latents, dataset, k_vals
        )
        entities_metrics['entities_sum'] = sum(entities_metrics.values())
        entities_metrics['entities_kmin_sum'] = sum([v for k, v in entities_metrics.items() if 'kmin' in k])

        return entities_metrics

    def test_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        # is the same logic of the validation step, except that (weird trick) we replace the dataloader_idx based on the
        # kind of dataloader we are using:
        # - dataloader_idx = 0 -> general retrieval dataloader
        # - dataloader_idx = 1 -> entities retrieval dataloader
        dataset = self.trainer.test_dataloaders.dataset
        dataloader_idx = 1 if dataset.only_TOK else 0

        self.validation_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:        
        bs = len(batch[0])
        with torch.no_grad():
            t_latents, v_latents = self.forward(*batch)
        t_latents = t_latents.cpu().float()
        v_latents = v_latents.cpu().float()

        # Store the latent vectors
        if dataloader_idx == 0:
            # store the 5-per-image text latents carrying the tok part + original caption
            self.validation_step_t_latents.append(t_latents)
        elif dataloader_idx == 1:
            self.validation_step_t_entity_latents.append(t_latents)

        # append image latents only once
        if (len(self.validation_step_t_latents) == 0) or (len(self.validation_step_t_entity_latents) == 0):
            self.validation_step_v_latents.append(v_latents)

        if dataloader_idx == 0:
            # update some data structures used to compute retrieval metrics
            # text has shape B x 5 x 77
            batch_size, captions_per_image, _, _ = batch[1].shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_index = (batch_idx * batch_size + i) * captions_per_image
                text_indices = list(range(text_index, text_index + captions_per_image))
                self.image_to_text_map.append(text_indices)

                # Each of the next captions_per_image text captions correspond to the same image
                self.text_to_image_map += [batch_idx * batch_size + i] * captions_per_image

    def on_validation_epoch_end(self):
        contrastive_metrics = self._collate_general_retrieval_metrics(split="val")
        entities_metrics = self._collate_entity_retrieval_metrics(split="val")

        metrics = {**contrastive_metrics, **entities_metrics}

        for metric_name in sorted(metrics):
            value = metrics[metric_name]
            self.log(
                f"val_{metric_name}_epoch",
                value,
                on_epoch=True,
                on_step=False,
            )

        self._clear_lists()

    def on_test_epoch_end(self) -> None:
        if self.trainer.test_dataloaders.dataset.only_TOK:
            metrics = self._collate_entity_retrieval_metrics(split="test")
        else:
            metrics = self._collate_general_retrieval_metrics(split="test")

        for metric_name in sorted(metrics):
            value = metrics[metric_name]
            self.log(
                f"{metric_name}",
                value,
                on_epoch=True,
                on_step=False,
            )

    def _clear_lists(self):
        self.validation_step_t_latents.clear()
        self.validation_step_v_latents.clear()
        self.validation_step_t_entity_latents.clear()
        self.text_to_image_map.clear()
        self.image_to_text_map.clear()