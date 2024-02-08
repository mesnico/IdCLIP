from typing import List, Dict, Optional
from .evaluation import only_tok_metrics, recall_at_k
from torch import Tensor

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class IdCLIP(LightningModule):
    r"""IDCLIP model.

    Args:
        clip_model: original CLIP model (required).
        translator_model: model performing face-features to text-token translation (required).
        contrastive_loss: contrastive loss function (required).
    """

    def __init__(
        self,
        clip_model,
        translator: nn.Module,
        contrastive_loss: nn.Module,
        lr: float = 1e-4,
        train_visual_encoder: bool = False,
        train_text_encoder: bool = False,
        encoders_lr: float = 1e-6,
    ) -> None:

        super().__init__()
        # adding the contrastive loss
        self.contrastive_loss_fn = contrastive_loss

        # adding the CLIP model
        self.clip_model, self.transform = clip_model

        # adding the translator model
        self.translator_model = translator

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_v_latents = []
        self.text_to_image_map = []
        self.image_to_text_map = []

        self.lr = lr
        self.encoder_lr = encoders_lr
        self.train_visual_encoder = train_visual_encoder
        self.train_text_encoder = train_text_encoder

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        # Forward pass
        text_features, image_features = self.forward(
            *batch
        )

        # Compute the loss
        contrastive_loss = self.contrastive_loss_fn(
            image_features, text_features
        )

        # Add the logits to the loss dict
        loss_dict = {'contrastive': contrastive_loss}

        loss_dict['loss'] = sum(loss_dict.values())

        if return_all:
            return loss_dict, image_features, text_features
        return loss_dict
    
    # Forward: image, text, facial_features -> text_features, image_features
    def forward(
        self,
        images: Tensor,
        texts: Tensor,
        facial_features: Tensor,
        single_caption: bool = True,
    ) -> List[Tensor]:
        
        facial_features = torch.flatten(facial_features, start_dim=0, end_dim=1)
        translated_features = self.translator_model(facial_features)

        texts = torch.flatten(texts, start_dim=0, end_dim=1)
        
        with torch.set_grad_enabled(self.train_visual_encoder):
            image_features = self.clip_model.encode_image(images)

        if facial_features is None:
            text_features = self.clip_model.encode_text(texts)
        else:
            text_features = self.clip_model.encode_text(texts, translated_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # TODO: ask about single_caption
        if not single_caption:
            image_features = image_features.unsqueeze(1).expand(-1,5,-1)
            image_features = torch.flatten(image_features, start_dim=0, end_dim=1)

        return text_features, image_features
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch[0])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        if dataloader_idx != 0:
            return
        
        bs = len(batch[0])
        t_latents, v_latents = self.forward(*batch)
        # t_latents = t_latents.cpu()
        # v_latents = v_latents.cpu()

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_v_latents.append(v_latents)

        # text has shape B x 5 x 77
        batch_size, captions_per_image, _ = batch[1].shape

        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_index = (batch_idx * batch_size + i) * captions_per_image
            text_indices = list(range(text_index, text_index + captions_per_image))
            self.image_to_text_map.append(text_indices)

            # Each of the next captions_per_image text captions correspond to the same image
            self.text_to_image_map += [batch_idx * batch_size + i] * captions_per_image

        # for loss_name in sorted(losses):
        #     loss_val = losses[loss_name]
        #     self.log(
        #         f"val_{loss_name}",
        #         loss_val,
        #         on_epoch=True,
        #         on_step=True,
        #         batch_size=bs,
        #     )

        # return losses["loss"]

    def on_validation_epoch_end(self):
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

        # Get the entities dataset
        val_dataset = self.trainer.val_dataloaders[1].dataset
        entities_metrics = only_tok_metrics(
            v_latents, t_latents, val_dataset, k_vals
        )

        metrics = {**contrastive_metrics, **entities_metrics}
        metrics['contrastive_sum'] = sum(contrastive_metrics.values())
        metrics['entities_sum'] = sum(entities_metrics.values())

        for metric_name in sorted(metrics):
            metric_val = metrics[metric_name]
            self.log(
                f"val_{metric_name}_epoch",
                metric_val,
                on_epoch=True,
                on_step=False,
            )

        self.validation_step_t_latents.clear()
        self.validation_step_v_latents.clear()
        self.text_to_image_map.clear()
        self.image_to_text_map.clear()

    # handle the optimizer
    def configure_optimizers(self):
        parameters = [{'params':self.translator_model.parameters(), 'lr': self.lr}]
        if self.train_visual_encoder:
            parameters += [{'params':self.clip_model.visual.parameters(), 'lr': self.encoders_lr}]
        if self.train_text_encoder:
            parameters += [{'params':self.clip_model.transformer.parameters(), 'lr': self.encoders_lr}]

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
        )
        return optimizer