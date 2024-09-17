from pathlib import Path
from typing import List, Dict, Optional
from .evaluation import only_tok_metrics, recall_at_k
from torch import Tensor
import clip

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from .clipbase import CLIPBaseline

class IdCLIP(CLIPBaseline):
    r"""IDCLIP model.

    Args:
        clip_model: original CLIP model (required).
        translator_model: model performing face-features to text-token translation (required).
        contrastive_loss: contrastive loss function (required).
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        translator: nn.Module,
        loss: nn.Module,
        lr: float = 1e-4,
        train_visual_encoder: str = None,
        train_text_encoder: str = None,
        encoders_lr: float = 1e-6,
        training_setup: Optional[Dict] = None,
    ) -> None:

        super().__init__(clip_model)
        # adding the contrastive loss
        self.loss_fn = loss

        # adding the CLIP model
        self.clip_model, self.transform = clip_model
        self.tokenizer = tokenizer if tokenizer is not None else clip.tokenize

        # adding the translator model
        self.translator_model = translator

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_t_entity_latents = []
        self.validation_step_v_latents = []
        self.text_to_image_map = []
        self.image_to_text_map = []

        self.lr = lr
        self.encoders_lr = encoders_lr
        self.train_visual_encoder = train_visual_encoder
        self.train_text_encoder = train_text_encoder
        self.training_setup = training_setup
        self.losses_weights = {
            k.replace('_weight', ''): v # nn.Parameter(torch.log(torch.tensor([v]).cuda())) 
            for k, v in training_setup.items() 
            if k.endswith('_weight') and training_setup[k.split('_weight')[0]]
        }
        # if len(self.losses_weights) == 1 and 'with_entities' in self.losses_weights:
        #     self.losses_weights = {'with_entities': torch.tensor(1.0).cuda()}

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        # Forward pass

        loss_dict = {}
        
        if self.training_setup['with_entities']:
            text_features, image_features = self.forward(
                *batch['with_entities']
            )

            # Compute the loss
            loss_dict['with_entities'] = self.loss_fn(
                image_features, text_features, entities=None
            )

        if self.training_setup['only_entities']:
            text_features, image_features = self.forward(
                *batch['only_entities']
            )

            # Compute the loss
            loss_dict['only_entities'] = self.loss_fn(
                image_features, text_features, entities=batch['only_entities'][3]
            )

        if self.training_setup['coco_original']:
            raise NotImplementedError
            text_features, image_features = self.forward(
                *batch['coco_original']
            )

            # Compute the loss
            loss_dict['coco_original'] = self.loss_fn(
                image_features, text_features, entities=None
            )

        loss_dict['loss'] = sum(
            self.losses_weights[k] * loss_dict[k] for k in loss_dict
        )

        if return_all:
            return loss_dict, image_features, text_features
        return loss_dict
    
    # Forward: image, text, facial_features -> text_features, image_features
    def forward(
        self,
        images: Tensor,
        texts: Tensor,
        facial_features: Tensor = None,
        entities: Optional[List] = None,
        single_caption: bool = True,
    ) -> List[Tensor]:

        bs, nc, np, _ = texts.shape
        texts = torch.flatten(texts, start_dim=0, end_dim=2)
        
        with torch.set_grad_enabled(self.train_visual_encoder is not None):
            image_features = self.clip_model.encode_image(images)

        if facial_features is None:
            text_features = self.clip_model.encode_text(texts)
        else:
            facial_features = torch.flatten(facial_features, start_dim=0, end_dim=1)
            translated_features = self.translator_model(facial_features)
            
            translated_features = translated_features.unsqueeze(1).expand(-1, np, -1, -1).flatten(start_dim=0, end_dim=1)
            text_features = self.clip_model.encode_text(texts, translated_features)

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
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch['with_entities'][0])
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

    def on_train_end(self) -> None:
        # if number of epochs matches the max_epochs, then training has surely ended
        if self.current_epoch >= self.trainer.max_epochs - 1:
            run_path = Path(self.trainer.logger.log_dir).parent
            # touch a file to signal that training has ended
            with open(run_path / "training_done", "w") as f:
                f.write("")

    # handle the optimizer
    def configure_optimizers(self):
        parameters = [{'params':self.translator_model.parameters(), 'lr': self.lr}]
        if self.train_visual_encoder == "finetune":
            parameters += [{'params':self.clip_model.visual.parameters(), 'lr': self.encoders_lr}]
        elif self.train_visual_encoder == "shallow-vpt":
            parameters += [{'params':self.clip_model.visual.shallow_visual_prompt_tokens, 'lr': self.encoders_lr}]
            
        if self.train_text_encoder == "finetune":
            parameters += [{'params':self.clip_model.transformer.parameters(), 'lr': self.encoders_lr}]
        elif self.train_text_encoder == "shallow-tpt":
            parameters += [{'params':self.clip_model.shallow_text_prompt_tokens, 'lr': self.encoders_lr}]

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
        )
        return optimizer