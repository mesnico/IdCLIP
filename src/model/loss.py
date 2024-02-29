import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

class InfoNCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, im, s, **kwargs):
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * im @ s.t()
        logits_per_text = logits_per_image.t()

        # compute bidirectional CE loss
        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
        loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        return loss
    
class InfoNCELossEntityAware(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, im, s, entities=None, **kwargs):
        bs = im.shape[0]
        # if entities is given, the GT matrix is non diagonal and instead filled with correct correspondences between entities
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * im @ s.t()
        logits_per_text = logits_per_image.t()

        # compute bidirectional CE loss
        if entities is None:
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)

            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            entities_id_map = {v: i for i, v in enumerate(set(entities))}
            entities_ids = [entities_id_map[m] for m in entities]

            entities_ids = torch.LongTensor(entities_ids).to(im.device)
            mask = entities_ids.unsqueeze(1).expand(-1, bs) == entities_ids.unsqueeze(0).expand(bs, -1)
            mask = mask.float()
            s_distribution = F.softmax(mask, dim=1)
            img_distribution = F.softmax(mask, dim=0)

            loss = (
                F.cross_entropy(logits_per_image, s_distribution) +
                F.cross_entropy(logits_per_text, img_distribution)
            ) / 2

        return loss