import torch

class CollateGeneralTextAndEntitiesText():
    def __init__(self, default_collate=False):
        self.default_collate = default_collate

    def __call__(self, batch):
        bs = len(batch)
        bs_halved = bs // 2
        collated = {}

        # call the default collate function
        images, targets, features, targets_entities, targets_original, entity_ids = torch.utils.data._utils.collate.default_collate(batch)

        # explode into different splits
        if not self.default_collate:
            collated['with_entities'] = [
                images[:bs_halved],
                targets[:bs_halved],
                features[:bs_halved],
                entity_ids[:bs_halved],
            ]

            collated['only_entities'] = [
                images[bs_halved:],
                targets_entities[bs_halved:],
                features[bs_halved:],
                entity_ids[bs_halved:],
            ]

            collated['coco_original'] = [
                images[::2],
                targets_original[::2],
                features[::2],
                entity_ids[::2],
            ]

        else:
            # retain the original collate
            collated['with_entities'] = [
                images,
                targets,
                features,
                entity_ids,
            ]

        # if self.percentage_entities == 1.0:
        #     return images, targets_entities, features, entity_ids
        # elif self.percentage_entities > 0.0:
        #     bs = len(batch)
        #     bs_percent = int(bs * self.percentage_entities)

        #     # we take as targets half of targets and half of targets_entities
        #     targets = torch.cat([targets[:bs_percent], targets_entities[bs_percent:]], dim=0)

        return collated

        