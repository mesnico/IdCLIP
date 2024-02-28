import torch
import more_itertools
import itertools
import random
from operator import add

class GeneralAndEntitiesPairsSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        entities_and_ids = [(i, v['id_image_entity'].split('_')[1]) for i, v in enumerate(dataset.dict_data)]
        entities_and_ids = itertools.groupby(entities_and_ids, key=lambda x: x[1])

        # materialize entities_and_ids
        self.entities_and_ids = [(k, list(v)) for k, v in entities_and_ids]

        # sort the entities_and_ids by the entity id
        # entities_and_ids = sorted(entities_and_ids, key=lambda x: x[1])

    def __iter__(self):
        # 1. The first half of the batch is for the general retrieval dataset
        # create a random permutation of indexes
        random_idxs = torch.randperm(len(self.dataset)).tolist()
        # create batches using more_itertools.chunked
        random_idxs = more_itertools.chunked(random_idxs, self.batch_size)

        # 2. The second half of the batch is for the entities retrieval (we force same entities in different scenarios to be similar)
        random.shuffle(self.entities_and_ids)
        # flatten self.entities_and_ids
        entity_and_ids = itertools.chain(*[v[1] for v in self.entities_and_ids])
        entity_ids, _ = more_itertools.unzip(entity_and_ids)
        entity_ids = more_itertools.chunked(entity_ids, self.batch_size)

        # we form a batch by taking half elements from the random_idxs and half from the entity_ids
        indexes = itertools.starmap(add, zip(random_idxs, entity_ids))

        yield from indexes

    def __len__(self):
        return len(self.dataset) // self.batch_size