import itertools
import json
import torch
import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
import random
import re
import csv
from collections import Counter

from torchvision.datasets import VisionDataset


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        features_json: str = None,
        face_swap_train: bool = False,
        single_caption: bool = False,
        normal_behavior: bool = False,
        templates: List[str] = None,
        templates_ensembling: bool = True,
        entity_expansion: str = None,
        key_words: List[str] = None,
        entities_to_names_file: str = None,
        use_original_names: bool = False,
        only_TOK: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids_tot = list(sorted(self.coco.imgs.keys())) # keys di imgs sono gli ID delle immagini presenti nel json sezione "images"
        #self.ids_tot_capt = list(sorted(self.coco.anns.keys()))
        # filtraggio con id delle immagini collezionate
        #print(type(self.ids_tot))
        self.face_swap_train = face_swap_train
        self.normal_behaviour = normal_behavior
        self.single_caption = single_caption
        self.only_TOK = only_TOK
        self.templates = templates
        self.entity_expansion = entity_expansion
        self.key_words = key_words

        if features_json is not None:
            with open(features_json, "r") as file_train:
                f = json.load(file_train)
            self.features = {key: torch.FloatTensor(value) for key, value in f.items()}
            if self.face_swap_train:
                self.ids = list(self.features.keys()) #************************************************#
            else:
                self.ids = list( set(list(self.features.keys())) & set(self.ids_tot))
        else:
            self.ids = self.ids_tot
            self.features = None

        if use_original_names:
            with open(entities_to_names_file, "r") as file:
                reader = csv.DictReader(file, skipinitialspace=True)
                self.entity_to_name = {row["Class_ID"]: row["Name"].replace('_', ' ') for row in reader}
        else:
            self.entity_to_name = None

        self.templates_ensembling = templates_ensembling
        #print(self.ids)

    def _load_image(self, id: int, entity = None) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"] # ritorna l'array di un singolo elemento, lo recupera e prende solo il nome dell'immagine
        #print(path)
        if entity is None:
            return Image.open(os.path.join(self.root, path)).convert("RGB") # caso in cui i nomi dei file non siano diversi da dagli id
        else:
            path_image_swap = path+'_'+entity+'.jpg'
            return Image.open(os.path.join(self.root, path_image_swap)).convert("RGB")
        
    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def return_ideal_k_map(self):
        k = None
        counter = None
        if self.face_swap_train and self.only_TOK:
            list_entities = [image_entity.split('_')[1] for image_entity in self.ids]
            counter = Counter(list_entities)
            max_entity = max(counter, key=counter.get)
            k = counter[max_entity]  

        return k, counter

    def caption_configuration(self, caption, templates=None, entity_expansion_template=None, entity_name=None):
        default_templates = [
            "An image with [ENTITY]. [CAPTION]",
            "An image with a person whose name is [ENTITY]. [CAPTION]",
            "[ENTITY]. [CAPTION]",
            "A picture of [ENTITY]. [CAPTION]",
            "[ENTITY] in the image. [CAPTION]",
            "There is [ENTITY] in the picture. [CAPTION]"
        ]

        assert not (any(['CAPTION' in t or 'C_CHUNK' in t for t in templates]) and caption is None)
        word_ref = None
        count = 0

        if self.key_words and caption is not None:
            target_to_analize = caption.lower()
            for word in self.key_words:
                pattern = fr'\b{word}\b'
                if re.search(pattern, target_to_analize):
                    count += 1
                    word_ref = word

        if word_ref is not None:
            if word_ref.capitalize() in caption:
                word_ref = word_ref.capitalize()
            elif word_ref.upper() in caption:
                word_ref = word_ref.upper()
            caption_split = caption.split(word_ref)

        templates_w_keyword = [t for t in templates if 'C_CHUNK_1' in t]
        templates_wo_keyword = [t for t in templates if 'C_CHUNK_1' not in t]
        new_captions = [c.replace('[CAPTION]', caption) if caption is not None else c for c in templates_wo_keyword]

        if word_ref is not None and count == 1 and len(caption_split) == 2:
            captions = [
                t
                .replace('[C_CHUNK_1]', caption_split[0].strip())
                .replace('[C_CHUNK_2]', caption_split[1].strip())
                .replace('[KEYWORD]', word_ref)
                for t in templates_w_keyword
            ]

            captions = [
                t
                .replace(' a [ENTITY]', ' [ENTITY]')
                .replace('A [ENTITY]', '[ENTITY]')
                .replace(' an [ENTITY]', ' [ENTITY]')
                .replace('An [ENTITY]', '[ENTITY]')
                for t in captions
            ]

            captions = [re.sub(fr',(?![\s\d])', ', ', t) if re.search(fr',(?!\s)', t) else t for t in captions]
            new_captions.extend(captions)
                
        elif (count != 1 or len(caption_split) != 2) and self.key_words is not None:
            # repeat some captions for the last 3 elements to have alsways the same number of elements in the list
            # otherwise it is difficult to batch them
            if len(new_captions) > 0:
                new_captions.extend(random.choices(new_captions, k=len(templates_w_keyword)))
            else:
                new_captions = random.choices(default_templates, k=len(templates_w_keyword))
                new_captions = [c.replace('[CAPTION]', caption) for c in new_captions]

        # substitute [ENTITY] with the entity_expansion
        split_captions = [c.split('[ENTITY]') if '[ENTITY]' in c else None for c in new_captions]
        new_captions = [
            entity_expansion_template
            .replace('[T_CHUNK_1]', sc[0])
            .replace('[T_CHUNK_2]', sc[1]) 
            .replace('[NAME]', entity_name)
            if sc is not None else c
            for sc, c in zip(split_captions, new_captions)
        ]

        assert not any(['[CAPTION]' in c for c in new_captions]), f"Something should not have happened."
        # assert not any([('[' in c) and (']' in c) for c in new_captions]), f"Error in caption configuration: {new_captions}"
        return new_captions
        

    def __getitem__(self, index: int, return_element:bool = False) -> Tuple[Any, Any, Any]: # aggiungere altro tipo, features delle immagini
        id = self.ids[index] # index = 0, 1, 2 ... *-*-* id = 139, ... (sono gli ID nella sezione "images" json)
        #print("index: "+str(index))
        #print("id55: "+str(id))
        #print(type(id))
        if self.only_TOK and return_element:
            return id.split('_')[1]
        
        if self.face_swap_train:
            list_id_entity = id.split('_')
            id_swap = int(list_id_entity[0])
            entity = list_id_entity[1]
            entity_name = self.entity_to_name[entity] if self.entity_to_name else None
            image = self._load_image(id_swap, entity)
            target_partial = self._load_target(id_swap)
        else:
            image = self._load_image(id)
            target_partial = self._load_target(id) # concatenare "Image with [TOK], ..."

        # some images have more than 5 captions, we take the first 5
        target_partial = target_partial[:5]

        list_features = self.features[id]
        if not self.single_caption:
            if self.templates_ensembling:
                target = [self.caption_configuration(str(target), entity_name=entity_name, templates=self.templates, entity_expansion_template=self.entity_expansion) for target in target_partial]
                list_features = list_features.repeat(5, 1, 1) # [n_caption, 1, 1] 
            else:
                target = [self.caption_configuration(str(target), entity_name=entity_name, templates=[template], entity_expansion_template=self.entity_expansion) for target, template in itertools.product(target_partial, self.templates)]
                list_features = list_features.repeat(5 * len(self.templates), 1, 1)
        else:
            # assert not self.templates_ensembling, "Single caption and ensembling templates are not compatible"
            target = [self.caption_configuration(None, entity_name=entity_name, templates=self.templates, entity_expansion_template=self.entity_expansion)]
            list_features = list_features.unsqueeze(0)

        if return_element and not self.face_swap_train:
            return image, target
        elif return_element and self.face_swap_train:
            return image, target, id.split('_')[1]
        
        if self.transforms is not None:
            image = self.transform(image)
            # flatten the target list using itertools
            num_captions = len(target)
            assert all(len(target[i]) == len(target[i+1]) for i in range(len(target)-1))

            num_prompts = len(target[0])
            target = list(itertools.chain(*target)) # flatten the list of lists
            target = self.target_transform(target)
            # reshape to original shape
            target = target.view(num_captions, num_prompts, -1)
            # image, target = self.transforms(image, target)

        if self.features is not None and self.normal_behaviour == False:
            return image, target, list_features
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.PILToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]


class CocoDetection_training(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        features_json: dict = None,
        face_swap_train: bool = False,
        mod_captions: bool = False,
        key_words: List[str] = None,
        entity_prompts: List[str] = ["An image with [TOK]"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.face_swap_train = face_swap_train
        self.dict_data_partial = list(sorted(self.coco.imgs.keys()))
        self.key_words = key_words
        self.mod_cap = mod_captions
        self.entity_prompts = entity_prompts
        ###***** DA SISTEMARE IL CODICE
        if features_json is None:
            # caso in cui devo allenare il modello con tutto coco e non devo utilizzare le features della faccia
            self.features = None
            for id_img in self.dict_data_partial:
                    ids_captions = self.coco.getAnnIds(id_img)
                    for id_caption in ids_captions:
                        self.dict_data.append({"id_image": id_img, "id_caption": id_caption})
        else:
            # in questo caso utilizzo le features della faccia
            with open(features_json, "r") as file_train:
                f = json.load(file_train)
            self.features = {key: torch.FloatTensor(value) for key, value in f.items()}
            self.dict_data = []
            if self.face_swap_train:
                #in questo caso creo l'indice basandomi sulla versione faceswap
                list_keys_json_features = list(self.features.keys()) #************************************************#
                print(len(list_keys_json_features))
                for key in list_keys_json_features:
                    id_image = int(key.split('_')[0])
                    ids_captions =  self.coco.getAnnIds(id_image)
                    if len(ids_captions) > 5:
                        #print("more than 5 captions: "+str(len(ids_captions)))
                        ids_captions = ids_captions[:5]
                    for id_caption in ids_captions:
                        self.dict_data.append({"id_image_entity": key, "id_caption":id_caption})

                    ## Continuare, fare considerazioni sulla tipologia di self.dict_data (dict or list)

            else:
                # in questo caso utilizzo la faccia ma non nella versione faceswap
                self.dict_data_img_ids = list(set(list(self.features.keys())) & set(self.dict_data_partial))
                for id_img in self.dict_data_img_ids:
                    ids_captions = self.coco.getAnnIds(id_img)
                    for id_caption in ids_captions:
                        self.dict_data.append({"id_image": id_img, "id_caption": id_caption})


    def _load_image(self, id: int, entity = None) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"] # ritorna l'array di un singolo elemento, lo recupera e prende solo il nome dell'immagine
        #print(path)
        if entity is None:
            return Image.open(os.path.join(self.root, path)).convert("RGB") # caso in cui i nomi dei file non siano diversi da dagli id
        else:
            path_image_swap = path+'_'+entity+'.jpg'
            return Image.open(os.path.join(self.root, path_image_swap)).convert("RGB")
        
    def _load_target(self, id: int) -> str:
        return self.coco.loadAnns(id)[0]['caption']
    

    def __getitem__(self, index: int, return_element:bool = False) -> Tuple[Any, Any, Any]: # *-*-*-*-
        id = self.dict_data[index] # index = 0, 1, 2 ... *-*-* id = 139
        
        if self.face_swap_train:
            image_entity_split = id['id_image_entity'].split('_')
            id_swap = int(image_entity_split[0])
            entity = image_entity_split[1]
            image = self._load_image(id_swap, entity)

            id_caption = id['id_caption']
            target_partial = self._load_target(id_caption)
        else:
            image = self._load_image(id['id_image'])
            target_partial = self._load_target(id['id_caption']) # concatenare "Image with [TOK], ..."
            entity = None
        
        if self.features is not None:
            if self.mod_cap and self.key_words is not None:
                word_ref = None
                target_to_analize = target_partial.lower()
                count = 0
                for word in self.key_words:
                    pattern = fr'\b{word}\b'
                    if re.search(pattern, target_to_analize):
                        count += 1
                        word_ref = word
                
                if word_ref is None or count > 1:
                    #print("Double subjects or not target word: ", target_partial)
                    target = [random.choice(self.entity_prompts) + ". " + str(target_partial)]
                else:
                    if word_ref.capitalize() in target_partial:
                        word_ref = word_ref.capitalize()
                    elif word_ref.upper() in target_partial:
                        word_ref = word_ref.upper()
                

                    caption_split = target_partial.split(word_ref)
                    if len(caption_split) != 2:
                        #print("Error on caption split: ", target_partial)
                        #print("Caption Split: ",caption_split, len(caption_split))
                        target = [random.choice(self.entity_prompts) + ". " + str(target_partial)]
                    else:
                        random_configuration_caption = random.randint(0,2)

                        match random_configuration_caption:
                            case 0:
                                target = [caption_split[0] + word_ref + " [TOK]"+caption_split[1]]
                            case 1:
                                target = [caption_split[0]+"[TOK] "+word_ref+caption_split[1]]
                            case 2:
                                target = [caption_split[0]+"[TOK]"+caption_split[1]]
                            case _:
                                target = [random.choice(self.entity_prompts) + ". " +str(target_partial)]

                        if re.search(fr',(?!\s)', target[0]):
                            target[0] = re.sub(fr',(?![\s\d])', ', ', target[0])
            else:
                target = [random.choice(self.entity_prompts) + ". " + str(target_partial)]

            target_only_entity = [random.choice(self.entity_prompts) + "."]
            if self.target_transform:
                target_only_entity = self.target_transform(target_only_entity)

            target_original = [target_partial]
            if self.target_transform:
                target_original = self.target_transform(target_original)
            
            if self.face_swap_train:
                list_features = self.features[id['id_image_entity']].unsqueeze(0) # possibile ripetizione delle features
            else:
                list_features = self.features[id['id_image']].unsqueeze(0)
            # if not self.single_caption:
            #     list_features = list_features.repeat(5,1,1) # [n_caption, 1, 1] 
            # else:
            #     list_features = list_features.unsqueeze(0)
        else:
            target = target_partial

        if return_element and not self.face_swap_train:
            return image, target
        elif return_element and self.face_swap_train:
            return image, target, id.split('_')[1]
        
        if self.transforms is not None:
            image, target = self.transforms(image, target) #*-*--*-*

        # artificially place the prompt dimension, used for prompt ensembling during validation
        target = target.unsqueeze(1)
        target_only_entity = target_only_entity.unsqueeze(1)
        target_original = target_original.unsqueeze(1)

        if self.features is not None:
            return image, target, list_features, target_only_entity, target_original, entity #*-*-*-*- 
        else:
            return image, target, target, target_only_entity, entity

    def __len__(self) -> int:
        return len(self.dict_data)


class CocoCaptions_training(CocoDetection_training):
    def _load_target(self, id: int) -> str:
        return super()._load_target(id)


if __name__ == '__main__':
    # test dataset
    from sampler import GeneralAndEntitiesPairsSampler
    import clip
    _, transform = clip.load("ViT-B/32")
    target_transform=lambda texts: clip.tokenize(texts)
    datadir = "../data_messina/DatiMaltese/"
    # dataset = CocoDetection_training(
    #     root=f"{datadir}/coco_dataset/coco_train_swapped",
    #     annFile=f"{datadir}/coco_dataset/annotations/captions_train2017.json",
    #     features_json=f"{datadir}/Visual_Name_Entity_Recognition/gender_faceswap_training_set/features_train_small_faceswap.json",
    #     transform=transform,
    #     target_transform=target_transform,
    #     face_swap_train=True,
    #     entity_prompts=[
    #        "An image with [TOK]",
    #        "An image containing [TOK]",
    #        "[TOK]",
    #        "A picture of [TOK]",
    #        "[TOK] in the image",
    #        "There is [TOK] in the picture",
    #     ],
    #     key_words=[
    #        "person",
    #        "man",
    #        "mother",
    #        "father",
    #        "dad",
    #        "mum",
    #        "woman",
    #        "child",
    #        "boy",
    #        "guy",
    #        "girl",
    #        "kid",
    #        "human",
    #        "adult",
    #        "male",
    #        "female",
    #        "lady",
    #        "teenager",
    #        "baby",
    #     ],
    #     mod_captions=True
    # )

    # batch_sampler = GeneralAndEntitiesPairsSampler(dataset, batch_size=32)
    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler)
    # for batch in loader:
    #     print(batch)

    dataset = CocoDetection(
        root=f"{datadir}/coco_dataset/coco_val_swap_new",
        annFile=f"{datadir}/coco_dataset/annotations/captions_val2017.json",
        features_json=f"{datadir}/Visual_Name_Entity_Recognition/gender_faceswap_new_val_test_set/features_new_val_small_faceswap.json",
        transform=transform,
        target_transform=target_transform,
        face_swap_train=True,
        entity_prompts=[
           "An image with [TOK]",
           "An image containing [TOK]",
           "[TOK]",
           "A picture of [TOK]",
           "[TOK] in the image",
           "There is [TOK] in the picture",
        ],
        only_TOK=True,
        single_caption=True
        # key_words=[
        #    "person",
        #    "man",
        #    "mother",
        #    "father",
        #    "dad",
        #    "mum",
        #    "woman",
        #    "child",
        #    "boy",
        #    "guy",
        #    "girl",
        #    "kid",
        #    "human",
        #    "adult",
        #    "male",
        #    "female",
        #    "lady",
        #    "teenager",
        #    "baby",
        # ],
        # mod_captions=True
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in dataloader:
        print(batch)