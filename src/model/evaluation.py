from typing import List, Dict, Optional
import numpy as np
import torch

# cross-modal recall metrics
def recall_at_k(image_encodings, text_encodings, image_to_text_map, text_to_image_map, k_vals: List[int]):
    print("Encoding all data...")
 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    # dist_matrix = dist_matrix

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)

    metrics = {}

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        metrics[f't2i-r@{k}'] = num_correct / num_text


    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).to(image_encodings.device)

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        metrics[f'i2t-r@{k}'] = num_correct / num_im

    print("Done.")
    return metrics


# map and recall metrics for entities only (Am I retrieving the right entity?)
def only_tok_metrics(image_encodings, text_encodings, dataset, k_vals: List[int]):

    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()
    # print(dist_matrix)

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    # print(inds)
    recalls_queries_entities = recall_at_k_only_TOK(inds, k_vals, dataset, k_min=True)
    recalls_queries_entities_only_relevant = recall_at_k_only_TOK(inds, k_vals, dataset, k_min=False)

    mean_avg_prec, mean_avg_prec_per_ent = map_at_k_only_TOK(inds, dataset) 
    metrics = {
        'mAP': mean_avg_prec,
        **recalls_queries_entities,
        **recalls_queries_entities_only_relevant,
    }
    
    return metrics


def recall_at_k_only_TOK(ranks, k_vals, dataset, k_min=False):
    max_relevant_entity, dict_count_entities = dataset.return_ideal_k_map()
    query_entities_gt = [image_entity.split('_')[1] for image_entity in dataset.ids]
    all_recall = {}
    for k in k_vals:
        topk = ranks[:, :k]
        topk_list = topk.tolist()
        already_analized = []

        mean_recall = []
        for i in range(len(query_entities_gt)):
            correct = 0
            entity_query = query_entities_gt[i]
            num_relevant_images_entity = dict_count_entities[entity_query]
            if entity_query not in already_analized:
                for j in range(len(topk_list[i])):
                    entity_retrieved = dataset.__getitem__(topk_list[i][j], True)
                    correct += entity_query == entity_retrieved
                
                if k_min:    
                    recall_query = correct / min(k, num_relevant_images_entity)
                else:
                    recall_query = correct/num_relevant_images_entity

                already_analized.append(entity_query)
                mean_recall.append(recall_query)
        # print(len(mean_recall))
        k_min_str = '-kmin' if k_min else ''
        all_recall[f'entity{k_min_str}-r@{k}'] = np.average(mean_recall)

    return all_recall

        
def map_at_k_only_TOK(ranks, dataset, k=10):
    max_relevant_entity, dict_count_entities = dataset.return_ideal_k_map()
    query_entities_gt = [image_entity.split('_')[1] for image_entity in dataset.ids]
    topk = ranks[:,:k]
    topk_list = topk.tolist()
    map_per_entity = {}
    mean_avg_prec_all_result = []
    already_analized = []
    for i in range(len(query_entities_gt)):
        ap = 0
        correct = 0
        entity_query = query_entities_gt[i]
        
        if entity_query not in already_analized:
            for j in range(k):
                entity_retrieved = dataset.__getitem__(topk_list[i][j], True)
                if entity_retrieved == entity_query:
                    # print(i)
                    correct += 1
                    ap = ap + (correct/(j+1))

            map_per_entity[entity_query] = ap/k
            mean_avg_prec_all_result.append(ap/k)
            already_analized.append(entity_query)

    return np.average(mean_avg_prec_all_result), map_per_entity
