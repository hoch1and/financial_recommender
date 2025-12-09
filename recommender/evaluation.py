import numpy as np
import pandas as pd
import time

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    return len(set(recommended) & relevant) / k

def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    return len(set(recommended) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    dcg = 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_model(model, test_df, k=10):
    precisions, recalls, ndcgs = [], [], []

    start_time = time.time()

    for user_id in test_df["user_id"].unique():
        relevant = set(
            test_df[test_df["user_id"] == user_id]["product_id"]
        )

        if not relevant:
            continue

        recommended = model.recommend(user_id, k)

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    elapsed_time = time.time() - start_time

    return {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs),
        "Inference Time (s)": elapsed_time
    }

def catalog_coverage(model, users, all_items, k=10):
    recommended_items = set()

    for user_id in users:
        recommended_items.update(model.recommend(user_id, k))

    return len(recommended_items) / len(all_items)

def diversity(recommended_lists, all_items):
    if not recommended_lists:
        return 0.0
    
    pairwise_diversity = []
    for i in range(len(recommended_lists)):
        for j in range(i+1, len(recommended_lists)):
            set_i = set(recommended_lists[i])
            set_j = set(recommended_lists[j])
            
            if len(set_i | set_j) > 0:
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                pairwise_diversity.append(1 - jaccard)  # 1 - сходство = разнообразие
    
    return np.mean(pairwise_diversity) if pairwise_diversity else 0.0

def novelty(model, users, k=10, popularity_counts=None):
    if popularity_counts is None:
        return 0.0
    
    novelty_scores = []
    for user_id in users:
        recommendations = model.recommend(user_id, k)
        rec_popularity = [popularity_counts.get(item, 0) for item in recommendations]
        
        if rec_popularity:
            novelty_scores.append(1 - np.mean(rec_popularity))
    
    return np.mean(novelty_scores) if novelty_scores else 0.0