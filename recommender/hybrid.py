import numpy as np

class HybridRecommender:
    def __init__(self, collab_model, content_model):
        self.collab_model = collab_model
        self.content_model = content_model

    def recommend(self, user_id, n_items, alpha=0.6, top_n=20):
        collab_items = self.collab_model.recommend(user_id, n_items, top_n*2)
        if not collab_items:
            return []

        best_item = collab_items[0]
        content_items = self.content_model.recommend(best_item, top_n*2)
        combined = collab_items + content_items
        unique_items, counts = np.unique(combined, return_counts=True)
        scores = alpha * counts + (1 - alpha) * np.random.rand(len(unique_items))
        top_items = unique_items[np.argsort(-scores)][:top_n]
        return top_items.tolist()