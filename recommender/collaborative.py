from lightfm import LightFM
from lightfm.data import Dataset

class CollaborativeRecommender:
    def __init__(self, users, products, ratings):
        self.dataset = Dataset()
        self.dataset.fit(users["user_id"], products["product_id"])
        (self.interactions, _) = self.dataset.build_interactions(
            [(int(u), int(i), float(r)) for u, i, r in ratings.values]
        )
        self.model = None

    def train(self, epochs=10):
        self.model = LightFM(loss="warp")
        self.model.fit(self.interactions, epochs=epochs, num_threads=2)

    def recommend(self, user_id, n_items, top_n=20):
        scores = self.model.predict(user_id, list(range(n_items)))
        top_items = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
        return top_items