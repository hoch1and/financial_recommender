import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, products_features: pd.DataFrame):
        self.sim_df = self._build_similarity_matrix(products_features)

    def _build_similarity_matrix(self, features: pd.DataFrame):
        sim_matrix = cosine_similarity(features.drop(columns=["product_id"]))
        return pd.DataFrame(sim_matrix, index=features["product_id"], columns=features["product_id"])

    def recommend(self, product_id, top_n=5):
        if product_id not in self.sim_df.columns:
            return []
        sim_scores = self.sim_df[product_id].sort_values(ascending=False)
        return sim_scores.iloc[1:top_n+1].index.tolist()