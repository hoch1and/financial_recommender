import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class SVDRecommender:
    def __init__(self, ratings_df, n_factors=20):
        self.ratings_df = ratings_df
        self.n_factors = n_factors
        self.user_map = {}
        self.item_map = {}
        self.svd = None
        
    def fit(self):
        # Создаем матрицу пользователь-товар
        users = self.ratings_df['user_id'].unique()
        items = self.ratings_df['product_id'].unique()
        
        # Маппинг индексов
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: idx for idx, i in enumerate(items)}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        # Создаем разреженную матрицу
        n_users = len(users)
        n_items = len(items)
        
        ratings = np.zeros((n_users, n_items))
        
        for _, row in self.ratings_df.iterrows():
            u_idx = self.user_map[row['user_id']]
            i_idx = self.item_map[row['product_id']]
            ratings[u_idx, i_idx] = row['rating']
        
        # Применяем SVD
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(ratings)
        self.item_factors = self.svd.components_.T
        
    def recommend(self, user_id, n_items=10):
        if user_id not in self.user_map:
            # Холодный старт - возвращаем популярные
            return self._get_popular_items(n_items)
        
        u_idx = self.user_map[user_id]
        user_vector = self.user_factors[u_idx]
        
        # Предсказываем рейтинги для всех товаров
        scores = np.dot(self.item_factors, user_vector)
        
        # Сортируем по убыванию
        top_indices = np.argsort(scores)[::-1][:n_items]
        
        # Возвращаем ID товаров
        return [self.reverse_item_map[idx] for idx in top_indices]
    
    def _get_popular_items(self, n_items):
        # Возвращаем популярные товары
        popular = self.ratings_df['product_id'].value_counts().index.tolist()
        return popular[:n_items]