import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

class AdvancedHybridRecommender:
    def __init__(self, collab_model, content_model, ratings_df, 
                 weight_strategy='adaptive', use_popularity=False):
        self.collab_model = collab_model
        self.content_model = content_model
        self.ratings_df = ratings_df
        self.weight_strategy = weight_strategy
        self.use_popularity = use_popularity
        
        # Рассчитаем популярность продуктов
        self.popularity_scores = self._calculate_popularity()
        
    def _calculate_popularity(self):
        if self.ratings_df is not None:
            popularity = self.ratings_df['product_id'].value_counts().to_dict()
            # Нормализуем
            max_pop = max(popularity.values()) if popularity else 1
            return {k: v/max_pop for k, v in popularity.items()}
        return {}
    
    def _get_user_confidence(self, user_id):
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if len(user_ratings) < 5:
            # Мало данных - больше доверяем контентной модели
            return 0.3  # 30% collaborative, 70% content
        elif len(user_ratings) < 20:
            # Среднее количество данных
            return 0.5  # 50/50
        else:
            # Много данных - больше доверяем коллаборативной
            return 0.7  # 70% collaborative, 30% content
    
    def recommend(self, user_id, n_items=10, alpha=None):
        user_id = int(user_id)
        
        # 1. Получаем рекомендации от обеих моделей
        collab_items = self.collab_model.recommend(user_id, n_items*3)
        collab_scores = {item: 1/(i+1) for i, item in enumerate(collab_items)}
        
        # 2. Контентные рекомендации
        content_items = []
        if self.ratings_df is not None:
            user_history = self.ratings_df[
                self.ratings_df['user_id'] == user_id
            ]['product_id'].unique()
            
            if len(user_history) > 0:
                all_content_items = []
                for product in user_history[:5]:  # Берем только последние 5
                    items = self.content_model.recommend(product, top_n=n_items*2)
                    all_content_items.extend(items)
                
                # Подсчитываем частоту
                from collections import Counter
                item_counts = Counter(all_content_items)
                content_items = [item for item, _ in item_counts.most_common(n_items*3)]
        
        content_scores = {item: 1/(i+1) for i, item in enumerate(content_items)}
        
        # 3. Определяем веса
        if alpha is None:
            if self.weight_strategy == 'adaptive':
                alpha = self._get_user_confidence(user_id)  # вес для collaborative
            else:
                alpha = 0.6  # фиксированный вес
        
        # 4. Объединяем с весами
        combined_scores = defaultdict(float)
        
        for item, score in collab_scores.items():
            combined_scores[item] += alpha * score
        
        for item, score in content_scores.items():
            combined_scores[item] += (1 - alpha) * score
        
        # 5. Учитываем популярность (опционально)
        if self.use_popularity:
            for item in combined_scores:
                pop_score = self.popularity_scores.get(item, 0.1)
                combined_scores[item] *= (0.8 + 0.2 * pop_score)  # небольшой бонус популярным
        
        # 6. Сортируем и возвращаем
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        recommended = [item for item, _ in sorted_items[:n_items]]
        
        # 7. Если мало рекомендаций, добавляем популярные
        if len(recommended) < n_items:
            popular_items = list(self.popularity_scores.keys())[:n_items*2]
            for item in popular_items:
                if item not in recommended:
                    recommended.append(item)
                if len(recommended) >= n_items:
                    break
        
        return recommended[:n_items]