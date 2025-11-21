import gradio as gr
import pandas as pd
import os

from recommender.data_generator import generate_synthetic_data
from recommender.preprocessing import preprocess_products
from recommender.content_based import ContentRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.hybrid import HybridRecommender
from recommender.logger import setup_logger

logger = setup_logger()
logger.info("Запуск пайплайна гибридной рекомендательной системы")

data_dir = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(data_dir, exist_ok=True)

users_path = os.path.join(data_dir, "users.csv")
products_path = os.path.join(data_dir, "products.csv")
ratings_path = os.path.join(data_dir, "ratings.csv")

if all(os.path.exists(p) for p in [users_path, products_path, ratings_path]):
    logger.info("Найдены готовые данные, загружаем из /data...")
    users = pd.read_csv(users_path)
    products = pd.read_csv(products_path)
    ratings = pd.read_csv(ratings_path)
    logger.info(f"Загружено {len(users)} пользователей, {len(products)} продуктов, {len(ratings)} взаимодействий")
else:
    logger.warning("Данные не найдены в /data, генерируем синтетический набор...")
    users, products, ratings = generate_synthetic_data()
    logger.info(f"Сгенерировано {len(users)} пользователей, {len(products)} продуктов, {len(ratings)} взаимодействий")

logger.info("Предобработка данных (нормализация, one-hot)...")
prod_features = preprocess_products(products)
logger.info("Предобработка завершена")

logger.info("Обучение контентной модели...")
content_model = ContentRecommender(prod_features)
logger.info("Content-based модель готова")

logger.info("Обучение коллаборативной модели (LightFM)...")
collab_model = CollaborativeRecommender(users, products, ratings)
collab_model.train()
logger.info("Collaborative модель готова")

hybrid = HybridRecommender(collab_model, content_model)
logger.info("Гибридная модель инициализирована")

def recommend_interface(user_id):
    user_id = int(user_id)
    logger.info(f"Запрос рекомендаций для пользователя ID={user_id}")

    recs_collab = collab_model.recommend(user_id, n_items=len(products))
    recs_content_based = content_model.recommend(user_id)
    recs_hybrid = hybrid.recommend(user_id, n_items=len(products))
    logger.info(f"Content-based рекомендации: {recs_content_based[:20]}")
    logger.info(f"Collaborative рекомендации: {recs_collab[:20]}")
    logger.info(f"Hybrid рекомендации: {recs_hybrid[:20]}")

    df = products.loc[products["product_id"].isin(recs_hybrid)][
        ["product_id", "type", "interest_rate", "term_months", "min_income_required"]
    ]
    return df

demo = gr.Interface(
    fn=recommend_interface,
    inputs=gr.Number(label="User ID"),
    outputs=gr.Dataframe(label="Рекомендованные продукты"),
    title="Гибридная рекомендательная система для финсервиса",
    description="Введите ID пользователя (1–999), чтобы получить персональные рекомендации."
)

if __name__ == "__main__":
    logger.info("Запуск Gradio интерфейса...")
    demo.launch()