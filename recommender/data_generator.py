import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_users=1000, n_products=100):
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)

    users = pd.DataFrame({
        "user_id": range(n_users),
        "age": np.random.randint(18, 65, n_users),
        "income": np.random.randint(30000, 200000, n_users)
    })

    product_types = ["loan", "credit_card", "insurance", "deposit"]
    products = pd.DataFrame({
        "product_id": range(n_products),
        "type": np.random.choice(product_types, n_products),
        "interest_rate": np.random.uniform(3, 15, n_products),
        "term_months": np.random.choice([6, 12, 24, 36, 48], n_products),
        "min_income_required": np.random.randint(20000, 150000, n_products)
    })

    ratings = pd.DataFrame({
        "user_id": np.random.randint(0, n_users, n_users * 10),
        "product_id": np.random.randint(0, n_products, n_users * 10),
        "rating": np.random.choice([1, 2, 3, 4, 5], n_users * 10)
    })

    users.to_csv(os.path.join(data_dir, "users.csv"), index=False)
    products.to_csv(os.path.join(data_dir, "products.csv"), index=False)
    ratings.to_csv(os.path.join(data_dir, "ratings.csv"), index=False)

    return users, products, ratings