import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_products(products: pd.DataFrame):
    prod_features = pd.get_dummies(products, columns=["type"], drop_first=True)
    scaler = MinMaxScaler()
    prod_features[["interest_rate", "term_months", "min_income_required"]] = scaler.fit_transform(
        prod_features[["interest_rate", "term_months", "min_income_required"]]
    )
    return prod_features