"""Ознаки, які додаються всередині пайплайна моделі.

Функція лежить в окремому модулі, а не в ноутбуці: pickle/joblib зберігає не код
функції, а лише посилання «features.add_water_cement_ratio». Тому застосунок, який
завантажує модель, має мати цей самий модуль поруч.
"""
import pandas as pd

RAW_FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer",
                "coarse_agg", "fine_agg", "age"]


def add_water_cement_ratio(X: pd.DataFrame) -> pd.DataFrame:
    """Закон Абрамса: міцність бетону сильно залежить від відношення вода/цемент."""
    X = X.copy()
    X["water_cement_ratio"] = X["water"] / X["cement"]
    return X
