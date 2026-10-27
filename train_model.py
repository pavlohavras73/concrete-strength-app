"""Крок 1 доповіді: «з ноутбука — у файл».

Повторює фінальну модель ЛР1 (міцність бетону, GradientBoosting) і зберігає
весь пайплайн (ознаки → масштабування → модель) в один файл через joblib.
Запуск: python train_model.py
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import RAW_FEATURES, add_water_cement_ratio

STUDENT_NO = 6  # той самий random_state, що й у ЛР1

raw = pd.read_csv("data_concrete.csv")
raw.columns = RAW_FEATURES + ["strength"]
df = raw.drop_duplicates().reset_index(drop=True)          # як у ЛР1: 25 дублікатів

X, y = df[RAW_FEATURES], df["strength"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=STUDENT_NO)

pipeline = Pipeline([
    ("features", FunctionTransformer(add_water_cement_ratio)),   # нова ознака — всередині моделі
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                                        subsample=0.8, random_state=STUDENT_NO)),
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
r2, mae = r2_score(y_test, pred), mean_absolute_error(y_test, pred)
print(f"Test R2 = {r2:.4f}, MAE = {mae:.4f} МПа  (train {len(X_train)}, test {len(X_test)})")

joblib.dump(pipeline, "model/concrete_model.joblib")

# Тестова вибірка окремо: застосунок показує приклади лише з неї — модель їх не бачила
X_test.assign(strength=y_test).to_csv("model/test_examples.csv", index=False)

# Метадані поруч із моделлю: застосунок бере з них межі повзунків і перевіряє версію sklearn
meta = {
    "trained_on": date.today().isoformat(),
    "sklearn_version": sklearn.__version__,
    "features": RAW_FEATURES,
    "ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in RAW_FEATURES},
    "metrics": {"test_r2": round(r2, 4), "test_mae": round(mae, 4)},
    "n_rows": int(len(df)),
}
with open("model/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Збережено: model/concrete_model.joblib, model/model_meta.json, model/test_examples.csv")

# Перевірка: завантажена модель дає ті самі відповіді
loaded = joblib.load("model/concrete_model.joblib")
print("Завантажена модель відповідає збереженій:", np.allclose(loaded.predict(X_test), pred))
