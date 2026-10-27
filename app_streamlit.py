"""Крок 2 доповіді: веб-застосунок на Streamlit.

Запуск: streamlit run app_streamlit.py
Streamlit перезапускає весь цей скрипт згори донизу після кожної зміни повзунка,
тому модель завантажується через @st.cache_resource — один раз, а не щоразу.
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from features import RAW_FEATURES

LABELS = {
    "cement": "Цемент, кг/м³",
    "slag": "Доменний шлак, кг/м³",
    "fly_ash": "Зола-винесення, кг/м³",
    "water": "Вода, кг/м³",
    "superplasticizer": "Суперпластифікатор, кг/м³",
    "coarse_agg": "Щебінь (крупний заповнювач), кг/м³",
    "fine_agg": "Пісок (дрібний заповнювач), кг/м³",
    "age": "Вік бетону, днів",
}

st.set_page_config(page_title="Міцність бетону — демо ML", page_icon="🧱", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("model/concrete_model.joblib")
    with open("model/model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_data():
    df = pd.read_csv("data_concrete.csv")
    df.columns = RAW_FEATURES + ["strength"]
    test = pd.read_csv("model/test_examples.csv")      # зразки, яких модель не бачила
    return df.drop_duplicates().reset_index(drop=True), test


model, meta = load_model()
examples, test_examples = load_data()
ranges = meta["ranges"]

st.title("🧱 Прогноз міцності бетону")
st.write(
    "Модель із лабораторної роботи №1 (GradientBoosting) за складом суміші передбачає "
    "**міцність на стиск** у мегапаскалях. Змінюйте повзунки ліворуч — прогноз оновлюється одразу."
)
if meta["sklearn_version"] != sklearn.__version__:
    st.warning(f"Модель збережена в scikit-learn {meta['sklearn_version']}, а зараз встановлено "
               f"{sklearn.__version__}. Різні версії можуть давати інші результати.")

# --- Готові приклади з датасету -------------------------------------------------
with st.sidebar:
    st.header("Склад суміші")
    preset = st.selectbox(
        "Почати з прикладу з датасету",
        ["— власні значення —", "Звичайний бетон, 28 днів", "З суперпластифікатором, 28 днів",
         "Молодий бетон, 3 дні", "Старий бетон, 9–12 місяців"],
    )
    rules = {
        "Звичайний бетон, 28 днів": (test_examples["superplasticizer"] == 0) & (test_examples["age"] == 28),
        "З суперпластифікатором, 28 днів": (test_examples["superplasticizer"] > 8) & (test_examples["age"] == 28),
        "Молодий бетон, 3 дні": test_examples["age"] == 3,
        "Старий бетон, 9–12 місяців": test_examples["age"] >= 270,
    }
    example_row = test_examples[rules[preset]].iloc[0] if preset in rules else None

    values = {}
    for name in RAW_FEATURES:
        low, high, median = ranges[name]
        start = float(example_row[name]) if example_row is not None else median
        step = 1.0 if name == "age" else 0.5
        values[name] = st.slider(LABELS[name], min_value=float(low), max_value=float(high),
                                 value=float(start), step=step, key=f"{preset}-{name}")

row = pd.DataFrame([values], columns=RAW_FEATURES)
prediction = float(model.predict(row)[0])

# --- Результат -----------------------------------------------------------------
left, middle, right = st.columns(3)
left.metric("Прогноз міцності", f"{prediction:.1f} МПа")
middle.metric("Вода / цемент", f"{values['water'] / values['cement']:.2f}",
              help="Закон Абрамса: що менше води на одиницю цементу, то міцніший бетон")
share = float((examples["strength"] < prediction).mean() * 100)
right.metric("Міцніше, ніж у датасеті", f"{share:.0f}% зразків")

if example_row is not None:
    st.info(f"Це справжній зразок із **тестової** вибірки — модель його не бачила під час навчання. "
            f"Виміряна міцність **{example_row['strength']:.1f} МПа**, модель каже **{prediction:.1f} МПа** "
            f"(якщо ви не рухали повзунки).")

# --- Як набирає міцність саме ця суміш -----------------------------------------
st.subheader("Як ця суміш набирає міцність з віком")
ages = np.array([1, 3, 7, 14, 28, 56, 90, 180, 365])
curve = pd.concat([row.assign(age=a) for a in ages], ignore_index=True)
chart = pd.DataFrame({"вік, днів": ages, "міцність, МПа": model.predict(curve)}).set_index("вік, днів")
st.line_chart(chart)
st.caption("Модель бачила вимірювання лише в певні дні (3, 7, 28, 56, 90…), тому між ними крива — "
           "це інтерполяція, а не реальні вимірювання.")

# --- Попередження про екстраполяцію --------------------------------------------
wc = values["water"] / values["cement"]
wc_all = examples["water"] / examples["cement"]
if not wc_all.min() <= wc <= wc_all.max():
    st.warning(f"Відношення вода/цемент {wc:.2f} виходить за межі навчальних даних "
               f"({wc_all.min():.2f}–{wc_all.max():.2f}). Прогноз ненадійний: модель такого не бачила.")

# --- Прогноз для багатьох сумішей одразу ---------------------------------------
st.subheader("Прогноз для файлу CSV")
st.write("Завантажте CSV зі стовпцями: " + ", ".join(f"`{c}`" for c in RAW_FEATURES))
uploaded = st.file_uploader("CSV-файл", type="csv")
if uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        missing = [c for c in RAW_FEATURES if c not in batch.columns]
        if missing:
            st.error("У файлі бракує стовпців: " + ", ".join(missing))
        else:
            batch["predicted_strength"] = model.predict(batch[RAW_FEATURES]).round(2)
            st.dataframe(batch)
            st.download_button("Завантажити результат", batch.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")
    except (pd.errors.ParserError, ValueError) as error:
        st.error(f"Не вдалося прочитати файл: {error}")

st.divider()
st.caption(f"Модель: GradientBoosting у Pipeline (ознака вода/цемент → StandardScaler → модель). "
           f"Якість на тестовій вибірці: R² = {meta['metrics']['test_r2']}, "
           f"MAE = {meta['metrics']['test_mae']} МПа. Навчена {meta['trained_on']}, "
           f"scikit-learn {meta['sklearn_version']}, {meta['n_rows']} зразків.")
