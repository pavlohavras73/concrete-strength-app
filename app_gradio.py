"""Той самий прогноз на Gradio — для порівняння зі Streamlit.

Запуск: python app_gradio.py   (відкрийте http://127.0.0.1:7860)
Gradio сам будує форму з опису входів і виходу: пишемо лише функцію predict.
"""
import json

import gradio as gr
import joblib
import pandas as pd

from features import RAW_FEATURES

model = joblib.load("model/concrete_model.joblib")
with open("model/model_meta.json", encoding="utf-8") as f:
    ranges = json.load(f)["ranges"]


def predict(*values):
    row = pd.DataFrame([values], columns=RAW_FEATURES)
    return round(float(model.predict(row)[0]), 1)


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Slider(low, high, value=median, label=name) for name, (low, high, median) in ranges.items()],
    outputs=gr.Number(label="Міцність, МПа"),
    title="Прогноз міцності бетону (ЛР1)",
    description="Модель GradientBoosting із лабораторної роботи №1.",
)

if __name__ == "__main__":
    demo.launch()
