"""Etapa 4 - Construção, treino e avaliação dos dois algoritmos concorrentes.

Cada candidato é um Pipeline completo (pré-processamento + modelo) treinado
com as MESMAS features eleitas, o MESMO treino e avaliado na MESMA validação
temporal. O Pipeline serializado carrega consigo imputador, escalonador e a
lista de features, garantindo previsões consistentes no notebook 05.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessamento import construir_preprocessador

ALGORITMOS = {
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
}


def construir_pipeline(features: list[str], algoritmo: str, params: dict) -> Pipeline:
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo não suportado: {algoritmo}")
    return Pipeline([
        ("preprocessamento", construir_preprocessador(features)),
        ("modelo", ALGORITMOS[algoritmo](**params)),
    ])


def avaliar(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)  # demanda não pode ser negativa
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prever(pipeline: Pipeline, X) -> np.ndarray:
    return np.clip(pipeline.predict(X), 0, None)


def treinar_e_avaliar(features, algoritmo, params, X_treino, y_treino, X_val, y_val) -> tuple[Pipeline, dict]:
    pipe = construir_pipeline(features, algoritmo, params)
    t0 = time.perf_counter()
    pipe.fit(X_treino[features], y_treino)
    duracao = time.perf_counter() - t0
    metr_val = avaliar(y_val, pipe.predict(X_val[features]))
    metr_treino = avaliar(y_treino, pipe.predict(X_treino[features]))
    metricas = {f"val_{k}": v for k, v in metr_val.items()}
    metricas.update({f"treino_{k}": v for k, v in metr_treino.items()})
    metricas["tempo_treino_s"] = duracao
    return pipe, metricas
