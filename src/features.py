"""Etapa 3 - Três métodos de seleção/importância e votação de features.

Todos os métodos são ajustados SOMENTE no conjunto de treino do ciclo, após o
mesmo pré-processamento (imputação + StandardScaler) ajustado nesse treino.

Regra de votação (definida em config.yaml antes da comparação):
  1. Cada método vota nas suas top-k features (k = 12).
  2. É eleita a feature com >= min_votos (2 de 3).
  3. Se menos de min_features (8) forem eleitas, completa-se com as melhores
     colocadas no ranking médio ponderado (pesos em config.yaml) entre as não
     eleitas. Empates no ranking médio são resolvidos pelo nº de votos e,
     depois, pela ordem alfabética (determinístico).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Ridge

from src.preprocessamento import construir_preprocessador


def _ranking(valores: pd.Series, maior_melhor: bool = True) -> pd.Series:
    """Ranking 1..n (1 = melhor) com desempate determinístico pelo nome."""
    ordem = sorted(valores.index, key=lambda f: ((-valores[f] if maior_melhor else valores[f]), f))
    return pd.Series({f: i + 1 for i, f in enumerate(ordem)})


def metodo_selectkbest(Xt: np.ndarray, y: np.ndarray, features: list[str], k: int) -> pd.DataFrame:
    sel = SelectKBest(score_func=f_regression, k=k).fit(Xt, y)
    scores = pd.Series(np.nan_to_num(sel.scores_), index=features)
    rank = _ranking(scores)
    return pd.DataFrame({
        "kbest_score_f": scores,
        "kbest_pvalor": pd.Series(sel.pvalues_, index=features),
        "kbest_rank": rank,
        "kbest_voto": rank <= k,
    })


def metodo_random_forest(Xt, y, features, k, params: dict) -> pd.DataFrame:
    rf = RandomForestRegressor(**params).fit(Xt, y)
    imp = pd.Series(rf.feature_importances_, index=features)
    rank = _ranking(imp)
    return pd.DataFrame({"rf_importancia": imp, "rf_rank": rank, "rf_voto": rank <= k})


def metodo_rfe(Xt, y, features, k, params: dict) -> pd.DataFrame:
    """RFE com Ridge. O ranking_ do RFE dá 1 a todas as selecionadas; para obter
    uma ordenação completa, as selecionadas são ordenadas por |coef| do
    estimador final e as eliminadas pela ordem inversa de eliminação."""
    est = Ridge(alpha=params.get("alpha", 1.0), random_state=params.get("random_state", 42))
    rfe = RFE(estimator=est, n_features_to_select=k).fit(Xt, y)
    rfe_rank = pd.Series(rfe.ranking_, index=features)
    coef = pd.Series(0.0, index=features)
    coef[np.array(features)[rfe.support_]] = np.abs(rfe.estimator_.coef_)
    ordem = sorted(features, key=lambda f: (rfe_rank[f], -coef[f], f))
    rank = pd.Series({f: i + 1 for i, f in enumerate(ordem)})
    return pd.DataFrame({
        "rfe_ranking_sklearn": rfe_rank,
        "rfe_abs_coef": coef,
        "rfe_rank": rank,
        "rfe_voto": pd.Series(rfe.support_, index=features),
    })


def executar_metodos(X_treino: pd.DataFrame, y_treino: pd.Series, features: list[str], cfg: dict) -> pd.DataFrame:
    """Executa os três métodos e devolve uma tabela por feature."""
    v = cfg["votacao"]
    k = int(v["k"])
    pre = construir_preprocessador(features)
    Xt = pre.fit_transform(X_treino[features])  # fit SOMENTE no treino
    y = y_treino.to_numpy()

    tab = pd.concat([
        metodo_selectkbest(Xt, y, features, k),
        metodo_random_forest(Xt, y, features, k, v["metodos"]["random_forest"]),
        metodo_rfe(Xt, y, features, k, v["metodos"]["rfe"]),
    ], axis=1)
    tab.index.name = "feature"
    return tab


def votar(tabela: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    v = cfg["votacao"]
    pesos = v["pesos_ranking"]
    tab = tabela.copy()
    tab["votos"] = tab[["kbest_voto", "rf_voto", "rfe_voto"]].astype(int).sum(axis=1)
    soma_pesos = pesos["selectkbest"] + pesos["random_forest"] + pesos["rfe"]
    tab["rank_medio_ponderado"] = (
        pesos["selectkbest"] * tab["kbest_rank"]
        + pesos["random_forest"] * tab["rf_rank"]
        + pesos["rfe"] * tab["rfe_rank"]
    ) / soma_pesos

    tab["eleita"] = tab["votos"] >= v["min_votos"]
    tab["criterio"] = np.where(tab["eleita"], f">= {v['min_votos']} votos", "")

    faltam = int(v["min_features"]) - int(tab["eleita"].sum())
    if faltam > 0:
        candidatas = tab[~tab["eleita"]].reset_index().sort_values(
            ["rank_medio_ponderado", "votos", "feature"], ascending=[True, False, True]
        )["feature"].head(faltam)
        tab.loc[candidatas, "eleita"] = True
        tab.loc[candidatas, "criterio"] = "desempate: ranking médio ponderado"

    tab = tab.reset_index().sort_values(
        ["eleita", "votos", "rank_medio_ponderado", "feature"], ascending=[False, False, True, True]
    ).set_index("feature")
    eleitas = tab.index[tab["eleita"]].tolist()
    return tab, eleitas
