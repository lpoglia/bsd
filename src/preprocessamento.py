"""ETL: diagnóstico, limpeza, engenharia de atributos, particionamento temporal
e construção do pré-processador (ajustado apenas no treino)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
def _outliers_iqr(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def diagnosticar(df: pd.DataFrame, cfg: dict) -> dict:
    """Ausentes, duplicatas, inconsistências de domínio, outliers e lacunas horárias."""
    chave = cfg["dados"]["chave"]
    dominios = cfg["dados"]["dominios"]
    fora = {}
    for col, (lo, hi) in dominios.items():
        if col in df:
            fora[col] = int(((df[col] < lo) | (df[col] > hi)).sum())

    datas = pd.to_datetime(df[cfg["dados"]["coluna_data"]])
    horas_esperadas = int(((datas.max() - datas.min()).days + 1) * 24)

    diag = {
        "linhas": int(len(df)),
        "periodo": [str(datas.min().date()), str(datas.max().date())],
        "ausentes": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "total_ausentes": int(df.isna().sum().sum()),
        "duplicatas_chave": int(df.duplicated(subset=chave).sum()),
        "duplicatas_linha": int(df.duplicated().sum()),
        "fora_do_dominio": {c: v for c, v in fora.items() if v > 0},
        "umidade_zero": int((df["hum"] == 0).sum()) if "hum" in df else 0,
        "inconsistencia_casual_registered": int(
            (df["casual"] + df["registered"] != df["cnt"]).sum()
        ) if {"casual", "registered"}.issubset(df.columns) else None,
        "inconsistencia_workingday": int(
            ((df["workingday"] == 1) & ((df["holiday"] == 1) | df["weekday"].isin([0, 6]))).sum()
        ),
        "outliers_iqr": {c: _outliers_iqr(df[c]) for c in ["cnt", "windspeed", "hum", "temp"] if c in df},
        "horas_esperadas": horas_esperadas,
        "horas_ausentes_no_periodo": int(horas_esperadas - df.drop_duplicates(subset=chave).shape[0]),
    }
    return diag


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
def limpar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Tratamentos justificados:

    * remove colunas de vazamento (instant, casual, registered);
    * remove duplicatas pela chave (dteday, hr);
    * valores fora do domínio físico -> NaN (imputados no Pipeline, fit só no treino);
    * hum == 0 é leitura impossível de sensor (umidade relativa nula) -> NaN;
    * outliers de cnt e windspeed são MANTIDOS: picos de demanda e rajadas de
      vento são fenômenos reais e removê-los enviesaria o MAE.
    """
    d = cfg["dados"]
    out = df.copy()
    acoes = {}

    remover = [c for c in d["colunas_vazamento"] if c in out]
    out = out.drop(columns=remover)
    acoes["colunas_removidas_vazamento"] = remover

    n0 = len(out)
    out = out.drop_duplicates(subset=d["chave"], keep="last")
    acoes["duplicatas_removidas"] = int(n0 - len(out))

    convertidos = {}
    for col, (lo, hi) in d["dominios"].items():
        mask = (out[col] < lo) | (out[col] > hi)
        if mask.any():
            convertidos[col] = int(mask.sum())
            out.loc[mask, col] = np.nan
    acoes["fora_do_dominio_para_nan"] = convertidos

    mask_hum = out["hum"] == 0
    out.loc[mask_hum, "hum"] = np.nan
    acoes["umidade_zero_para_nan"] = int(mask_hum.sum())

    n1 = len(out)
    out = out.dropna(subset=[d["alvo"]])
    acoes["linhas_sem_alvo_removidas"] = int(n1 - len(out))

    out[d["coluna_data"]] = pd.to_datetime(out[d["coluna_data"]])
    out = ordenar(out, cfg)
    return out, acoes


def ordenar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return df.sort_values(cfg["dados"]["chave"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Engenharia de atributos (transformações linha a linha: sem estatísticas
# globais, portanto não há vazamento entre treino/validação/teste)
# ---------------------------------------------------------------------------
def criar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    out = df.copy()
    out["hr_sin"] = np.sin(2 * np.pi * out["hr"] / 24)
    out["hr_cos"] = np.cos(2 * np.pi * out["hr"] / 24)
    out["mnth_sin"] = np.sin(2 * np.pi * out["mnth"] / 12)
    out["mnth_cos"] = np.cos(2 * np.pi * out["mnth"] / 12)
    out["is_rush_hour"] = ((out["workingday"] == 1) & out["hr"].isin(f["horas_pico"])).astype(int)
    out["is_weekend"] = out["weekday"].isin(f["dias_fim_semana"]).astype(int)
    out["temp_diff"] = out["atemp"] - out["temp"]
    out["temp_hum_interaction"] = out["temp"] * out["hum"]
    return out


def lista_features(cfg: dict) -> list[str]:
    return list(cfg["features"]["originais"]) + list(cfg["features"]["engenharia"])


def preparar(df_bruto: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Limpeza + engenharia (usado igualmente para treino, validação e teste)."""
    limpo, acoes = limpar(df_bruto, cfg)
    return criar_features(limpo, cfg), acoes


# ---------------------------------------------------------------------------
# Particionamento temporal
# ---------------------------------------------------------------------------
def separar_treino_validacao(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout cronológico: últimos `fracao_validacao` registros = validação."""
    df = ordenar(df, cfg)
    frac = cfg["validacao"]["fracao_validacao"]
    corte = int(round(len(df) * (1 - frac)))
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def xy(df: pd.DataFrame, cfg: dict, features: list[str] | None = None):
    features = features or lista_features(cfg)
    return df[features], df[cfg["dados"]["alvo"]].astype(float)


# ---------------------------------------------------------------------------
# Pré-processador (fit somente no treino, dentro do Pipeline)
# ---------------------------------------------------------------------------
def construir_preprocessador(features: list[str]) -> ColumnTransformer:
    numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalonador", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numerico, list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
