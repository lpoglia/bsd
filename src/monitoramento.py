"""Critério 10 - Inovação incremental.

(A) Monitoramento de data drift entre o histórico e o lote novo
    (Kolmogorov-Smirnov + PSI) nas variáveis contínuas e na distribuição das
    previsões do modelo vigente, com alerta automático.
(B) Explicabilidade com SHAP do vencedor de cada ciclo (summary plot +
    importância média absoluta), salva em evidencias/ e no MLflow.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402


# ---------------------------------------------------------------------------
# (A) Data drift
# ---------------------------------------------------------------------------
def psi(referencia: np.ndarray, atual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index com faixas por quantis da referência.
    < 0.1 estável | 0.1-0.2 moderado | > 0.2 mudança significativa."""
    referencia = np.asarray(referencia, dtype=float)
    atual = np.asarray(atual, dtype=float)
    referencia = referencia[~np.isnan(referencia)]
    atual = atual[~np.isnan(atual)]
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_ref = np.histogram(referencia, cortes)[0] / len(referencia)
    p_atu = np.histogram(atual, cortes)[0] / len(atual)
    p_ref = np.clip(p_ref, eps, None)
    p_atu = np.clip(p_atu, eps, None)
    return float(np.sum((p_atu - p_ref) * np.log(p_atu / p_ref)))


def comparar_distribuicoes(ref: np.ndarray, atual: np.ndarray, cfg_drift: dict) -> dict:
    ks = ks_2samp(ref[~np.isnan(ref)], atual[~np.isnan(atual)])
    valor_psi = psi(ref, atual, cfg_drift["psi_bins"])
    return {
        "media_referencia": float(np.nanmean(ref)),
        "media_atual": float(np.nanmean(atual)),
        "ks_estatistica": float(ks.statistic),
        "ks_pvalor": float(ks.pvalue),
        "psi": valor_psi,
        "nivel_psi": "estavel" if valor_psi < 0.1 else ("moderado" if valor_psi < cfg_drift["psi_alerta"] else "significativo"),
        # Alerta exige as duas evidências: KS significativo E PSI acima do limiar
        # (com milhares de linhas o KS sozinho acusa diferenças irrelevantes).
        "drift": bool(ks.pvalue < cfg_drift["ks_alpha"] and valor_psi > cfg_drift["psi_alerta"]),
    }


def relatorio_drift(ref: pd.DataFrame, novo: pd.DataFrame, cfg: dict,
                    pred_ref: np.ndarray | None = None, pred_novo: np.ndarray | None = None) -> dict:
    cd = cfg["monitoramento"]["drift"]
    variaveis = {v: comparar_distribuicoes(ref[v].to_numpy(float), novo[v].to_numpy(float), cd)
                 for v in cd["variaveis"]}
    rel = {
        "linhas_referencia": int(len(ref)),
        "linhas_lote_novo": int(len(novo)),
        "variaveis": variaveis,
        "previsoes_vigente": None,
    }
    if pred_ref is not None and pred_novo is not None:
        rel["previsoes_vigente"] = comparar_distribuicoes(np.asarray(pred_ref), np.asarray(pred_novo), cd)
    com_drift = [v for v, r in variaveis.items() if r["drift"]]
    if rel["previsoes_vigente"] and rel["previsoes_vigente"]["drift"]:
        com_drift.append("previsoes_vigente")
    rel["variaveis_com_drift"] = com_drift
    rel["alerta"] = bool(com_drift)
    rel["psi_maximo"] = float(max(r["psi"] for r in variaveis.values()))
    return rel


def tabela_drift(rel: dict) -> pd.DataFrame:
    linhas = dict(rel["variaveis"])
    if rel.get("previsoes_vigente"):
        linhas["previsoes_vigente"] = rel["previsoes_vigente"]
    return pd.DataFrame(linhas).T


def grafico_drift(ref: pd.DataFrame, novo: pd.DataFrame, variaveis: list[str], destino: Path,
                  pred_ref=None, pred_novo=None) -> Path:
    series = [(v, ref[v], novo[v]) for v in variaveis]
    if pred_ref is not None:
        series.append(("previsões do vigente", pd.Series(pred_ref), pd.Series(pred_novo)))
    fig, axes = plt.subplots(1, len(series), figsize=(3.2 * len(series), 3))
    for ax, (nome, a, b) in zip(np.atleast_1d(axes), series):
        ax.hist(a.dropna(), bins=30, alpha=0.55, density=True, label="histórico")
        ax.hist(b.dropna(), bins=30, alpha=0.55, density=True, label="lote novo")
        ax.set_title(nome, fontsize=9)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destino, dpi=110)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# (B) Explicabilidade com SHAP
# ---------------------------------------------------------------------------
def explicar_shap(pipeline, X: pd.DataFrame, features: list[str], cfg: dict,
                  destino_png: Path) -> pd.DataFrame:
    """Calcula valores SHAP do modelo final do Pipeline sobre uma amostra
    (já transformada pelo pré-processador do próprio Pipeline)."""
    import shap

    cs = cfg["monitoramento"]["shap"]
    amostra = X[features].sample(n=min(cs["amostra"], len(X)), random_state=cs["random_state"])
    pre = pipeline.named_steps["preprocessamento"]
    modelo = pipeline.named_steps["modelo"]
    Xt = pd.DataFrame(pre.transform(amostra), columns=features, index=amostra.index)

    if hasattr(modelo, "estimators_"):          # ensembles de árvores
        explainer = shap.TreeExplainer(modelo)
        valores = explainer.shap_values(Xt, check_additivity=False)
    else:                                         # modelos lineares
        explainer = shap.LinearExplainer(modelo, Xt)
        valores = explainer.shap_values(Xt)

    plt.figure()
    shap.summary_plot(valores, Xt, show=False, max_display=len(features))
    plt.title(f"SHAP - {type(modelo).__name__}", fontsize=10)
    plt.tight_layout()
    plt.savefig(destino_png, dpi=110, bbox_inches="tight")
    plt.close("all")

    imp = pd.DataFrame({
        "feature": features,
        "shap_media_absoluta": np.abs(valores).mean(axis=0),
    }).sort_values("shap_media_absoluta", ascending=False).reset_index(drop=True)
    return imp
