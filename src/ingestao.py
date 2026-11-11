"""Entrada de dados, controle de ciclos e versionamento dos lotes.

Simulação de chegada de dados: config.yaml lista os lotes em ordem cronológica.
Cada execução do pipeline incorpora o próximo lote ainda não processado
(o histórico fica em models/estado_pipeline.json, persistido no repositório).
Se não houver lote novo, o ciclo é reexecutado com a mesma data_version
(retreino de rotina) e a regra de promoção decide se algo muda.
"""
from __future__ import annotations

import os

import pandas as pd

from src.utils import (
    CAMINHO_CICLO,
    CAMINHO_ESTADO,
    RAIZ,
    agora_iso,
    ler_json,
    salvar_json,
    sha256_arquivo,
)

ESTADO_INICIAL = {"execucoes": 0, "lotes_incorporados": [], "historico": []}


def carregar_estado() -> dict:
    return ler_json(CAMINHO_ESTADO, padrao=dict(ESTADO_INICIAL, historico=[], lotes_incorporados=[]))


def salvar_estado(estado: dict) -> None:
    salvar_json(estado, CAMINHO_ESTADO)


def definir_ciclo(cfg: dict, estado: dict) -> dict:
    """Define cycle_id, data_version e os lotes que compõem o ciclo."""
    lotes_cfg = cfg["dados"]["lotes"]
    incorporados = list(estado.get("lotes_incorporados", []))
    pendentes = [l for l in lotes_cfg if l["arquivo"] not in incorporados]

    if pendentes:
        novo = pendentes[0]
        lotes_ciclo = incorporados + [novo["arquivo"]]
        data_version = novo["data_version"]
        lote_novo = novo["arquivo"]
    else:
        if not incorporados:
            raise RuntimeError("Nenhum lote configurado em config.yaml")
        lotes_ciclo = incorporados
        data_version = next(l["data_version"] for l in lotes_cfg if l["arquivo"] == incorporados[-1])
        lote_novo = None

    n = int(estado.get("execucoes", 0)) + 1
    dir_raw = RAIZ / cfg["dados"]["dir_raw"]
    return {
        "cycle_id": f"ciclo_{n}",
        "numero_execucao": n,
        "data_version": data_version,
        "lotes_ciclo": lotes_ciclo,
        "lote_novo": lote_novo,
        "lotes_anteriores": incorporados,
        "hash_lotes": {a: sha256_arquivo(dir_raw / a) for a in lotes_ciclo},
        "iniciado_em": agora_iso(),
        "gatilho": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
    }


def registrar_ciclo(ciclo: dict) -> None:
    salvar_json(ciclo, CAMINHO_CICLO)


def carregar_lote(cfg: dict, arquivo: str) -> pd.DataFrame:
    df = pd.read_csv(RAIZ / cfg["dados"]["dir_raw"] / arquivo)
    df["lote_origem"] = arquivo
    return df


def incorporar_lotes(cfg: dict, arquivos: list[str]) -> tuple[pd.DataFrame, dict]:
    """Concatena os lotes do ciclo e remove duplicatas pela chave (dteday, hr).

    Em caso de chave repetida entre lotes, mantém o registro do lote mais
    recente (dado corrigido/reenviado prevalece sobre o antigo).
    """
    chave = cfg["dados"]["chave"]
    partes = [carregar_lote(cfg, a) for a in arquivos]
    bruto = pd.concat(partes, ignore_index=True)
    dup_mask = bruto.duplicated(subset=chave, keep="last")
    consolidado = bruto.loc[~dup_mask].reset_index(drop=True)
    relatorio = {
        "linhas_por_lote": {a: int(len(p)) for a, p in zip(arquivos, partes)},
        "linhas_concatenadas": int(len(bruto)),
        "duplicatas_chave_removidas": int(dup_mask.sum()),
        "linhas_consolidadas": int(len(consolidado)),
    }
    return consolidado, relatorio


def carregar_teste_final(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(RAIZ / cfg["dados"]["dir_raw"] / cfg["dados"]["teste_final"])
    df["lote_origem"] = cfg["dados"]["teste_final"]
    return df
