"""Etapa 5 - Eleição do vencedor do ciclo e promoção controlada.

Regra (config.yaml -> promocao):
  * Sem modelo vigente (primeiro ciclo): o vencedor é promovido diretamente.
  * Com vigente: ambos são avaliados na MESMA validação temporal do ciclo atual.
    Para métrica a minimizar, promove se  vigente - desafiante >= limiar.
    Caso contrário o vigente é mantido (e continua servindo previsões).
Modelos promovidos ficam preservados em models/registro/ (nunca sobrescritos).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import joblib

from src.utils import (
    CAMINHO_MANIFESTO,
    DIR_REGISTRO,
    RAIZ,
    agora_iso,
    ler_json,
    relativo,
    salvar_json,
    sha256_arquivo,
)


def eleger_vencedor(resultados: list[dict], metrica: str, direcao: str) -> dict:
    """resultados: [{'papel','algoritmo','metricas':{...},...}]"""
    chave = f"val_{metrica}"
    reverso = direcao == "maximize"
    ordenados = sorted(resultados, key=lambda r: r["metricas"][chave], reverse=reverso)
    return ordenados[0]


def carregar_manifesto() -> dict | None:
    return ler_json(CAMINHO_MANIFESTO)


def carregar_modelo_vigente(manifesto: dict):
    caminho = RAIZ / manifesto["arquivo_modelo"]
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo vigente não encontrado em {caminho}")
    if sha256_arquivo(caminho) != manifesto["sha256_modelo"]:
        raise ValueError("Hash do modelo vigente não confere com o manifesto")
    return joblib.load(caminho)


def decidir(valor_desafiante: float, valor_vigente: float | None, limiar: float, direcao: str) -> dict:
    if valor_vigente is None:
        return {
            "promover": True,
            "decisao": "PROMOVIDO",
            "motivo": "Não havia modelo vigente: vencedor do ciclo promovido diretamente.",
            "melhoria": None,
        }
    melhoria = (valor_vigente - valor_desafiante) if direcao == "minimize" else (valor_desafiante - valor_vigente)
    promover = melhoria >= limiar
    motivo = (
        f"Melhoria de {melhoria:.4f} {'>=' if promover else '<'} limiar {limiar} "
        f"(vigente={valor_vigente:.4f}, desafiante={valor_desafiante:.4f})."
    )
    return {
        "promover": bool(promover),
        "decisao": "PROMOVIDO" if promover else "MANTIDO",
        "motivo": motivo,
        "melhoria": float(melhoria),
    }


def arquivar_modelo(caminho_candidato: Path, cycle_id: str, algoritmo: str) -> Path:
    DIR_REGISTRO.mkdir(parents=True, exist_ok=True)
    destino = DIR_REGISTRO / f"{cycle_id}__{algoritmo}.joblib"
    shutil.copy2(caminho_candidato, destino)
    return destino


def atualizar_manifesto(manifesto_antigo: dict | None, vencedor: dict, ciclo: dict, decisao: dict,
                        metrica: str, direcao: str, run_id_decisao: str, arquivo_modelo: Path | None,
                        valor_vigente_atual: float | None, experimento: str) -> dict:
    """Grava o manifesto. Se mantido, preserva o vigente e apenas registra a decisão."""
    historico = list((manifesto_antigo or {}).get("historico_decisoes", []))
    registro = {
        "cycle_id": ciclo["cycle_id"],
        "data_version": ciclo["data_version"],
        "data_hora": agora_iso(),
        "desafiante": {
            "algoritmo": vencedor["algoritmo"],
            "run_id": vencedor["run_id"],
            f"val_{metrica}": vencedor["metricas"][f"val_{metrica}"],
        },
        "vigente_anterior": None if manifesto_antigo is None else {
            "modelo_id": manifesto_antigo["modelo_id"],
            f"val_{metrica}_no_ciclo": valor_vigente_atual,
        },
        "decisao": decisao["decisao"],
        "motivo": decisao["motivo"],
        "melhoria": decisao["melhoria"],
        "run_id_decisao": run_id_decisao,
    }
    historico.append(registro)

    if decisao["promover"]:
        manifesto = {
            "modelo_id": f"{ciclo['cycle_id']}__{vencedor['algoritmo']}",
            "algoritmo": vencedor["algoritmo"],
            "params": vencedor["params"],
            "cycle_id": ciclo["cycle_id"],
            "data_version": ciclo["data_version"],
            "lotes_treino": ciclo["lotes_ciclo"],
            "features": vencedor["features"],
            "metrica": {
                "nome": metrica,
                "direcao": direcao,
                "valor_validacao": vencedor["metricas"][f"val_{metrica}"],
                "base": f"validação temporal (últimos 20%) do {ciclo['cycle_id']}",
            },
            "mlflow": {
                "experimento": experimento,
                "run_id_treino": vencedor["run_id"],
                "run_id_votacao": ciclo.get("run_id_votacao"),
                "run_id_decisao": run_id_decisao,
            },
            "arquivo_modelo": relativo(arquivo_modelo),
            "sha256_modelo": sha256_arquivo(arquivo_modelo),
            "promovido_em": agora_iso(),
        }
    else:
        manifesto = dict(manifesto_antigo)

    manifesto["ultima_avaliacao"] = {
        "cycle_id": ciclo["cycle_id"],
        "data_version": ciclo["data_version"],
        "decisao": decisao["decisao"],
        "data_hora": registro["data_hora"],
    }
    manifesto["historico_decisoes"] = historico
    salvar_json(manifesto, CAMINHO_MANIFESTO)
    return manifesto
