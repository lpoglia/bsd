"""Utilitários comuns: caminhos, configuração, JSON e MLflow.

Todos os caminhos são resolvidos a partir da raiz do repositório, de modo que
notebooks executados via nbconvert (cwd = notebooks/) e scripts leiam e gravem
artefatos sempre nos mesmos locais.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_CONFIG = RAIZ / "config" / "config.yaml"
DIR_MODELS = RAIZ / "models"
DIR_REGISTRO = DIR_MODELS / "registro"
DIR_EVIDENCIAS = RAIZ / "evidencias"
DIR_PROCESSED = RAIZ / "data" / "processed"
CAMINHO_MANIFESTO = DIR_MODELS / "modelo_vigente.json"
CAMINHO_ESTADO = DIR_MODELS / "estado_pipeline.json"
CAMINHO_CICLO = DIR_PROCESSED / "ciclo_atual.json"
MLFLOW_DB = RAIZ / "mlflow.db"
MLFLOW_ARTIFACTS = RAIZ / "mlartifacts"


def carregar_config(caminho: Path = CAMINHO_CONFIG) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def agora_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Tipo não serializável: {type(obj)}")


def salvar_json(dados: Any, caminho: Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=_json_default)
    return caminho


def ler_json(caminho: Path, padrao: Any = None) -> Any:
    caminho = Path(caminho)
    if not caminho.exists():
        return padrao
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def relativo(caminho: Path) -> str:
    """Caminho relativo à raiz do repositório (portável entre máquinas/runners)."""
    return Path(caminho).resolve().relative_to(RAIZ).as_posix()


def dir_evidencias(cycle_id: str) -> Path:
    d = DIR_EVIDENCIAS / cycle_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def carregar_ciclo() -> dict:
    """Identificadores do ciclo em execução (gerados pelo notebook 01)."""
    ciclo = ler_json(CAMINHO_CICLO)
    if ciclo is None:
        raise FileNotFoundError(
            "data/processed/ciclo_atual.json não encontrado: execute o notebook 01 primeiro."
        )
    return ciclo


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
def relocar_artefatos_mlflow() -> int:
    """Torna o histórico do MLflow portável entre máquinas.

    O MLflow grava caminhos ABSOLUTOS de artefatos no banco. Como mlflow.db e
    mlartifacts/ são versionados no Git, o caminho muda entre o runner do
    GitHub Actions e um clone local. Esta função reescreve o prefixo antigo
    para a raiz atual do repositório. Retorna o nº de registros ajustados.
    """
    import sqlite3

    if not MLFLOW_DB.exists():
        return 0
    novo = MLFLOW_ARTIFACTS.as_uri()
    con = sqlite3.connect(MLFLOW_DB)
    try:
        linhas = con.execute("SELECT artifact_location FROM experiments").fetchall()
        antigos = {
            loc[: loc.index("/mlartifacts") + len("/mlartifacts")]
            for (loc,) in linhas
            if loc and "/mlartifacts" in loc
        }
        alterados = 0
        alvos = [
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"),
            ("model_versions", "source"),
            ("model_versions", "storage_location"),
        ]
        existentes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for antigo in antigos:
            if antigo == novo:
                continue
            for tabela, coluna in alvos:
                if tabela not in existentes:
                    continue
                cur = con.execute(
                    f"UPDATE {tabela} SET {coluna} = REPLACE({coluna}, ?, ?) WHERE {coluna} LIKE ?",
                    (antigo, novo, antigo + "%"),
                )
                alterados += cur.rowcount
        con.commit()
        return alterados
    finally:
        con.close()


def configurar_mlflow(cfg: dict) -> str:
    """Aponta o MLflow para o backend SQLite versionado no repositório.

    O banco (mlflow.db) e a pasta de artefatos (mlartifacts/) ficam na raiz do
    repositório e são persistidos entre execuções do GitHub Actions por commit
    automático + upload de artifact ao fim do workflow.
    """
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    relocar_artefatos_mlflow()
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    nome = cfg["projeto"]["experimento_mlflow"]
    exp = mlflow.get_experiment_by_name(nome)
    if exp is None:
        MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        exp_id = mlflow.create_experiment(nome, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(nome)
    return exp_id


def tags_ciclo(ciclo: dict, etapa: str) -> dict:
    return {
        "cycle_id": ciclo["cycle_id"],
        "data_version": ciclo["data_version"],
        "etapa": etapa,
        "gatilho": ciclo.get("gatilho", "local"),
        "github_run_id": ciclo.get("github_run_id", ""),
    }
