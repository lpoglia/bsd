"""Abre a interface do MLflow sobre o histórico versionado no repositório.

Uso:
    git pull                      # traz os ciclos executados pelo GitHub Actions
    python scripts/abrir_mlflow.py   # http://127.0.0.1:5000

Antes de subir a UI, ajusta os caminhos absolutos de artefatos gravados no
runner do GitHub para a pasta deste clone (ver src/utils.py).
"""
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.utils import MLFLOW_DB, relocar_artefatos_mlflow  # noqa: E402

if __name__ == "__main__":
    if not MLFLOW_DB.exists():
        sys.exit("mlflow.db não encontrado: execute o pipeline ou faça git pull primeiro.")
    print("Registros de artefatos ajustados:", relocar_artefatos_mlflow())
    os.chdir(RAIZ)
    subprocess.run([sys.executable, "-m", "mlflow", "ui",
                    "--backend-store-uri", f"sqlite:///{MLFLOW_DB.as_posix()}",
                    "--port", os.environ.get("MLFLOW_PORT", "5000")], check=True)
