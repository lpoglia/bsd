"""Apaga TODO o estado do pipeline (MLflow, manifesto, modelos, evidências).

Use apenas para recomeçar a simulação do zero (ex.: antes da primeira
execução agendada). Pede confirmação, a menos que --sim seja informado.
"""
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ALVOS = [
    RAIZ / "mlflow.db",
    RAIZ / "mlartifacts",
    RAIZ / "models" / "modelo_vigente.json",
    RAIZ / "models" / "estado_pipeline.json",
    RAIZ / "models" / "registro",
    RAIZ / "data" / "processed",
    RAIZ / "evidencias",
]

if __name__ == "__main__":
    existentes = [p for p in ALVOS if p.exists()]
    if not existentes:
        sys.exit("Nada para apagar.")
    print("Serão apagados:\n  " + "\n  ".join(str(p.relative_to(RAIZ)) for p in existentes))
    if "--sim" not in sys.argv and input("Confirmar? [s/N] ").strip().lower() != "s":
        sys.exit("Cancelado.")
    for p in existentes:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for d in ["data/processed", "evidencias", "models/registro"]:
        (RAIZ / d).mkdir(parents=True, exist_ok=True)
        (RAIZ / d / ".gitkeep").touch()
    print("Estado reiniciado.")
