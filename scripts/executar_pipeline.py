"""Executa localmente o mesmo fluxo único do GitHub Actions (01 -> 05).

Uso:
    python scripts/executar_pipeline.py            # um ciclo
    python scripts/executar_pipeline.py --ciclos 2 # dois ciclos seguidos

Cada notebook executado é salvo em evidencias/<cycle_id>/notebooks/.
Qualquer erro interrompe a sequência (nenhuma promoção indevida ocorre,
pois o manifesto só é escrito no fim do notebook 04).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalizar_ciclo import RAIZ, TMP, finalizar  # noqa: E402

NOTEBOOKS = [
    "01_etl.ipynb",
    "02_votacao_features.ipynb",
    "03_treino_competicao.ipynb",
    "04_selecao_promocao.ipynb",
    "05_previsoes.ipynb",
]


def executar_ciclo() -> str:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    for nb in NOTEBOOKS:
        print(f"==> {nb}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
             f"notebooks/{nb}", "--output-dir", str(TMP),
             "--output", nb.replace(".ipynb", "_executado.ipynb"),
             "--ExecutePreprocessor.timeout=1800"],
            cwd=RAIZ, check=True,
        )
    return finalizar()["cycle_id"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciclos", type=int, default=1)
    args = ap.parse_args()
    for _ in range(args.ciclos):
        executar_ciclo()
