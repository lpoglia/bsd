"""Organiza as evidências do ciclo recém-executado.

* move os notebooks executados de evidencias/_execucao_atual/ para
  evidencias/<cycle_id>/notebooks/;
* no GitHub Actions, publica o resumo do ciclo no "Job summary" e exporta
  cycle_id / data_version / decisao como outputs do step.
"""
import json
import os
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
TMP = RAIZ / "evidencias" / "_execucao_atual"


def finalizar() -> dict:
    ciclo = json.loads((RAIZ / "data" / "processed" / "ciclo_atual.json").read_text(encoding="utf-8"))
    dir_ciclo = RAIZ / "evidencias" / ciclo["cycle_id"]
    destino = dir_ciclo / "notebooks"
    if TMP.exists():
        if destino.exists():
            shutil.rmtree(destino)
        shutil.move(str(TMP), str(destino))

    decisao = json.loads((dir_ciclo / "decisao_promocao.json").read_text(encoding="utf-8"))
    resumo = dir_ciclo / "resumo_ciclo.md"

    if os.environ.get("GITHUB_STEP_SUMMARY") and resumo.exists():
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(resumo.read_text(encoding="utf-8"))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"cycle_id={ciclo['cycle_id']}\n")
            f.write(f"data_version={ciclo['data_version']}\n")
            f.write(f"decisao={decisao['decisao']}\n")
    print(f"{ciclo['cycle_id']} ({ciclo['data_version']}): {decisao['decisao']} - evidências em "
          f"{dir_ciclo.relative_to(RAIZ).as_posix()}")
    return ciclo


if __name__ == "__main__":
    finalizar()
