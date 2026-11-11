"""Gera os lotes cronológicos a partir do hour.csv original (executar uma única vez).

Uso:
    python scripts/preparar_lotes.py caminho/para/hour.csv

Fonte: UCI Machine Learning Repository - Bike Sharing Dataset (CC BY 4.0)
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "raw"

CORTES = {
    "lote_1.csv": ("2011-01-01", "2012-06-30"),
    "lote_2.csv": ("2012-07-01", "2012-10-31"),
    "teste_final.csv": ("2012-11-01", "2012-12-31"),
}


def main(origem: str) -> None:
    df = pd.read_csv(origem)
    datas = pd.to_datetime(df["dteday"])
    DESTINO.mkdir(parents=True, exist_ok=True)
    total = 0
    for nome, (ini, fim) in CORTES.items():
        parte = df[(datas >= ini) & (datas <= fim)]
        parte.to_csv(DESTINO / nome, index=False)
        total += len(parte)
        print(f"{nome}: {len(parte)} linhas ({ini} a {fim})")
    assert total == len(df), "Os cortes devem cobrir todo o dataset sem sobreposição"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
