## Ciclo `ciclo_1` · dados `v1` · gatilho `workflow_dispatch`

- Lotes: lote_1.csv (novo: lote_1.csv)
- Features eleitas (13): `is_rush_hour, hr_sin, hr_cos, atemp, temp, hum, season, weathersit, hr, temp_hum_interaction, workingday, yr, mnth_cos`

| Algoritmo | MAE val | RMSE val | R² val |
|---|---|---|---|
| Ridge | 98.835 | 132.139 | 0.624 |
| RandomForestRegressor | 77.778 | 107.456 | 0.751 |

**Decisão: PROMOVIDO** — Não havia modelo vigente: vencedor do ciclo promovido diretamente.

- Modelo vigente: `ciclo_1__RandomForestRegressor` · MAE no teste reservado: **69.459**
