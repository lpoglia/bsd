## Ciclo `ciclo_2` · dados `v2` · gatilho `schedule`

- Lotes: lote_1.csv, lote_2.csv (novo: lote_2.csv)
- Features eleitas (13): `is_rush_hour, hr_sin, atemp, temp, hr_cos, yr, hum, weathersit, hr, temp_hum_interaction, season, workingday, mnth_cos`

| Algoritmo | MAE val | RMSE val | R² val |
|---|---|---|---|
| Ridge | 109.711 | 137.775 | 0.643 |
| RandomForestRegressor | 51.624 | 79.465 | 0.881 |

**Decisão: PROMOVIDO** — Melhoria de 46.3641 >= limiar 0.5 (vigente=97.9880, desafiante=51.6238).

- Modelo vigente: `ciclo_2__RandomForestRegressor` · MAE no teste reservado: **57.213**
- Drift: ⚠️ ALERTA em temp, atemp, cnt (PSI máx. 1.866)
