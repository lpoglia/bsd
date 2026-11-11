# Previsão de demanda horária de bicicletas · Pipeline MLOps (Trabalho N3)

Projeto integrador de **Análise Preditiva** (Católica de Santa Catarina): um **único pipeline agendado** no GitHub Actions que, a cada ciclo, incorpora dados novos, executa ETL, elege features por **votação de três métodos**, faz **dois algoritmos competirem**, registra tudo no **MLflow** e decide **automaticamente** se o modelo vigente é substituído.

| | |
|---|---|
| **Dataset** | [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) – `hour.csv`, UCI ML Repository, licença **CC BY 4.0** (17.379 registros horários, 2011–2012) |
| **Problema** | Regressão – prever `cnt` (aluguéis na hora) |
| **Unidade de observação** | 1 hora de operação, chave `["dteday", "hr"]` |
| **Métrica principal** | **MAE** (`minimize`), em bicicletas/hora; RMSE e R² secundárias |
| **Modelos** | A: `Ridge` · B: `RandomForestRegressor` |
| **Seleção de features** | `SelectKBest(f_regression)` + importância de `RandomForest` + `RFE(Ridge)` → votação ≥ 2/3, mínimo 8 |
| **Promoção** | Vencedor substitui o vigente só se `MAE_vigente − MAE_desafiante ≥ 0,5` na mesma validação |
| **Inovação (critério 10)** | Monitoramento de **data drift** (KS + PSI) com alerta e **explicabilidade SHAP** do vencedor |

> Integrantes: _preencher_

---

## Arquitetura do fluxo único

```
schedule (cron) ──► .github/workflows/pipeline_mlops.yml  (1 job, passos sequenciais)
   │
   ├─ checkout: código + data/raw + ESTADO PERSISTENTE (mlflow.db, mlartifacts/, models/, evidencias/)
   ├─ 01_etl.ipynb ............ define cycle_id/data_version, incorpora lote novo sem duplicar,
   │                            diagnóstico antes/depois, features, validações, DRIFT do lote novo
   ├─ 02_votacao_features ..... 3 métodos (só no treino) → tabela de votos → features eleitas
   ├─ 03_treino_competicao .... Ridge × RandomForest, mesmas features, mesmo holdout temporal
   ├─ 04_selecao_promocao ..... vencedor do ciclo × vigente → PROMOVIDO/MANTIDO, SHAP, manifesto
   ├─ 05_previsoes ............ previsões do vigente no teste reservado + teste integrado + histórico
   ├─ finalizar_ciclo.py ...... evidências do ciclo + resumo no Job Summary
   ├─ git commit/push ......... persiste o novo estado (só se todos os passos passaram)
   └─ upload-artifact ......... pacote completo de resultados (90 dias)
```

Cada execução recebe um `cycle_id` (`ciclo_N`) e um `data_version` (`v1`, `v2`, …), que aparecem como tags em **todos** os runs do MLflow (ETL, votação, os dois treinos, decisão e previsões), no manifesto e nas evidências.

### Particionamento e chegada de dados

| Arquivo | Período | Papel |
|---|---|---|
| `data/raw/lote_1.csv` | 2011-01-01 → 2012-06-30 (13.003 linhas) | `ciclo_1`, `v1` |
| `data/raw/lote_2.csv` | 2012-07-01 → 2012-10-31 (2.916 linhas) | incorporado no `ciclo_2`, `v2` (deduplicação pela chave) |
| validação interna | últimos 20% cronológicos do ciclo, sem shuffle | escolha do vencedor e confronto com o vigente |
| `data/raw/teste_final.csv` | 2012-11-01 → 2012-12-31 (1.460 linhas) | **reservado** – só no notebook 05 |

Cada execução incorpora o **próximo lote pendente** (lista em `config/config.yaml`; lotes já processados em `models/estado_pipeline.json`). Sem lote novo, o ciclo reexecuta o pipeline com a mesma `data_version` (retreino de rotina) e a regra de promoção decide se algo muda.

**Sem vazamento:** `instant`, `casual` e `registered` são removidos (`casual + registered = cnt`); `dteday` serve só para ordenação/corte/deduplicação; imputador, escalonador, os 3 métodos de seleção e os modelos são ajustados **apenas no treino** do ciclo (`Pipeline`/`ColumnTransformer`). Os candidatos são treinados apenas na parte de treino, então a validação do ciclo seguinte é inédita tanto para o desafiante quanto para o vigente.

---

## Estrutura do repositório

```
.github/workflows/pipeline_mlops.yml   # workflow único (schedule + workflow_dispatch)
config/config.yaml                     # dados, features, votação, modelos, métrica, limiar de promoção
data/raw/                              # lote_1.csv, lote_2.csv, teste_final.csv
data/processed/                        # intermediários do ciclo (não versionados)
notebooks/01_etl.ipynb … 05_previsoes.ipynb
src/
  utils.py            # caminhos, config, MLflow (SQLite) e portabilidade do histórico
  ingestao.py         # ciclos, lotes, deduplicação, hashes de versão dos dados
  preprocessamento.py # diagnóstico, limpeza, engenharia de atributos, split temporal, ColumnTransformer
  features.py         # SelectKBest, RandomForest, RFE e regra de votação
  modelagem.py        # Pipelines Ridge / RandomForest, métricas
  promocao.py         # eleição, regra de promoção, manifesto, registro de modelos
  monitoramento.py    # drift (KS + PSI) e SHAP
scripts/
  preparar_lotes.py   # gera os lotes a partir do hour.csv original
  executar_pipeline.py# roda localmente o mesmo fluxo do workflow
  finalizar_ciclo.py  # organiza evidências e resumo do ciclo
  abrir_mlflow.py     # abre a UI do MLflow sobre o histórico versionado
  reset_estado.py     # recomeça a simulação do zero
models/
  modelo_vigente.json # manifesto: modelo, data_version, features, métrica, run_ids, decisão, histórico
  estado_pipeline.json# lotes incorporados e histórico de ciclos
  registro/           # todos os modelos já promovidos (nunca sobrescritos)
mlflow.db, mlartifacts/                # tracking do MLflow (persistido entre execuções)
evidencias/<cycle_id>/                 # tabelas, gráficos, JSONs e notebooks executados de cada ciclo
evidencias/historico_ciclos.csv        # comparação entre ciclos (gerada a partir do MLflow)
reports/relatorio_n3.pdf               # relatório técnico
```

---

## Como executar

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Execução local (mesmo fluxo do workflow)

```bash
python scripts/executar_pipeline.py             # executa 1 ciclo (01 → 05)
python scripts/executar_pipeline.py --ciclos 2  # ciclo inicial + ciclo com o lote 2
```

Ou notebook a notebook, na ordem 01 → 05 (Jupyter ou `jupyter nbconvert --to notebook --execute notebooks/01_etl.ipynb …`). Para recomeçar a simulação do zero: `python scripts/reset_estado.py`.

### Execução agendada (GitHub Actions)

* O workflow `.github/workflows/pipeline_mlops.yml` roda pelo `schedule` (`17 9,21 * * *`, 06:17 e 18:17 em Brasília) e pode ser disparado manualmente em **Actions → pipeline_mlops → Run workflow** (só para depuração – a evidência exigida é a execução pelo gatilho `schedule`).
* Requer **Settings → Actions → General → Workflow permissions → Read and write** para o commit automático do estado.
* O GitHub só agenda workflows do branch padrão e pode atrasar execuções agendadas; agendamentos são desativados após 60 dias sem atividade no repositório.
* Cada execução publica: o **Job Summary** com o resumo do ciclo, o **commit** `MLOps ciclo_N (vX): PROMOVIDO|MANTIDO [gatilho: schedule, run …]` e o **artifact** `resultados-ciclo_N-run…`.

### Consultar o MLflow

```bash
git pull                          # traz os ciclos executados no GitHub Actions
python scripts/abrir_mlflow.py    # http://127.0.0.1:5000
```

Experimento `n3_bike_sharing_demanda`; filtre por `tags.cycle_id = 'ciclo_2'` ou `tags.etapa = '03_treino'`. O Model Registry `bike-sharing-mlops` mantém uma versão por promoção, com alias `vigente`. O MLflow grava caminhos absolutos de artefatos; `abrir_mlflow.py` (e toda execução do pipeline) ajusta esses caminhos para a pasta do clone atual.

### Reproduzir as previsões

```python
import joblib, json, pandas as pd
from src import utils, ingestao, preprocessamento as pp
cfg = utils.carregar_config()
man = json.load(open("models/modelo_vigente.json"))
modelo = joblib.load(man["arquivo_modelo"])          # Pipeline: pré-processamento + modelo
teste, _ = pp.preparar(ingestao.carregar_teste_final(cfg), cfg)
previsoes = modelo.predict(teste[man["features"]]).clip(0)
```

O notebook `05_previsoes.ipynb` faz isso, verifica o hash SHA-256 do modelo contra o manifesto e salva `evidencias/<cycle_id>/previsoes_teste_final.csv`.

---

## Regras definidas antes do treino (`config/config.yaml`)

**Votação.** Cada método vota nas suas top-12 features (de 20 candidatas). Eleita = ≥ 2 votos. Se menos de 8 forem eleitas, completa-se com o melhor **ranking médio ponderado** (pesos iguais) entre as demais; empates → mais votos → ordem alfabética. O RFE dá posição 1 a todas as selecionadas, então elas são ordenadas por |coeficiente| do Ridge final para gerar um ranking completo.

**Promoção.** Ciclo sem vigente → promoção direta. Com vigente → ambos avaliados na validação do ciclo atual (o vigente com as próprias features e o próprio pré-processamento); promove se `MAE_vigente − MAE_desafiante ≥ 0,5`. Caso contrário, o vigente é **mantido** e a decisão fica registrada no manifesto, no MLflow e em `evidencias/<cycle_id>/decisao_promocao.json`.

**Complexidade do RandomForest.** `n_estimators=100, max_depth=14, min_samples_leaf=5`. O histórico do MLflow é versionado no Git a cada ciclo; essa configuração reduz o artefato de ~37 MB para ~13 MB com perda pequena de MAE na validação.

---

## Critério 10 – Inovação incremental

### Monitoramento de data drift com alerta
* **Problema:** os dados chegam em lotes ao longo do tempo (sazonalidade, crescimento do sistema); um modelo antigo pode degradar sem ninguém perceber.
* **Solução:** `src/monitoramento.py` compara o lote novo com o histórico usando **Kolmogorov-Smirnov** e **PSI** em `temp`, `atemp`, `hum`, `windspeed`, `cnt` e nas **previsões do modelo vigente**. Alerta = KS com p < 0,05 **e** PSI > 0,2.
* **Integração:** roda no notebook 01 sempre que um lote novo entra; gera `drift_relatorio.json`, `drift_tabela.csv` e `drift_distribuicoes.png`, registra `drift_psi_*` e a tag `drift_alerta` no MLflow e aparece no Job Summary do GitHub Actions.

### Explicabilidade com SHAP
* **Problema:** a decisão de promoção se baseia só no MAE; é preciso auditar *por que* o modelo vencedor prevê o que prevê e se isso é coerente com a votação.
* **Solução:** no notebook 04, `TreeExplainer` (RandomForest) ou `LinearExplainer` (Ridge) sobre uma amostra da validação gera o *summary plot* e a importância média |SHAP|, salvos em `evidencias/<cycle_id>/` e no run de decisão do MLflow.

---

## Evidências por ciclo (`evidencias/<cycle_id>/`)

| Arquivo | Conteúdo |
|---|---|
| `etl_antes_depois.csv`, `etl_diagnostico.json`, `etl_visao_geral.png` | diagnóstico antes/depois, ações de limpeza, incorporação de lotes |
| `drift_*` | relatório de drift do lote novo |
| `votacao_tabela.csv`, `votacao_rankings.png`, `features_eleitas.json` | scores/rankings dos 3 métodos, votos e conjunto eleito |
| `competicao_modelos.csv`, `competicao_validacao.png` | métricas dos dois candidatos |
| `decisao_promocao.json`, `shap_*` | decisão automática e explicabilidade |
| `previsoes_teste_final.csv/.png` | previsões do modelo vigente no teste reservado |
| `mudancas_vs_ciclo_anterior.json`, `resumo_ciclo.md` | o que mudou entre ciclos |
| `notebooks/*_executado.ipynb` | notebooks executados com todas as saídas |

## Limitações

* Dados de uma única cidade e de apenas dois anos; a tendência de crescimento é captada apenas por `yr`, o que penaliza a validação de períodos com crescimento acima do visto no treino.
* Variáveis climáticas normalizadas (sem unidade física) e algumas horas ausentes no dataset original.
* O teste reservado cobre só nov–dez/2012 (inverno), período sazonalmente diferente dos lotes de treino.
* O estado persistente fica no próprio repositório: simples e auditável, mas o tamanho do repositório cresce a cada ciclo (≈ 15 MB). Em produção, o adequado seria um servidor MLflow com armazenamento de objetos.
