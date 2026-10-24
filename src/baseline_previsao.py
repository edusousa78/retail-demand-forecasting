

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACAO DOS DIRETORIOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARQUIVO_TREINO = PROCESSED_DIR / "treino.csv"
ARQUIVO_VALIDACAO = PROCESSED_DIR / "validacao.csv"
ARQUIVO_SAIDA = PROCESSED_DIR / "previsoes_baseline.csv"


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

treino = pd.read_csv(
    ARQUIVO_TREINO,
    parse_dates=["date"]
)

validacao = pd.read_csv(
    ARQUIVO_VALIDACAO,
    parse_dates=["date"]
)

treino = treino.sort_values(
    ["id", "date"]
).reset_index(drop=True)

validacao = validacao.sort_values(
    ["id", "date"]
).reset_index(drop=True)


# ============================================================
# 3. VALIDACAO TEMPORAL
# ============================================================

assert not treino.empty, "Treinamento vazio."
assert not validacao.empty, "Validacao vazia."

assert treino["date"].max() < validacao["date"].min(), (
    "Erro: existe sobreposicao temporal."
)

assert set(validacao["id"]).issubset(set(treino["id"])), (
    "Existem produtos/lojas na validacao sem historico."
)


# ============================================================
# 4. CONSTRUCAO DOS BASELINES
# ============================================================

# Baseline 1: ultima venda conhecida
baseline_ultima_venda = (
    treino.groupby("id")["vendas"]
    .last()
    .rename("baseline_ultima_venda")
)

# Baseline 2: media dos ultimos 7 dias
baseline_media_7 = (
    treino.groupby("id")["vendas"]
    .apply(lambda serie: serie.tail(7).mean())
    .rename("baseline_media_7")
)

# Baseline 3: media dos ultimos 3 dias
baseline_media_3 = (
    treino.groupby("id")["vendas"]
    .apply(lambda serie: serie.tail(3).mean())
    .rename("baseline_media_3")
)


# ============================================================
# 5. ASSOCIACAO DAS PREVISOES
# ============================================================

previsoes = validacao.copy()

baselines = [
    baseline_ultima_venda,
    baseline_media_7,
    baseline_media_3
]

for baseline in baselines:
    previsoes = previsoes.merge(
        baseline,
        left_on="id",
        right_index=True,
        how="left",
        validate="many_to_one"
    )

colunas_modelos = [
    "baseline_ultima_venda",
    "baseline_media_7",
    "baseline_media_3"
]

assert previsoes[colunas_modelos].notna().all().all(), (
    "Existem previsoes ausentes."
)


# ============================================================
# 6. METRICAS DE AVALIACAO
# ============================================================

def calcular_mae(real, previsto):
    return np.mean(np.abs(real - previsto))


def calcular_wape(real, previsto):
    denominador = np.sum(np.abs(real))

    if denominador == 0:
        return np.nan

    erro_total = np.sum(np.abs(real - previsto))

    return (erro_total / denominador) * 100


# ============================================================
# 7. AVALIACAO DOS MODELOS
# ============================================================

real = previsoes["vendas"].to_numpy()

print("=" * 55)
print("BASELINE DE PREVISAO - NOVAVAREJO")
print("=" * 55)

print(f"Registros de treinamento: {len(treino)}")
print(f"Registros de validacao: {len(validacao)}")
print(f"Series temporais: {validacao['id'].nunique()}")

resultados = []

for coluna in colunas_modelos:
    previsto = previsoes[coluna].to_numpy()

    mae = calcular_mae(real, previsto)
    wape = calcular_wape(real, previsto)

    resultados.append({
        "modelo": coluna,
        "mae": mae,
        "wape": wape
    })

    print(f"\nModelo: {coluna}")
    print(f"MAE: {mae:.4f}")

    if np.isnan(wape):
        print(
            "WAPE: indefinido "
            "(vendas reais totais = 0)"
        )
    else:
        print(f"WAPE: {wape:.2f}%")


# ============================================================
# 8. EXPORTACAO DOS RESULTADOS
# ============================================================

previsoes.to_csv(
    ARQUIVO_SAIDA,
    index=False
)

resumo = pd.DataFrame(resultados)

arquivo_resumo = (
    PROCESSED_DIR / "metricas_baseline.csv"
)

resumo.to_csv(
    arquivo_resumo,
    index=False
)

print("\n" + "=" * 55)
print("PROCESSAMENTO FINALIZADO")
print("=" * 55)

print(f"Previsoes: {ARQUIVO_SAIDA}")
print(f"Metricas: {arquivo_resumo}")

print("\nValidacoes concluidas.")
