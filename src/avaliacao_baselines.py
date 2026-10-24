
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARQUIVO_TREINO = PROCESSED_DIR / "treino_modelagem.csv"
ARQUIVO_VALIDACAO = PROCESSED_DIR / "validacao_modelagem.csv"

ARQUIVO_PREVISOES = (
    PROCESSED_DIR / "previsoes_baseline_modelagem.csv"
)

ARQUIVO_METRICAS = (
    PROCESSED_DIR / "metricas_baseline_modelagem.csv"
)


# ============================================================
# 2. CARREGAMENTO
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
# 3. VALIDACAO DOS DADOS
# ============================================================

assert not treino.empty, "Treinamento vazio."
assert not validacao.empty, "Validacao vazia."

assert treino["date"].max() < validacao["date"].min(), (
    "Existe sobreposicao temporal."
)

assert set(treino["id"]) == set(validacao["id"]), (
    "Series diferentes entre treino e validacao."
)

assert not treino.duplicated(
    subset=["id", "date"]
).any()

assert not validacao.duplicated(
    subset=["id", "date"]
).any()

assert (treino["vendas"] >= 0).all()
assert (validacao["vendas"] >= 0).all()

assert validacao["vendas"].sum() > 0, (
    "A validacao nao possui vendas positivas."
)

dias_validacao = validacao["date"].nunique()

assert dias_validacao == 7, (
    "O horizonte esperado e de sete dias."
)

assert (
    validacao.groupby("id")["date"].nunique() == 7
).all(), "Existem series com dias faltantes."


# ============================================================
# 4. CONSTRUCAO DOS BASELINES
# ============================================================

grupo = treino.groupby("id")["vendas"]

ultima_venda = (
    grupo.last()
    .rename("baseline_ultima_venda")
)

media_7 = (
    grupo.apply(
        lambda serie: serie.tail(7).mean()
    )
    .rename("baseline_media_7")
)

media_3 = (
    grupo.apply(
        lambda serie: serie.tail(3).mean()
    )
    .rename("baseline_media_3")
)

baselines = pd.concat(
    [ultima_venda, media_7, media_3],
    axis=1
)

colunas_modelos = [
    "baseline_ultima_venda",
    "baseline_media_7",
    "baseline_media_3"
]


# ============================================================
# 5. PREVISOES PARA SETE DIAS
# ============================================================

previsoes = validacao.merge(
    baselines,
    left_on="id",
    right_index=True,
    how="left",
    validate="many_to_one"
)

assert previsoes[colunas_modelos].notna().all().all(), (
    "Existem previsoes ausentes."
)

assert (
    previsoes.groupby("id")[colunas_modelos]
    .nunique()
    .eq(1)
    .all()
    .all()
), "As previsoes deveriam ser fixas por serie."


# ============================================================
# 6. METRICAS
# ============================================================

def calcular_mae(real, previsto):
    return float(
        np.mean(np.abs(real - previsto))
    )


def calcular_wape(real, previsto):
    denominador = np.sum(np.abs(real))

    if denominador == 0:
        return np.nan

    erro_absoluto = np.sum(
        np.abs(real - previsto)
    )

    return float(
        (erro_absoluto / denominador) * 100
    )


# ============================================================
# 7. AVALIACAO DOS MODELOS
# ============================================================

real = previsoes["vendas"].to_numpy(
    dtype=float
)

resultados = []

print("=" * 60)
print("AVALIACAO DE BASELINES - NOVAVAREJO")
print("=" * 60)

print(f"Series temporais: {treino['id'].nunique()}")
print(f"Registros de treino: {len(treino)}")
print(f"Registros de validacao: {len(validacao)}")
print(f"Horizonte de previsao: {dias_validacao} dias")

print(
    f"\nUltima data do treino: "
    f"{treino['date'].max().date()}"
)

print(
    f"Primeira data da validacao: "
    f"{validacao['date'].min().date()}"
)

print(
    f"Vendas reais na validacao: "
    f"{int(real.sum())}"
)

for modelo in colunas_modelos:
    previsto = previsoes[modelo].to_numpy(
        dtype=float
    )

    mae = calcular_mae(real, previsto)
    wape = calcular_wape(real, previsto)

    resultados.append({
        "modelo": modelo,
        "mae": mae,
        "wape": wape
    })

    print(f"\nModelo: {modelo}")
    print(f"MAE: {mae:.4f}")
    print(f"WAPE: {wape:.2f}%")


# ============================================================
# 8. COMPARACAO E RANKING
# ============================================================

metricas = pd.DataFrame(resultados)

metricas = metricas.sort_values(
    ["wape", "mae", "modelo"]
).reset_index(drop=True)

melhor_modelo = metricas.iloc[0]

print("\n" + "=" * 60)
print("RANKING DOS MODELOS")
print("=" * 60)

print(
    metricas.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}"
    )
)

print(
    f"\nMelhor baseline: "
    f"{melhor_modelo['modelo']}"
)

print(
    f"WAPE do melhor baseline: "
    f"{melhor_modelo['wape']:.2f}%"
)


# ============================================================
# 9. EXPORTACAO
# ============================================================

previsoes.to_csv(
    ARQUIVO_PREVISOES,
    index=False
)

metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False
)

print("\nArquivos gerados:")
print(f"- {ARQUIVO_PREVISOES}")
print(f"- {ARQUIVO_METRICAS}")

print("\nAvaliacao concluida com sucesso.")
