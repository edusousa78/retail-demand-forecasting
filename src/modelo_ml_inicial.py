
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARQUIVO_TREINO = PROCESSED_DIR / "treino_modelagem.csv"
ARQUIVO_VALIDACAO = PROCESSED_DIR / "validacao_modelagem.csv"

ARQUIVO_PREVISOES = (
    PROCESSED_DIR / "previsoes_random_forest.csv"
)

ARQUIVO_METRICAS = (
    PROCESSED_DIR / "metricas_random_forest.csv"
)

HORIZONTE_MAXIMO = 7

ATRIBUTOS_NUMERICOS = [
    "horizonte",
    "venda_origem",
    "lag_1",
    "lag_7",
    "media_7",
    "media_28",
    "dia_semana_alvo",
    "mes_alvo",
]

ATRIBUTOS_CATEGORICOS = ["id"]

ATRIBUTOS_MODELO = (
    ATRIBUTOS_CATEGORICOS
    + ATRIBUTOS_NUMERICOS
)


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 60)
print("PRIMEIRO MODELO DE MACHINE LEARNING - NOVAVAREJO")
print("=" * 60)

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

assert not treino.empty
assert not validacao.empty

assert treino["date"].max() < validacao["date"].min()

assert set(treino["id"]) == set(validacao["id"])

assert not treino.duplicated(
    subset=["id", "date"]
).any()

assert not validacao.duplicated(
    subset=["id", "date"]
).any()

assert (treino["vendas"] >= 0).all()
assert (validacao["vendas"] >= 0).all()

assert validacao["date"].nunique() == HORIZONTE_MAXIMO

assert (
    validacao.groupby("id")["date"].nunique()
    == HORIZONTE_MAXIMO
).all()


# ============================================================
# 3. ENGENHARIA DE ATRIBUTOS
# ============================================================

historico = treino.rename(
    columns={"date": "data_origem"}
).copy()

grupo = historico.groupby(
    "id",
    sort=False
)["vendas"]

historico["venda_origem"] = (
    historico["vendas"].astype(float)
)

historico["lag_1"] = grupo.shift(1)

historico["lag_7"] = grupo.shift(7)

historico["media_7"] = grupo.transform(
    lambda serie: (
        serie.rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)

historico["media_28"] = grupo.transform(
    lambda serie: (
        serie.rolling(
            window=28,
            min_periods=28
        ).mean()
    )
)


# ============================================================
# 4. EXEMPLOS SUPERVISIONADOS DE TREINAMENTO
# ============================================================

print("\nConstruindo exemplos supervisionados...")

exemplos = []

for horizonte in range(1, HORIZONTE_MAXIMO + 1):

    parte = historico[
        [
            "id",
            "data_origem",
            "venda_origem",
            "lag_1",
            "lag_7",
            "media_7",
            "media_28",
        ]
    ].copy()

    parte["horizonte"] = horizonte

    parte["data_alvo"] = (
        parte["data_origem"]
        + pd.Timedelta(days=horizonte)
    )

    # O alvo pertence somente ao conjunto de treino.
    parte["alvo"] = grupo.shift(-horizonte)

    parte["dia_semana_alvo"] = (
        parte["data_alvo"].dt.day_of_week
    )

    parte["mes_alvo"] = (
        parte["data_alvo"].dt.month
    )

    exemplos.append(parte)

dados_supervisionados = pd.concat(
    exemplos,
    ignore_index=True
)

dados_supervisionados = (
    dados_supervisionados.dropna(
        subset=ATRIBUTOS_MODELO + ["alvo"]
    ).reset_index(drop=True)
)

assert not dados_supervisionados.empty

assert (
    dados_supervisionados["data_alvo"].max()
    <= treino["date"].max()
), "Alvos futuros foram incluidos no treinamento."

assert (
    dados_supervisionados["data_origem"]
    < dados_supervisionados["data_alvo"]
).all()

print(
    f"Exemplos de treinamento: "
    f"{len(dados_supervisionados)}"
)


# ============================================================
# 5. PREPARACAO DA VALIDACAO
# ============================================================

print("\nPreparando previsoes futuras...")

# Ultima observacao conhecida de cada serie.
origens = (
    historico.groupby("id", sort=False)
    .tail(1)
    .copy()
)

assert (
    origens["data_origem"].nunique() == 1
), "As series possuem datas finais diferentes."

partes_futuras = []

for horizonte in range(1, HORIZONTE_MAXIMO + 1):

    parte = origens.copy()

    parte["horizonte"] = horizonte

    parte["data_alvo"] = (
        parte["data_origem"]
        + pd.Timedelta(days=horizonte)
    )

    parte["dia_semana_alvo"] = (
        parte["data_alvo"].dt.day_of_week
    )

    parte["mes_alvo"] = (
        parte["data_alvo"].dt.month
    )

    partes_futuras.append(parte)

futuro = pd.concat(
    partes_futuras,
    ignore_index=True
)

# As vendas reais entram apenas para avaliacao.
real_validacao = validacao[
    ["id", "date", "vendas"]
].rename(
    columns={
        "date": "data_alvo",
        "vendas": "vendas_reais",
    }
)

futuro = futuro.merge(
    real_validacao,
    on=["id", "data_alvo"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert futuro["_merge"].eq("both").all()

futuro = futuro.drop(columns=["_merge"])

assert len(futuro) == len(validacao)

assert futuro[ATRIBUTOS_MODELO].notna().all().all()


# ============================================================
# 6. TREINAMENTO DO RANDOM FOREST
# ============================================================

print("\nTreinando Random Forest...")

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ATRIBUTOS_CATEGORICOS,
        ),
        (
            "numericas",
            "passthrough",
            ATRIBUTOS_NUMERICOS,
        ),
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessamento),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

X_treino = dados_supervisionados[
    ATRIBUTOS_MODELO
]

y_treino = dados_supervisionados["alvo"]

modelo.fit(X_treino, y_treino)

print("Treinamento concluido.")


# ============================================================
# 7. PREVISOES DE SETE DIAS
# ============================================================

X_validacao = futuro[ATRIBUTOS_MODELO]

previsao_ml = modelo.predict(X_validacao)

# Quantidades vendidas nao podem ser negativas.
futuro["previsao_random_forest"] = np.maximum(
    previsao_ml,
    0,
)

# Baseline: media dos sete dias conhecidos.
futuro["baseline_media_7"] = futuro["media_7"]

assert (
    futuro.groupby("id")["baseline_media_7"]
    .nunique()
    .eq(1)
    .all()
)

assert (
    futuro["previsao_random_forest"] >= 0
).all()


# ============================================================
# 8. METRICAS DE AVALIACAO
# ============================================================

def calcular_mae(real, previsto):
    return float(
        np.mean(np.abs(real - previsto))
    )


def calcular_wape(real, previsto):
    denominador = np.sum(np.abs(real))

    if denominador == 0:
        return np.nan

    return float(
        100
        * np.sum(np.abs(real - previsto))
        / denominador
    )


real = futuro["vendas_reais"].to_numpy(
    dtype=float
)

resultados = []

modelos_avaliados = [
    "baseline_media_7",
    "previsao_random_forest",
]

print("\n" + "=" * 60)
print("RESULTADOS DA VALIDACAO")
print("=" * 60)

print(f"Series avaliadas: {futuro['id'].nunique()}")
print(f"Horizonte: {HORIZONTE_MAXIMO} dias")
print(f"Observacoes: {len(futuro)}")
print(f"Vendas reais: {int(real.sum())}")

for nome in modelos_avaliados:

    previsto = futuro[nome].to_numpy(
        dtype=float
    )

    mae = calcular_mae(real, previsto)
    wape = calcular_wape(real, previsto)

    resultados.append({
        "modelo": nome,
        "mae": mae,
        "wape": wape,
    })

    print(f"\nModelo: {nome}")
    print(f"MAE: {mae:.4f}")
    print(f"WAPE: {wape:.2f}%")


# ============================================================
# 9. COMPARACAO DOS RESULTADOS
# ============================================================

metricas = pd.DataFrame(resultados)

metricas = metricas.sort_values(
    ["wape", "mae"]
).reset_index(drop=True)

wape_baseline = metricas.loc[
    metricas["modelo"] == "baseline_media_7",
    "wape"
].iloc[0]

wape_ml = metricas.loc[
    metricas["modelo"] == "previsao_random_forest",
    "wape"
].iloc[0]

ganho_relativo = (
    (wape_baseline - wape_ml)
    / wape_baseline
) * 100

print("\n" + "=" * 60)
print("COMPARACAO COM O BASELINE")
print("=" * 60)

print(f"WAPE baseline: {wape_baseline:.2f}%")
print(f"WAPE Random Forest: {wape_ml:.2f}%")

print(
    f"Variacao relativa do erro: "
    f"{ganho_relativo:+.2f}%"
)

if wape_ml < wape_baseline:
    print("Random Forest superou o baseline.")
elif wape_ml > wape_baseline:
    print("Random Forest nao superou o baseline.")
else:
    print("Os modelos apresentaram o mesmo WAPE.")


# ============================================================
# 10. EXPORTACAO
# ============================================================

colunas_saida = [
    "id",
    "data_origem",
    "data_alvo",
    "horizonte",
    "vendas_reais",
    "baseline_media_7",
    "previsao_random_forest",
]

futuro[colunas_saida].to_csv(
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

print("\nModelo inicial concluido com sucesso.")
