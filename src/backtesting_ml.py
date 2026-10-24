
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

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ARQUIVO_VENDAS = RAW_DIR / "sales_train_validation.csv"
ARQUIVO_CALENDARIO = RAW_DIR / "calendar.csv"

QUANTIDADE_SERIES = 30
DIAS_HISTORICO = 180
HORIZONTE_MAXIMO = 7
NUMERO_JANELAS = 4

ULTIMO_DIA = 1913
PRIMEIRO_DIA = ULTIMO_DIA - DIAS_HISTORICO + 1

COLUNAS_ID = [
    "id",
    "item_id",
    "store_id",
]

COLUNAS_DIAS = [
    f"d_{dia}"
    for dia in range(PRIMEIRO_DIA, ULTIMO_DIA + 1)
]

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

MODELOS = [
    "baseline_media_7",
    "previsao_random_forest",
]


# ============================================================
# 2. FUNCOES AUXILIARES
# ============================================================

def calcular_mae(real, previsto):
    return float(
        np.mean(np.abs(real - previsto))
    )


def calcular_wape(real, previsto):
    denominador = np.abs(real).sum()

    if denominador == 0:
        return np.nan

    return float(
        100 * np.abs(real - previsto).sum()
        / denominador
    )


def criar_modelo():
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

    return Pipeline(
        steps=[
            (
                "preprocessamento",
                preprocessamento
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ]
    )


def criar_atributos(treino):
    historico = treino.rename(
        columns={"date": "data_origem"}
    ).copy()

    historico = historico.sort_values(
        ["id", "data_origem"]
    ).reset_index(drop=True)

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
        lambda serie: serie.rolling(
            window=7,
            min_periods=7
        ).mean()
    )

    historico["media_28"] = grupo.transform(
        lambda serie: serie.rolling(
            window=28,
            min_periods=28
        ).mean()
    )

    return historico


def criar_exemplos_treinamento(historico):
    exemplos = []

    grupo = historico.groupby(
        "id",
        sort=False
    )["vendas"]

    for horizonte in range(
        1,
        HORIZONTE_MAXIMO + 1
    ):
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

        parte["alvo"] = grupo.shift(
            -horizonte
        )

        parte["dia_semana_alvo"] = (
            parte["data_alvo"].dt.day_of_week
        )

        parte["mes_alvo"] = (
            parte["data_alvo"].dt.month
        )

        exemplos.append(parte)

    dados = pd.concat(
        exemplos,
        ignore_index=True
    )

    dados = dados.dropna(
        subset=ATRIBUTOS_MODELO + ["alvo"]
    ).reset_index(drop=True)

    return dados


def criar_previsoes_futuras(historico):
    origens = (
        historico.groupby(
            "id",
            sort=False
        )
        .tail(1)
        .copy()
    )

    partes = []

    for horizonte in range(
        1,
        HORIZONTE_MAXIMO + 1
    ):
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

        partes.append(parte)

    return pd.concat(
        partes,
        ignore_index=True
    )


# ============================================================
# 3. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 60)
print("BACKTESTING DE MACHINE LEARNING - NOVAVAREJO")
print("=" * 60)

print("\nCarregando dados originais...")

tipos = {
    coluna: "int32"
    for coluna in COLUNAS_DIAS
}

tipos.update({
    coluna: "string"
    for coluna in COLUNAS_ID
})

vendas = pd.read_csv(
    ARQUIVO_VENDAS,
    usecols=COLUNAS_ID + COLUNAS_DIAS,
    dtype=tipos,
)

calendario = pd.read_csv(
    ARQUIVO_CALENDARIO,
    usecols=["d", "date"],
)

calendario["date"] = pd.to_datetime(
    calendario["date"]
)

assert vendas["id"].is_unique
assert calendario["d"].is_unique

assert (
    vendas[COLUNAS_DIAS] >= 0
).all().all()

print(f"Series disponiveis: {len(vendas)}")
print(f"Historico carregado: {DIAS_HISTORICO} dias")


# ============================================================
# 4. BACKTESTING TEMPORAL
# ============================================================

todas_previsoes = []
todas_metricas = []

for janela in range(
    1,
    NUMERO_JANELAS + 1
):

    print("\n" + "-" * 60)
    print(f"JANELA {janela}")
    print("-" * 60)

    # --------------------------------------------------------
    # 4.1. Definir corte temporal
    # --------------------------------------------------------

    dias_ate_fim = (
        NUMERO_JANELAS - janela + 1
    ) * HORIZONTE_MAXIMO

    ultimo_dia_treino = (
        ULTIMO_DIA - dias_ate_fim
    )

    dias_treino = [
        f"d_{dia}"
        for dia in range(
            PRIMEIRO_DIA,
            ultimo_dia_treino + 1
        )
    ]

    dias_validacao = [
        f"d_{dia}"
        for dia in range(
            ultimo_dia_treino + 1,
            ultimo_dia_treino
            + HORIZONTE_MAXIMO + 1
        )
    ]

    assert len(dias_treino) >= 35
    assert len(dias_validacao) == 7

    # --------------------------------------------------------
    # 4.2. Selecionar series apenas com o passado
    # --------------------------------------------------------

    volume_treino = vendas[
        dias_treino
    ].sum(axis=1)

    ranking = (
        vendas[COLUNAS_ID]
        .assign(
            volume_treino=volume_treino
        )
        .sort_values(
            ["volume_treino", "id"],
            ascending=[False, True]
        )
    )

    ids_selecionados = (
        ranking.head(
            QUANTIDADE_SERIES
        )["id"]
    )

    selecionadas = vendas.loc[
        vendas["id"].isin(
            ids_selecionados
        ),
        COLUNAS_ID
        + dias_treino
        + dias_validacao
    ].copy()

    assert (
        len(selecionadas)
        == QUANTIDADE_SERIES
    )

    # --------------------------------------------------------
    # 4.3. Converter dados para formato longo
    # --------------------------------------------------------

    dados_longos = selecionadas.melt(
        id_vars=COLUNAS_ID,
        value_vars=(
            dias_treino + dias_validacao
        ),
        var_name="d",
        value_name="vendas",
    )

    dados_longos = dados_longos.merge(
        calendario,
        on="d",
        how="left",
        validate="many_to_one",
    )

    assert dados_longos["date"].notna().all()

    dados_longos = dados_longos.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4.4. Separar treino e validacao
    # --------------------------------------------------------

    treino = dados_longos[
        dados_longos["d"].isin(
            dias_treino
        )
    ].copy()

    validacao = dados_longos[
        dados_longos["d"].isin(
            dias_validacao
        )
    ].copy()

    assert (
        treino["date"].max()
        < validacao["date"].min()
    )

    assert len(validacao) == (
        QUANTIDADE_SERIES
        * HORIZONTE_MAXIMO
    )

    assert (
        validacao.groupby("id")["date"]
        .nunique()
        .eq(HORIZONTE_MAXIMO)
        .all()
    )

    # --------------------------------------------------------
    # 4.5. Criar atributos sem vazamento
    # --------------------------------------------------------

    historico = criar_atributos(
        treino
    )

    exemplos = criar_exemplos_treinamento(
        historico
    )

    futuro = criar_previsoes_futuras(
        historico
    )

    assert not exemplos.empty

    assert (
        exemplos["data_alvo"].max()
        <= treino["date"].max()
    ), "Vazamento nos alvos de treinamento."

    assert (
        exemplos["data_origem"]
        < exemplos["data_alvo"]
    ).all()

    assert (
        futuro["data_origem"].max()
        < futuro["data_alvo"].min()
    )

    assert (
        futuro[ATRIBUTOS_MODELO]
        .notna()
        .all()
        .all()
    )

    # --------------------------------------------------------
    # 4.6. Treinar Random Forest
    # --------------------------------------------------------

    print(
        f"Exemplos de treino: "
        f"{len(exemplos)}"
    )

    modelo = criar_modelo()

    X_treino = exemplos[
        ATRIBUTOS_MODELO
    ]

    y_treino = exemplos["alvo"]

    modelo.fit(
        X_treino,
        y_treino
    )

    # --------------------------------------------------------
    # 4.7. Gerar previsoes
    # --------------------------------------------------------

    X_futuro = futuro[
        ATRIBUTOS_MODELO
    ]

    futuro["previsao_random_forest"] = (
        np.maximum(
            modelo.predict(X_futuro),
            0
        )
    )

    futuro["baseline_media_7"] = (
        futuro["media_7"]
    )

    # --------------------------------------------------------
    # 4.8. Adicionar valores reais para avaliacao
    # --------------------------------------------------------

    reais = validacao[
        ["id", "date", "vendas"]
    ].rename(
        columns={
            "date": "data_alvo",
            "vendas": "vendas_reais",
        }
    )

    previsoes = futuro.merge(
        reais,
        on=["id", "data_alvo"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert previsoes["_merge"].eq(
        "both"
    ).all()

    previsoes = previsoes.drop(
        columns=["_merge"]
    )

    assert len(previsoes) == (
        QUANTIDADE_SERIES
        * HORIZONTE_MAXIMO
    )

    assert (
        previsoes["vendas_reais"]
        .notna()
        .all()
    )

    assert (
        previsoes[MODELOS]
        .notna()
        .all()
        .all()
    )

    previsoes["janela"] = janela

    # --------------------------------------------------------
    # 4.9. Avaliar modelos
    # --------------------------------------------------------

    real = previsoes[
        "vendas_reais"
    ].to_numpy(dtype=float)

    print(
        f"Treino ate: "
        f"{treino['date'].max().date()}"
    )

    print(
        f"Validacao: "
        f"{validacao['date'].min().date()} "
        f"ate "
        f"{validacao['date'].max().date()}"
    )

    print(
        f"Vendas reais: "
        f"{int(real.sum())}"
    )

    for nome_modelo in MODELOS:

        previsto = previsoes[
            nome_modelo
        ].to_numpy(dtype=float)

        mae = calcular_mae(
            real,
            previsto
        )

        wape = calcular_wape(
            real,
            previsto
        )

        erro_absoluto = float(
            np.abs(real - previsto).sum()
        )

        todas_metricas.append({
            "janela": janela,
            "modelo": nome_modelo,
            "mae": mae,
            "wape": wape,
            "erro_absoluto": erro_absoluto,
            "vendas_reais": float(
                real.sum()
            ),
            "observacoes": len(real),
        })

        print(
            f"{nome_modelo}: "
            f"MAE={mae:.4f} | "
            f"WAPE={wape:.2f}%"
        )

    todas_previsoes.append(
        previsoes[
            [
                "janela",
                "id",
                "data_origem",
                "data_alvo",
                "horizonte",
                "vendas_reais",
                "baseline_media_7",
                "previsao_random_forest",
            ]
        ].copy()
    )


# ============================================================
# 5. CONSOLIDACAO
# ============================================================

previsoes_finais = pd.concat(
    todas_previsoes,
    ignore_index=True
)

metricas = pd.DataFrame(
    todas_metricas
)

resumo = (
    metricas.groupby(
        "modelo",
        as_index=False
    )
    .agg(
        erro_absoluto=(
            "erro_absoluto",
            "sum"
        ),
        vendas_reais=(
            "vendas_reais",
            "sum"
        ),
        observacoes=(
            "observacoes",
            "sum"
        ),
    )
)

resumo["mae_global"] = (
    resumo["erro_absoluto"]
    / resumo["observacoes"]
)

resumo["wape_global"] = np.where(
    resumo["vendas_reais"] > 0,
    100 * resumo["erro_absoluto"]
    / resumo["vendas_reais"],
    np.nan,
)

resumo = resumo.sort_values(
    ["wape_global", "mae_global"]
).reset_index(drop=True)

assert len(previsoes_finais) == (
    NUMERO_JANELAS
    * QUANTIDADE_SERIES
    * HORIZONTE_MAXIMO
)


# ============================================================
# 6. RESULTADOS FINAIS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADO CONSOLIDADO - MACHINE LEARNING")
print("=" * 60)

print(
    f"Janelas avaliadas: "
    f"{NUMERO_JANELAS}"
)

print(
    f"Previsoes avaliadas: "
    f"{len(previsoes_finais)}"
)

print(
    "\n"
    + resumo[
        [
            "modelo",
            "mae_global",
            "wape_global",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

wape_baseline = resumo.loc[
    resumo["modelo"] == "baseline_media_7",
    "wape_global"
].iloc[0]

wape_ml = resumo.loc[
    resumo["modelo"] == "previsao_random_forest",
    "wape_global"
].iloc[0]

ganho_relativo = (
    (wape_baseline - wape_ml)
    / wape_baseline
) * 100

print("\nCOMPARACAO FINAL")
print(f"WAPE baseline: {wape_baseline:.2f}%")
print(f"WAPE Random Forest: {wape_ml:.2f}%")

print(
    f"Reducao relativa do erro: "
    f"{ganho_relativo:+.2f}%"
)

if wape_ml < wape_baseline:
    print(
        "Resultado: Random Forest superou "
        "o baseline no consolidado."
    )
elif wape_ml > wape_baseline:
    print(
        "Resultado: Random Forest nao superou "
        "o baseline no consolidado."
    )
else:
    print(
        "Resultado: desempenho equivalente."
    )


# ============================================================
# 7. EXPORTACAO
# ============================================================

ARQUIVO_PREVISOES = (
    PROCESSED_DIR
    / "backtesting_ml_previsoes.csv"
)

ARQUIVO_METRICAS = (
    PROCESSED_DIR
    / "backtesting_ml_metricas.csv"
)

ARQUIVO_RESUMO = (
    PROCESSED_DIR
    / "backtesting_ml_resumo.csv"
)

previsoes_finais.to_csv(
    ARQUIVO_PREVISOES,
    index=False
)

metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False
)

resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)

print("\nArquivos gerados:")
print(f"- {ARQUIVO_PREVISOES}")
print(f"- {ARQUIVO_METRICAS}")
print(f"- {ARQUIVO_RESUMO}")

print("\nBacktesting de ML concluido com sucesso.")
