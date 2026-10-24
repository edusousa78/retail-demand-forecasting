
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

ARQUIVO_VENDAS = RAW_DIR / "sales_train_validation.csv"
ARQUIVO_CALENDARIO = RAW_DIR / "calendar.csv"

ARQUIVO_REFERENCIA = (
    PROCESSED_DIR / "backtesting_ml_previsoes.csv"
)

QUANTIDADE_SERIES = 30
DIAS_HISTORICO = 180
HORIZONTE = 7
NUMERO_JANELAS = 4
ULTIMO_DIA = 1913

PRIMEIRO_DIA = ULTIMO_DIA - DIAS_HISTORICO + 1

COLUNAS_ID = ["id", "item_id", "store_id"]

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
    "lag_14",
    "media_14",
    "desvio_7",
    "tendencia_7_28",
    "fim_semana_alvo",
]

ATRIBUTOS = ["id"] + ATRIBUTOS_NUMERICOS

COLUNAS_PREVISAO = [
    "baseline_media_7",
    "previsao_rf_original",
    "previsao_rf_atributos",
]


# ============================================================
# 2. FUNCOES AUXILIARES
# ============================================================

def calcular_metricas(real, previsto):
    erro = np.abs(real - previsto)
    total_real = np.abs(real).sum()

    mae = float(erro.mean())

    wape = (
        float(100 * erro.sum() / total_real)
        if total_real > 0
        else np.nan
    )

    vies = float((previsto - real).mean())

    return {
        "mae": mae,
        "wape": wape,
        "vies_medio": vies,
    }


def criar_modelo():
    preprocessamento = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                ["id"],
            ),
            (
                "numericas",
                "passthrough",
                ATRIBUTOS_NUMERICOS,
            ),
        ]
    )

    return Pipeline([
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
    ])


def preparar_historico(treino):
    historico = treino.rename(
        columns={"date": "data_origem"}
    ).sort_values(
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
    historico["lag_14"] = grupo.shift(14)

    for dias in [7, 14, 28]:
        historico[f"media_{dias}"] = grupo.transform(
            lambda serie: serie.rolling(
                window=dias,
                min_periods=dias,
            ).mean()
        )

    historico["desvio_7"] = grupo.transform(
        lambda serie: serie.rolling(
            window=7,
            min_periods=7,
        ).std(ddof=0)
    )

    historico["tendencia_7_28"] = (
        historico["media_7"]
        - historico["media_28"]
    )

    return historico


def adicionar_atributos_alvo(dados):
    dados["dia_semana_alvo"] = (
        dados["data_alvo"].dt.day_of_week
    )

    dados["mes_alvo"] = (
        dados["data_alvo"].dt.month
    )

    dados["fim_semana_alvo"] = (
        dados["dia_semana_alvo"] >= 5
    ).astype(int)

    return dados


def criar_treinamento(historico):
    exemplos = []

    grupo = historico.groupby(
        "id",
        sort=False
    )["vendas"]

    colunas_origem = [
        "id",
        "data_origem",
        "venda_origem",
        "lag_1",
        "lag_7",
        "lag_14",
        "media_7",
        "media_14",
        "media_28",
        "desvio_7",
        "tendencia_7_28",
    ]

    for horizonte in range(1, HORIZONTE + 1):
        parte = historico[colunas_origem].copy()

        parte["horizonte"] = horizonte

        parte["data_alvo"] = (
            parte["data_origem"]
            + pd.Timedelta(days=horizonte)
        )

        parte["alvo"] = grupo.shift(-horizonte)

        parte = adicionar_atributos_alvo(parte)

        exemplos.append(parte)

    resultado = pd.concat(
        exemplos,
        ignore_index=True
    )

    resultado = resultado.dropna(
        subset=ATRIBUTOS + ["alvo"]
    ).reset_index(drop=True)

    assert not resultado.empty

    assert (
        resultado["data_origem"]
        < resultado["data_alvo"]
    ).all()

    return resultado


def criar_futuro(historico):
    origens = (
        historico.groupby("id", sort=False)
        .tail(1)
        .copy()
    )

    partes = []

    for horizonte in range(1, HORIZONTE + 1):
        parte = origens.copy()

        parte["horizonte"] = horizonte

        parte["data_alvo"] = (
            parte["data_origem"]
            + pd.Timedelta(days=horizonte)
        )

        parte = adicionar_atributos_alvo(parte)

        partes.append(parte)

    futuro = pd.concat(
        partes,
        ignore_index=True
    )

    assert futuro[ATRIBUTOS].notna().all().all()

    return futuro


# ============================================================
# 3. CARREGAR DADOS
# ============================================================

print("=" * 60)
print("EXPERIMENTO DE ATRIBUTOS - NOVAVAREJO")
print("=" * 60)

print("\nCarregando dados...")

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

referencia = pd.read_csv(
    ARQUIVO_REFERENCIA,
    parse_dates=["data_alvo"],
)

referencia = referencia.rename(
    columns={
        "vendas_reais": "vendas_reais_ref",
        "previsao_random_forest": "previsao_rf_original",
    }
)

assert vendas["id"].is_unique
assert calendario["d"].is_unique
assert len(referencia) == 840

print(f"Series disponiveis: {len(vendas)}")
print(f"Previsoes anteriores: {len(referencia)}")


# ============================================================
# 4. BACKTESTING DO NOVO MODELO
# ============================================================

resultados = []

for janela in range(1, NUMERO_JANELAS + 1):

    dias_ate_fim = (
        NUMERO_JANELAS - janela + 1
    ) * HORIZONTE

    ultimo_treino = ULTIMO_DIA - dias_ate_fim

    dias_treino = [
        f"d_{dia}"
        for dia in range(
            PRIMEIRO_DIA,
            ultimo_treino + 1
        )
    ]

    dias_validacao = [
        f"d_{dia}"
        for dia in range(
            ultimo_treino + 1,
            ultimo_treino + HORIZONTE + 1
        )
    ]

    # Selecionar produtos usando apenas o treino.
    volumes = vendas[dias_treino].sum(axis=1)

    ranking = (
        vendas[COLUNAS_ID]
        .assign(volume_treino=volumes)
        .sort_values(
            ["volume_treino", "id"],
            ascending=[False, True]
        )
    )

    ids = ranking.head(QUANTIDADE_SERIES)["id"]

    selecionadas = vendas.loc[
        vendas["id"].isin(ids),
        COLUNAS_ID + dias_treino + dias_validacao
    ].copy()

    assert len(selecionadas) == QUANTIDADE_SERIES

    dados = selecionadas.melt(
        id_vars=COLUNAS_ID,
        value_vars=dias_treino + dias_validacao,
        var_name="d",
        value_name="vendas",
    )

    dados = dados.merge(
        calendario,
        on="d",
        how="left",
        validate="many_to_one",
    )

    assert dados["date"].notna().all()

    dados = dados.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    treino = dados[
        dados["d"].isin(dias_treino)
    ].copy()

    validacao = dados[
        dados["d"].isin(dias_validacao)
    ].copy()

    assert treino["date"].max() < validacao["date"].min()

    # Construir atributos e exemplos supervisionados.
    historico = preparar_historico(treino)

    exemplos = criar_treinamento(historico)

    assert (
        exemplos["data_alvo"].max()
        <= treino["date"].max()
    ), "Vazamento temporal no treinamento."

    futuro = criar_futuro(historico)

    # Treinar o novo modelo.
    modelo = criar_modelo()

    modelo.fit(
        exemplos[ATRIBUTOS],
        exemplos["alvo"],
    )

    futuro["previsao_rf_atributos"] = np.maximum(
        modelo.predict(futuro[ATRIBUTOS]),
        0,
    )

    futuro["baseline_novo"] = futuro["media_7"]

    # Adicionar valores reais somente para avaliacao.
    reais = validacao[
        ["id", "date", "vendas"]
    ].rename(
        columns={
            "date": "data_alvo",
            "vendas": "vendas_reais",
        }
    )

    novo = futuro.merge(
        reais,
        on=["id", "data_alvo"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert novo["_merge"].eq("both").all()

    novo = novo.drop(columns=["_merge"])

    novo["janela"] = janela

    resultados.append(
        novo[
            [
                "janela",
                "id",
                "data_alvo",
                "horizonte",
                "vendas_reais",
                "baseline_novo",
                "previsao_rf_atributos",
            ]
        ].copy()
    )

    print(
        f"\nJanela {janela}: "
        f"{len(exemplos)} exemplos de treino, "
        f"{len(novo)} previsoes"
    )


# ============================================================
# 5. COMPARACAO COM O MODELO ORIGINAL
# ============================================================

novo = pd.concat(
    resultados,
    ignore_index=True
)

chaves = [
    "janela",
    "id",
    "data_alvo",
    "horizonte",
]

assert not novo.duplicated(chaves).any()
assert not referencia.duplicated(chaves).any()

comparacao = novo.merge(
    referencia[
        chaves
        + [
            "vendas_reais_ref",
            "baseline_media_7",
            "previsao_rf_original",
        ]
    ],
    on=chaves,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert len(comparacao) == 840

assert comparacao["_merge"].eq("both").all(), (
    "As amostras dos experimentos sao diferentes."
)

comparacao = comparacao.drop(
    columns=["_merge"]
)

assert np.allclose(
    comparacao["vendas_reais"],
    comparacao["vendas_reais_ref"],
)

assert np.allclose(
    comparacao["baseline_novo"],
    comparacao["baseline_media_7"],
)

comparacao = comparacao.drop(
    columns=[
        "vendas_reais_ref",
        "baseline_novo",
    ]
)

comparacao = comparacao.sort_values(
    chaves
).reset_index(drop=True)


# ============================================================
# 6. AVALIACAO
# ============================================================

nomes_modelos = {
    "baseline_media_7": "Baseline 7 dias",
    "previsao_rf_original": "RF original",
    "previsao_rf_atributos": "RF novos atributos",
}


def avaliar_grupo(grupo, agrupamento):
    registros = []

    real = grupo["vendas_reais"].to_numpy(
        dtype=float
    )

    for coluna, nome in nomes_modelos.items():
        previsto = grupo[coluna].to_numpy(
            dtype=float
        )

        metricas = calcular_metricas(real, previsto)

        registros.append({
            "grupo": agrupamento,
            "modelo": nome,
            "observacoes": len(grupo),
            **metricas,
        })

    return registros


metricas_janelas = []

for janela, grupo in comparacao.groupby("janela"):
    metricas_janelas.extend(
        avaliar_grupo(grupo, f"janela_{janela}")
    )

metricas_horizontes = []

for horizonte, grupo in comparacao.groupby("horizonte"):
    metricas_horizontes.extend(
        avaliar_grupo(grupo, f"horizonte_{horizonte}")
    )

metricas_globais = pd.DataFrame(
    avaliar_grupo(comparacao, "global")
)

metricas_janelas = pd.DataFrame(metricas_janelas)
metricas_horizontes = pd.DataFrame(metricas_horizontes)


# ============================================================
# 7. RESULTADOS FINAIS
# ============================================================

print("\n" + "=" * 60)
print("COMPARACAO GLOBAL")
print("=" * 60)

print(
    metricas_globais[
        ["modelo", "mae", "wape", "vies_medio"]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

wape_original = metricas_globais.loc[
    metricas_globais["modelo"] == "RF original",
    "wape"
].iloc[0]

wape_novo = metricas_globais.loc[
    metricas_globais["modelo"] == "RF novos atributos",
    "wape"
].iloc[0]

wape_baseline = metricas_globais.loc[
    metricas_globais["modelo"] == "Baseline 7 dias",
    "wape"
].iloc[0]

assert np.isclose(
    wape_original,
    22.0747,
    atol=0.01,
)

assert np.isclose(
    wape_baseline,
    27.5197,
    atol=0.01,
)

ganho = (
    (wape_original - wape_novo)
    / wape_original
) * 100

print("\nCOMPARACAO ENTRE RANDOM FORESTS")
print(f"WAPE original: {wape_original:.2f}%")
print(f"WAPE novos atributos: {wape_novo:.2f}%")

print(
    f"Reducao relativa do erro: {ganho:+.2f}%"
)

if wape_novo < wape_original:
    print("Novos atributos melhoraram o WAPE global.")
elif wape_novo > wape_original:
    print("Novos atributos pioraram o WAPE global.")
else:
    print("WAPE global equivalente.")

print("\nRESULTADOS POR JANELA")

print(
    metricas_janelas.pivot(
        index="grupo",
        columns="modelo",
        values="wape",
    ).round(2).to_string()
)

print("\nRESULTADOS POR HORIZONTE")

print(
    metricas_horizontes.pivot(
        index="grupo",
        columns="modelo",
        values="wape",
    ).round(2).to_string()
)


# ============================================================
# 8. EXPORTACAO
# ============================================================

arquivo_previsoes = (
    PROCESSED_DIR
    / "experimento_atributos_previsoes.csv"
)

arquivo_global = (
    PROCESSED_DIR
    / "experimento_atributos_global.csv"
)

arquivo_janelas = (
    PROCESSED_DIR
    / "experimento_atributos_janelas.csv"
)

arquivo_horizontes = (
    PROCESSED_DIR
    / "experimento_atributos_horizontes.csv"
)

comparacao.to_csv(
    arquivo_previsoes,
    index=False
)

metricas_globais.to_csv(
    arquivo_global,
    index=False
)

metricas_janelas.to_csv(
    arquivo_janelas,
    index=False
)

metricas_horizontes.to_csv(
    arquivo_horizontes,
    index=False
)

print("\nArquivos gerados:")
print(f"- {arquivo_previsoes}")
print(f"- {arquivo_global}")
print(f"- {arquivo_janelas}")
print(f"- {arquivo_horizontes}")

print("\nExperimento concluido com sucesso.")
