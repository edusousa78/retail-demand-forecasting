
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
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ARQUIVO_TREINO = RAW_DIR / "sales_train_validation.csv"
ARQUIVO_TESTE = RAW_DIR / "sales_train_evaluation.csv"
ARQUIVO_CALENDARIO = RAW_DIR / "calendar.csv"

PRIMEIRO_DIA = 1734
ULTIMO_DIA_TREINO = 1913
HORIZONTE = 7
QUANTIDADE_SERIES = 30

DIAS_TREINO = [
    f"d_{dia}"
    for dia in range(
        PRIMEIRO_DIA,
        ULTIMO_DIA_TREINO + 1
    )
]

DIAS_TESTE = [
    f"d_{dia}"
    for dia in range(
        ULTIMO_DIA_TREINO + 1,
        ULTIMO_DIA_TREINO + HORIZONTE + 1
    )
]

DIAS_CONFERENCIA = [
    f"d_{dia}"
    for dia in range(
        ULTIMO_DIA_TREINO - 6,
        ULTIMO_DIA_TREINO + 1
    )
]

ATRIBUTOS_ORIGINAIS = [
    "horizonte",
    "venda_origem",
    "lag_1",
    "lag_7",
    "media_7",
    "media_28",
    "dia_semana_alvo",
    "mes_alvo",
]

ATRIBUTOS_NOVOS = [
    "lag_14",
    "media_14",
    "desvio_7",
    "tendencia_7_28",
    "fim_semana_alvo",
]

ATRIBUTOS_APRIMORADOS = (
    ATRIBUTOS_ORIGINAIS + ATRIBUTOS_NOVOS
)


# ============================================================
# 2. FUNCOES
# ============================================================

def criar_modelo(atributos_numericos):
    transformador = ColumnTransformer([
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            ["id"],
        ),
        (
            "numericas",
            "passthrough",
            atributos_numericos,
        ),
    ])

    floresta = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessamento", transformador),
        ("random_forest", floresta),
    ])


def preparar_historico(treino):
    historico = (
        treino.rename(
            columns={"date": "data_origem"}
        )
        .sort_values(["id", "data_origem"])
        .reset_index(drop=True)
    )

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

    for dias in (7, 14, 28):
        historico[f"media_{dias}"] = (
            grupo.transform(
                lambda serie: serie.rolling(
                    window=dias,
                    min_periods=dias
                ).mean()
            )
        )

    historico["desvio_7"] = (
        grupo.transform(
            lambda serie: serie.rolling(
                window=7,
                min_periods=7
            ).std(ddof=0)
        )
    )

    historico["tendencia_7_28"] = (
        historico["media_7"]
        - historico["media_28"]
    )

    return historico


def adicionar_calendario_alvo(dados):
    dados["dia_semana_alvo"] = (
        dados["data_alvo"].dt.dayofweek
    )

    dados["mes_alvo"] = (
        dados["data_alvo"].dt.month
    )

    dados["fim_semana_alvo"] = (
        dados["dia_semana_alvo"] >= 5
    ).astype(int)

    return dados


def criar_exemplos(historico, atributos):
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

        parte = adicionar_calendario_alvo(parte)

        exemplos.append(parte)

    resultado = pd.concat(
        exemplos,
        ignore_index=True
    )

    resultado = resultado.dropna(
        subset=["id", "alvo"] + atributos
    ).reset_index(drop=True)

    assert not resultado.empty

    return resultado


def criar_previsoes_futuras(historico):
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

        parte = adicionar_calendario_alvo(parte)

        partes.append(parte)

    futuro = pd.concat(
        partes,
        ignore_index=True
    )

    assert futuro[
        ["id"] + ATRIBUTOS_APRIMORADOS
    ].notna().all().all()

    return futuro


def avaliar(real, previsto):
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)

    erro = previsto - real

    mae = float(np.abs(erro).mean())

    volume_real = float(np.abs(real).sum())

    wape = (
        float(100 * np.abs(erro).sum() / volume_real)
        if volume_real > 0
        else np.nan
    )

    return {
        "mae": mae,
        "wape": wape,
        "vies_medio": float(erro.mean()),
        "erro_absoluto_total": float(np.abs(erro).sum()),
    }


# ============================================================
# 3. VERIFICAR ARQUIVOS
# ============================================================

print("=" * 60)
print("AVALIACAO FINAL FORA DA AMOSTRA - NOVAVAREJO")
print("=" * 60)

for arquivo in [
    ARQUIVO_TREINO,
    ARQUIVO_TESTE,
    ARQUIVO_CALENDARIO,
]:
    if not arquivo.is_file():
        raise FileNotFoundError(
            f"\nArquivo obrigatorio ausente:\n{arquivo}"
        )

print("\nTodos os arquivos foram encontrados.")


# ============================================================
# 4. CARREGAR HISTORICO
# ============================================================

print("\nCarregando historico de treinamento...")

tipos_treino = {
    coluna: "int32"
    for coluna in DIAS_TREINO
}

tipos_treino.update({
    "id": "string",
    "item_id": "string",
    "store_id": "string",
})

vendas = pd.read_csv(
    ARQUIVO_TREINO,
    usecols=[
        "id",
        "item_id",
        "store_id",
    ] + DIAS_TREINO,
    dtype=tipos_treino,
)

assert vendas["id"].is_unique

assert not vendas.duplicated(
    ["item_id", "store_id"]
).any()

print(f"Series disponiveis: {len(vendas)}")


# ============================================================
# 5. SELECAO SEM VAZAMENTO
# ============================================================

volumes = vendas[DIAS_TREINO].sum(axis=1)

ranking = (
    vendas[["id", "item_id", "store_id"]]
    .assign(volume_treino=volumes)
    .sort_values(
        ["volume_treino", "id"],
        ascending=[False, True]
    )
)

ids_selecionados = (
    ranking.head(QUANTIDADE_SERIES)["id"]
)

selecionadas = vendas.loc[
    vendas["id"].isin(ids_selecionados)
].copy()

assert len(selecionadas) == QUANTIDADE_SERIES

print(f"Series selecionadas: {len(selecionadas)}")


# ============================================================
# 6. CONSTRUIR DADOS DE TREINAMENTO
# ============================================================

calendario = pd.read_csv(
    ARQUIVO_CALENDARIO,
    usecols=["d", "date"],
)

calendario["date"] = pd.to_datetime(
    calendario["date"]
)

assert calendario["d"].is_unique

treino = selecionadas.melt(
    id_vars=["id", "item_id", "store_id"],
    value_vars=DIAS_TREINO,
    var_name="d",
    value_name="vendas",
)

treino = treino.merge(
    calendario,
    on="d",
    how="left",
    validate="many_to_one",
)

assert treino["date"].notna().all()

historico = preparar_historico(treino)

ultimo_dia = historico["data_origem"].max()

print(
    "Ultima data do treinamento:",
    ultimo_dia.date()
)

exemplos_original = criar_exemplos(
    historico,
    ATRIBUTOS_ORIGINAIS
)

exemplos_aprimorado = criar_exemplos(
    historico,
    ATRIBUTOS_APRIMORADOS
)

assert (
    exemplos_original["data_alvo"].max()
    <= ultimo_dia
)

assert (
    exemplos_aprimorado["data_alvo"].max()
    <= ultimo_dia
)

print(
    "Exemplos RF original:",
    len(exemplos_original)
)

print(
    "Exemplos RF aprimorada:",
    len(exemplos_aprimorado)
)


# ============================================================
# 7. TREINAR OS MODELOS
# ============================================================

print("\nTreinando Random Forest original...")

modelo_original = criar_modelo(
    ATRIBUTOS_ORIGINAIS
)

modelo_original.fit(
    exemplos_original[
        ["id"] + ATRIBUTOS_ORIGINAIS
    ],
    exemplos_original["alvo"],
)

print("Treinando Random Forest aprimorada...")

modelo_aprimorado = criar_modelo(
    ATRIBUTOS_APRIMORADOS
)

modelo_aprimorado.fit(
    exemplos_aprimorado[
        ["id"] + ATRIBUTOS_APRIMORADOS
    ],
    exemplos_aprimorado["alvo"],
)

print("Treinamentos concluidos.")


# ============================================================
# 8. GERAR PREVISOES SEM DADOS FUTUROS
# ============================================================

print("\nGerando previsoes futuras...")

futuro = criar_previsoes_futuras(historico)

futuro["baseline_media_7"] = futuro["media_7"]

futuro["previsao_rf_original"] = np.maximum(
    modelo_original.predict(
        futuro[["id"] + ATRIBUTOS_ORIGINAIS]
    ),
    0
)

futuro["previsao_rf_aprimorada"] = np.maximum(
    modelo_aprimorado.predict(
        futuro[["id"] + ATRIBUTOS_APRIMORADOS]
    ),
    0
)

assert futuro["data_alvo"].min() > ultimo_dia

assert len(futuro) == QUANTIDADE_SERIES * HORIZONTE


# ============================================================
# 9. CARREGAR DADOS FUTUROS APENAS PARA AVALIACAO
# ============================================================

print("\nCarregando vendas reais do teste...")

colunas_avaliacao = (
    ["item_id", "store_id"]
    + DIAS_CONFERENCIA
    + DIAS_TESTE
)

tipos_avaliacao = {
    coluna: "int32"
    for coluna in DIAS_CONFERENCIA + DIAS_TESTE
}

tipos_avaliacao.update({
    "item_id": "string",
    "store_id": "string",
})

avaliacao = pd.read_csv(
    ARQUIVO_TESTE,
    usecols=colunas_avaliacao,
    dtype=tipos_avaliacao,
)

assert not avaliacao.duplicated(
    ["item_id", "store_id"]
).any()

# Conferir se os dois arquivos concordam nos
# ultimos sete dias conhecidos do treinamento.
conferencia = selecionadas[
    ["item_id", "store_id"] + DIAS_CONFERENCIA
].merge(
    avaliacao[
        ["item_id", "store_id"] + DIAS_CONFERENCIA
    ],
    on=["item_id", "store_id"],
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("_treino", "_avaliacao"),
)

assert conferencia["_merge"].eq("both").all()

for dia in DIAS_CONFERENCIA:
    assert (
        conferencia[f"{dia}_treino"]
        == conferencia[f"{dia}_avaliacao"]
    ).all(), (
        f"Historico inconsistente no dia {dia}"
    )

print("Historico entre arquivos: consistente.")

reais = avaliacao.melt(
    id_vars=["item_id", "store_id"],
    value_vars=DIAS_TESTE,
    var_name="d",
    value_name="vendas_reais",
)

reais = reais.merge(
    calendario,
    on="d",
    how="left",
    validate="many_to_one",
)

assert reais["date"].notna().all()

reais = reais.rename(
    columns={"date": "data_alvo"}
)

resultado = futuro.merge(
    reais[
        [
            "item_id",
            "store_id",
            "data_alvo",
            "vendas_reais",
        ]
    ],
    on=["item_id", "store_id", "data_alvo"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert resultado["_merge"].eq("both").all()

resultado = resultado.drop(columns=["_merge"])

assert len(resultado) == 210

assert resultado["vendas_reais"].notna().all()

assert resultado["data_alvo"].min() > ultimo_dia

print(
    "Periodo de teste:",
    resultado["data_alvo"].min().date(),
    "ate",
    resultado["data_alvo"].max().date()
)

print(f"Previsoes avaliadas: {len(resultado)}")


# ============================================================
# 10. METRICAS FINAIS
# ============================================================

modelos = {
    "Baseline 7 dias": "baseline_media_7",
    "RF original": "previsao_rf_original",
    "RF aprimorada": "previsao_rf_aprimorada",
}

registros = []

for nome, coluna in modelos.items():
    metricas = avaliar(
        resultado["vendas_reais"],
        resultado[coluna]
    )

    registros.append({
        "modelo": nome,
        "observacoes": len(resultado),
        **metricas,
    })

metricas_finais = pd.DataFrame(registros)

print("\n" + "=" * 60)
print("RESULTADO FINAL - TESTE INDEPENDENTE")
print("=" * 60)

print(
    metricas_finais[
        ["modelo", "mae", "wape", "vies_medio"]
    ].to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}"
    )
)


# ============================================================
# 11. RESULTADOS POR HORIZONTE
# ============================================================

registros_horizonte = []

for horizonte, grupo in resultado.groupby("horizonte"):
    for nome, coluna in modelos.items():
        metricas = avaliar(
            grupo["vendas_reais"],
            grupo[coluna]
        )

        registros_horizonte.append({
            "horizonte": horizonte,
            "modelo": nome,
            **metricas,
        })

metricas_horizonte = pd.DataFrame(
    registros_horizonte
)

print("\nWAPE POR HORIZONTE")

print(
    metricas_horizonte.pivot(
        index="horizonte",
        columns="modelo",
        values="wape"
    ).round(2).to_string()
)


# ============================================================
# 12. INTERPRETACAO DO TESTE
# ============================================================

wapes = metricas_finais.set_index(
    "modelo"
)["wape"]

wape_baseline = wapes["Baseline 7 dias"]
wape_original = wapes["RF original"]
wape_aprimorado = wapes["RF aprimorada"]

ganho_relativo = (
    100 * (wape_original - wape_aprimorado)
    / wape_original
)

print("\n" + "=" * 60)
print("DECISAO DA AVALIACAO FINAL")
print("=" * 60)

print(f"WAPE baseline: {wape_baseline:.2f}%")
print(f"WAPE RF original: {wape_original:.2f}%")
print(f"WAPE RF aprimorada: {wape_aprimorado:.2f}%")

print(
    "Variacao relativa contra RF original:",
    f"{ganho_relativo:+.2f}%"
)

if (
    wape_aprimorado < wape_original
    and wape_aprimorado < wape_baseline
):
    conclusao = (
        "A RF aprimorada superou os dois modelos "
        "de referencia no teste independente."
    )
elif wape_aprimorado < wape_baseline:
    conclusao = (
        "A RF aprimorada superou o baseline, "
        "mas nao superou a RF original."
    )
else:
    conclusao = (
        "A RF aprimorada nao superou o baseline "
        "neste teste independente."
    )

print(conclusao)


# ============================================================
# 13. EXPORTACAO E RELATORIO
# ============================================================

arquivo_previsoes = (
    PROCESSED_DIR / "avaliacao_final_previsoes.csv"
)

arquivo_metricas = (
    PROCESSED_DIR / "avaliacao_final_metricas.csv"
)

arquivo_horizontes = (
    PROCESSED_DIR / "avaliacao_final_horizontes.csv"
)

arquivo_relatorio = (
    REPORTS_DIR / "avaliacao_final.md"
)

colunas_exportacao = [
    "id",
    "item_id",
    "store_id",
    "data_alvo",
    "horizonte",
    "vendas_reais",
    "baseline_media_7",
    "previsao_rf_original",
    "previsao_rf_aprimorada",
]

resultado[
    colunas_exportacao
].sort_values(
    ["id", "data_alvo"]
).to_csv(
    arquivo_previsoes,
    index=False
)

metricas_finais.to_csv(
    arquivo_metricas,
    index=False
)

metricas_horizonte.to_csv(
    arquivo_horizontes,
    index=False
)

relatorio = f"""# Avaliacao final - NovaVarejo Brasil

## Projeto

Retail Demand Forecasting

## Protocolo

- Treinamento: 28/10/2015 a 24/04/2016.
- Teste: 25/04/2016 a 01/05/2016.
- Series: {QUANTIDADE_SERIES}.
- Horizonte: {HORIZONTE} dias.
- Previsoes: {len(resultado)}.
- Selecao de series: volume no treino.
- Modelos: baseline, RF original e RF aprimorada.

## Resultados

| Modelo | WAPE |
|---|---:|
| Baseline 7 dias | {wape_baseline:.2f}% |
| RF original | {wape_original:.2f}% |
| RF aprimorada | {wape_aprimorado:.2f}% |

## Comparacao

Variacao relativa da RF aprimorada contra a
RF original: {ganho_relativo:+.2f}%.

## Conclusao

{conclusao}

## Limitacoes

- Avaliacao concentrada em 30 series de alto volume.
- Apenas uma semana de teste independente.
- O teste nao comprova desempenho em producao.
- Novos ajustes nao devem ser selecionados usando
  repetidamente esta mesma semana de teste.

## Proximos passos

- Revisar testes automatizados.
- Documentar metodologia e resultados.
- Preparar README e publicacao no GitHub.
"""

arquivo_relatorio.write_text(
    relatorio,
    encoding="utf-8"
)

print("\nArquivos gerados:")
print(f"- {arquivo_previsoes}")
print(f"- {arquivo_metricas}")
print(f"- {arquivo_horizontes}")
print(f"- {arquivo_relatorio}")

print("\nAvaliacao final concluida com sucesso.")
