
from pathlib import Path

import numpy as np
import pandas as pd


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
DIAS_PREVISAO = 7
NUMERO_JANELAS = 4
ULTIMO_DIA = 1913

PRIMEIRO_DIA = (
    ULTIMO_DIA - DIAS_HISTORICO + 1
)

COLUNAS_ID = ["id", "item_id", "store_id"]

COLUNAS_DIAS = [
    f"d_{dia}"
    for dia in range(PRIMEIRO_DIA, ULTIMO_DIA + 1)
]

MODELOS = [
    "baseline_ultima_venda",
    "baseline_media_7",
    "baseline_media_3",
]


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 60)
print("BACKTESTING TEMPORAL - NOVAVAREJO")
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

assert calendario["d"].is_unique
assert vendas["id"].is_unique
assert (vendas[COLUNAS_DIAS] >= 0).all().all()

print(f"Series disponiveis: {len(vendas)}")
print(f"Historico carregado: {DIAS_HISTORICO} dias")


# ============================================================
# 3. FUNCOES DE AVALIACAO
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
        100 * np.sum(np.abs(real - previsto))
        / denominador
    )


# ============================================================
# 4. EXECUCAO DAS JANELAS
# ============================================================

todas_previsoes = []
metricas_janelas = []

for numero_janela in range(1, NUMERO_JANELAS + 1):

    # Data de corte de cada simulacao
    dias_ate_fim = (
        NUMERO_JANELAS - numero_janela + 1
    ) * DIAS_PREVISAO

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
            ultimo_dia_treino + DIAS_PREVISAO + 1
        )
    ]

    assert len(dias_treino) >= 7
    assert len(dias_validacao) == DIAS_PREVISAO

    # --------------------------------------------------------
    # Selecao sem vazamento temporal
    # --------------------------------------------------------

    volume_treino = vendas[dias_treino].sum(
        axis=1
    )

    ranking = (
        vendas[COLUNAS_ID]
        .assign(volume_treino=volume_treino)
        .sort_values(
            ["volume_treino", "id"],
            ascending=[False, True]
        )
    )

    ids_selecionados = (
        ranking.head(QUANTIDADE_SERIES)["id"]
    )

    selecionadas = (
        vendas.loc[
            vendas["id"].isin(ids_selecionados),
            COLUNAS_ID + dias_treino + dias_validacao
        ]
        .copy()
    )

    assert len(selecionadas) == QUANTIDADE_SERIES

    # --------------------------------------------------------
    # Previsoes calculadas apenas com o treino
    # --------------------------------------------------------

    historico = selecionadas[dias_treino]

    baselines = selecionadas[COLUNAS_ID].copy()

    baselines["baseline_ultima_venda"] = (
        historico.iloc[:, -1]
    )

    baselines["baseline_media_7"] = (
        historico.iloc[:, -7:].mean(axis=1)
    )

    baselines["baseline_media_3"] = (
        historico.iloc[:, -3:].mean(axis=1)
    )

    # --------------------------------------------------------
    # Observacoes reais dos sete dias seguintes
    # --------------------------------------------------------

    observacoes = selecionadas.melt(
        id_vars=COLUNAS_ID,
        value_vars=dias_validacao,
        var_name="d",
        value_name="vendas",
    )

    observacoes = observacoes.merge(
        calendario,
        on="d",
        how="left",
        validate="many_to_one",
    )

    previsoes = observacoes.merge(
        baselines[["id"] + MODELOS],
        on="id",
        how="left",
        validate="many_to_one",
    )

    previsoes["janela"] = numero_janela

    previsoes = previsoes.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    assert len(previsoes) == (
        QUANTIDADE_SERIES * DIAS_PREVISAO
    )

    assert previsoes["date"].notna().all()
    assert previsoes[MODELOS].notna().all().all()

    assert (
        previsoes.groupby("id")[MODELOS]
        .nunique()
        .eq(1)
        .all()
        .all()
    ), "Previsoes mudaram dentro da janela."

    ultima_data_treino = calendario.loc[
        calendario["d"] == dias_treino[-1],
        "date"
    ].iloc[0]

    primeira_data_validacao = (
        previsoes["date"].min()
    )

    assert (
        ultima_data_treino < primeira_data_validacao
    ), "Vazamento temporal identificado."

    # --------------------------------------------------------
    # Avaliacao da janela
    # --------------------------------------------------------

    real = previsoes["vendas"].to_numpy(
        dtype=float
    )

    print(f"\nJANELA {numero_janela}")
    print(
        f"Treino ate: "
        f"{ultima_data_treino.date()}"
    )

    print(
        f"Validacao: "
        f"{previsoes['date'].min().date()} "
        f"ate {previsoes['date'].max().date()}"
    )

    print(f"Vendas reais: {int(real.sum())}")

    for modelo in MODELOS:
        previsto = previsoes[modelo].to_numpy(
            dtype=float
        )

        mae = calcular_mae(real, previsto)
        wape = calcular_wape(real, previsto)

        erro_absoluto = float(
            np.abs(real - previsto).sum()
        )

        metricas_janelas.append({
            "janela": numero_janela,
            "modelo": modelo,
            "mae": mae,
            "wape": wape,
            "erro_absoluto": erro_absoluto,
            "vendas_reais": float(real.sum()),
            "observacoes": len(real),
        })

        texto_wape = (
            f"{wape:.2f}%"
            if not np.isnan(wape)
            else "indefinido"
        )

        print(
            f"{modelo}: "
            f"MAE={mae:.4f} | "
            f"WAPE={texto_wape}"
        )

    todas_previsoes.append(previsoes)


# ============================================================
# 5. CONSOLIDACAO DAS METRICAS
# ============================================================

previsoes_finais = pd.concat(
    todas_previsoes,
    ignore_index=True
)

metricas = pd.DataFrame(metricas_janelas)

resumo = (
    metricas.groupby("modelo", as_index=False)
    .agg(
        erro_absoluto=("erro_absoluto", "sum"),
        vendas_reais=("vendas_reais", "sum"),
        observacoes=("observacoes", "sum"),
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


# ============================================================
# 6. RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADO CONSOLIDADO DO BACKTESTING")
print("=" * 60)

print(f"Janelas avaliadas: {NUMERO_JANELAS}")
print(f"Previsoes avaliadas: {len(previsoes_finais)}")

print(
    "\n" +
    resumo[
        ["modelo", "mae_global", "wape_global"]
    ].to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}"
    )
)

melhor = resumo.iloc[0]

print(f"\nMelhor baseline: {melhor['modelo']}")
print(
    f"WAPE global: "
    f"{melhor['wape_global']:.2f}%"
)


# ============================================================
# 7. EXPORTACAO
# ============================================================

ARQUIVO_PREVISOES = (
    PROCESSED_DIR / "backtesting_previsoes.csv"
)

ARQUIVO_METRICAS = (
    PROCESSED_DIR / "backtesting_metricas_janelas.csv"
)

ARQUIVO_RESUMO = (
    PROCESSED_DIR / "backtesting_resumo_modelos.csv"
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

print("\nBacktesting concluido com sucesso.")
