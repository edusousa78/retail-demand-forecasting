
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARQUIVO_ENTRADA = (
    PROCESSED_DIR / "backtesting_ml_previsoes.csv"
)

ARQUIVO_HORIZONTES = (
    PROCESSED_DIR / "diagnostico_horizontes.csv"
)

ARQUIVO_JANELAS = (
    PROCESSED_DIR / "diagnostico_janelas.csv"
)

ARQUIVO_SERIES = (
    PROCESSED_DIR / "diagnostico_series.csv"
)

ARQUIVO_MAIORES_ERROS = (
    PROCESSED_DIR / "diagnostico_maiores_erros.csv"
)

ARQUIVO_RESUMO = (
    PROCESSED_DIR / "diagnostico_resumo.csv"
)

COLUNA_REAL = "vendas_reais"
COLUNA_BASELINE = "baseline_media_7"
COLUNA_MODELO = "previsao_random_forest"


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 60)
print("DIAGNOSTICO DO MODELO - NOVAVAREJO")
print("=" * 60)

print("\nCarregando previsoes do backtesting...")

dados = pd.read_csv(
    ARQUIVO_ENTRADA,
    parse_dates=[
        "data_origem",
        "data_alvo",
    ],
)

colunas_obrigatorias = [
    "janela",
    "id",
    "data_origem",
    "data_alvo",
    "horizonte",
    COLUNA_REAL,
    COLUNA_BASELINE,
    COLUNA_MODELO,
]

faltantes = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in dados.columns
]

assert not faltantes, (
    f"Colunas ausentes: {faltantes}"
)

assert not dados.empty, "Arquivo de previsoes vazio."

assert dados[colunas_obrigatorias].notna().all().all()

assert not dados.duplicated(
    subset=["janela", "id", "data_alvo"]
).any(), "Existem previsoes duplicadas."

assert (
    dados["data_origem"] < dados["data_alvo"]
).all(), "Datas de origem invalidas."

assert (
    (dados["data_alvo"] - dados["data_origem"]).dt.days
    == dados["horizonte"]
).all(), "Horizontes inconsistentes."

assert dados["horizonte"].between(1, 7).all()

assert (
    dados[
        [COLUNA_REAL, COLUNA_BASELINE, COLUNA_MODELO]
    ] >= 0
).all().all()

print(f"Observacoes: {len(dados)}")
print(f"Janelas: {dados['janela'].nunique()}")
print(f"Series distintas: {dados['id'].nunique()}")


# ============================================================
# 3. CALCULO DOS ERROS
# ============================================================

dados["erro_baseline"] = (
    dados[COLUNA_BASELINE] - dados[COLUNA_REAL]
)

dados["erro_modelo"] = (
    dados[COLUNA_MODELO] - dados[COLUNA_REAL]
)

dados["erro_abs_baseline"] = (
    dados["erro_baseline"].abs()
)

dados["erro_abs_modelo"] = (
    dados["erro_modelo"].abs()
)

dados["ganho_absoluto"] = (
    dados["erro_abs_baseline"]
    - dados["erro_abs_modelo"]
)


# ============================================================
# 4. FUNCAO DE AGRUPAMENTO
# ============================================================

def resumir_grupo(grupo):
    """
    Calcula indicadores agregados de um grupo.

    Erro positivo:
        previsao maior que venda real.

    Erro negativo:
        previsao menor que venda real.
    """

    vendas_reais = grupo[COLUNA_REAL].sum()

    erro_baseline = grupo["erro_abs_baseline"].sum()
    erro_modelo = grupo["erro_abs_modelo"].sum()

    quantidade = len(grupo)

    mae_baseline = erro_baseline / quantidade
    mae_modelo = erro_modelo / quantidade

    if vendas_reais > 0:
        wape_baseline = (
            100 * erro_baseline / vendas_reais
        )

        wape_modelo = (
            100 * erro_modelo / vendas_reais
        )
    else:
        wape_baseline = np.nan
        wape_modelo = np.nan

    if erro_baseline > 0:
        ganho_relativo = (
            100
            * (erro_baseline - erro_modelo)
            / erro_baseline
        )
    else:
        ganho_relativo = np.nan

    return pd.Series({
        "observacoes": quantidade,
        "vendas_reais": vendas_reais,
        "erro_abs_baseline": erro_baseline,
        "erro_abs_modelo": erro_modelo,
        "mae_baseline": mae_baseline,
        "mae_modelo": mae_modelo,
        "wape_baseline": wape_baseline,
        "wape_modelo": wape_modelo,
        "ganho_relativo_pct": ganho_relativo,
        "vies_medio_modelo": grupo["erro_modelo"].mean(),
    })


def agregar(dados_entrada, colunas):
    """
    Agrupa os registros e calcula metricas.
    """

    grupos = []

    for chave, grupo in dados_entrada.groupby(
        colunas,
        sort=True
    ):
        if not isinstance(chave, tuple):
            chave = (chave,)

        resultado = {
            coluna: valor
            for coluna, valor in zip(colunas, chave)
        }

        resultado.update(
            resumir_grupo(grupo).to_dict()
        )

        grupos.append(resultado)

    return pd.DataFrame(grupos)


# ============================================================
# 5. DIAGNOSTICO POR HORIZONTE
# ============================================================

print("\n" + "=" * 60)
print("DESEMPENHO POR HORIZONTE")
print("=" * 60)

por_horizonte = agregar(
    dados,
    ["horizonte"]
)

print(
    por_horizonte[
        [
            "horizonte",
            "mae_baseline",
            "mae_modelo",
            "wape_modelo",
            "ganho_relativo_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 6. DIAGNOSTICO POR JANELA
# ============================================================

print("\n" + "=" * 60)
print("DESEMPENHO POR JANELA")
print("=" * 60)

por_janela = agregar(
    dados,
    ["janela"]
)

print(
    por_janela[
        [
            "janela",
            "wape_baseline",
            "wape_modelo",
            "ganho_relativo_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 7. DIAGNOSTICO POR SERIE
# ============================================================

print("\n" + "=" * 60)
print("SERIES COM MAIOR ERRO ABSOLUTO")
print("=" * 60)

por_serie = agregar(
    dados,
    ["id"]
)

por_serie = por_serie.sort_values(
    ["erro_abs_modelo", "id"],
    ascending=[False, True]
).reset_index(drop=True)

print(
    por_serie[
        [
            "id",
            "observacoes",
            "vendas_reais",
            "erro_abs_modelo",
            "wape_modelo",
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

# Uma serie pode participar de mais de uma janela.
# A agregacao considera todas as suas participacoes.


# ============================================================
# 8. MAIORES ERROS INDIVIDUAIS
# ============================================================

print("\n" + "=" * 60)
print("DEZ MAIORES ERROS INDIVIDUAIS")
print("=" * 60)

maiores_erros = dados.sort_values(
    "erro_abs_modelo",
    ascending=False
).head(10).copy()

print(
    maiores_erros[
        [
            "janela",
            "id",
            "data_alvo",
            "horizonte",
            "vendas_reais",
            "previsao_random_forest",
            "erro_abs_modelo",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 9. ANALISE DE VIES
# ============================================================

print("\n" + "=" * 60)
print("ANALISE DE VIES DA RANDOM FOREST")
print("=" * 60)

vies_medio = dados["erro_modelo"].mean()

total_real = dados[COLUNA_REAL].sum()
total_previsto = dados[COLUNA_MODELO].sum()

diferenca_total = total_previsto - total_real

print(f"Vies medio: {vies_medio:.4f}")
print(f"Vendas reais totais: {total_real:.2f}")
print(f"Vendas previstas totais: {total_previsto:.2f}")

print(
    f"Diferenca total prevista - real: "
    f"{diferenca_total:+.2f}"
)

if vies_medio > 0:
    print(
        "Tendencia agregada: superestimacao."
    )
elif vies_medio < 0:
    print(
        "Tendencia agregada: subestimacao."
    )
else:
    print(
        "Sem vies medio agregado."
    )

print(
    "\nAtencao: erros positivos e negativos "
    "podem se compensar."
)


# ============================================================
# 10. RESUMO GLOBAL
# ============================================================

resumo = resumir_grupo(dados).to_frame().T

resumo["vendas_previstas"] = total_previsto
resumo["diferenca_total"] = diferenca_total

print("\n" + "=" * 60)
print("RESUMO GLOBAL")
print("=" * 60)

print(
    f"MAE baseline: "
    f"{resumo['mae_baseline'].iloc[0]:.4f}"
)

print(
    f"MAE Random Forest: "
    f"{resumo['mae_modelo'].iloc[0]:.4f}"
)

print(
    f"WAPE baseline: "
    f"{resumo['wape_baseline'].iloc[0]:.2f}%"
)

print(
    f"WAPE Random Forest: "
    f"{resumo['wape_modelo'].iloc[0]:.2f}%"
)

print(
    f"Reducao relativa do erro: "
    f"{resumo['ganho_relativo_pct'].iloc[0]:.2f}%"
)


# ============================================================
# 11. VALIDACAO DE CONSISTENCIA
# ============================================================

assert len(dados) == 840, (
    "Quantidade de previsoes diferente da esperada."
)

assert dados["janela"].nunique() == 4

assert dados["horizonte"].nunique() == 7

assert (
    dados.groupby(["janela", "id"])["horizonte"]
    .nunique()
    .eq(7)
    .all()
)

assert np.isclose(
    resumo["wape_modelo"].iloc[0],
    22.0747,
    atol=0.01
), "WAPE diferente do resultado da Aula 12."

assert np.isclose(
    resumo["wape_baseline"].iloc[0],
    27.5197,
    atol=0.01
), "Baseline diferente do resultado da Aula 12."


# ============================================================
# 12. EXPORTACAO
# ============================================================

por_horizonte.to_csv(
    ARQUIVO_HORIZONTES,
    index=False
)

por_janela.to_csv(
    ARQUIVO_JANELAS,
    index=False
)

por_serie.to_csv(
    ARQUIVO_SERIES,
    index=False
)

maiores_erros.to_csv(
    ARQUIVO_MAIORES_ERROS,
    index=False
)

resumo.to_csv(
    ARQUIVO_RESUMO,
    index=False
)

print("\nArquivos gerados:")
print(f"- {ARQUIVO_HORIZONTES}")
print(f"- {ARQUIVO_JANELAS}")
print(f"- {ARQUIVO_SERIES}")
print(f"- {ARQUIVO_MAIORES_ERROS}")
print(f"- {ARQUIVO_RESUMO}")

print("\nDiagnostico concluido com sucesso.")
