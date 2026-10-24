
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

ARQUIVOS = {
    "backtesting": (
        PROCESSED_DIR / "backtesting_ml_previsoes.csv"
    ),
    "experimento": (
        PROCESSED_DIR / "experimento_atributos_previsoes.csv"
    ),
    "final": (
        PROCESSED_DIR / "avaliacao_final_previsoes.csv"
    ),
    "metricas": (
        PROCESSED_DIR / "avaliacao_final_metricas.csv"
    ),
    "horizontes": (
        PROCESSED_DIR / "avaliacao_final_horizontes.csv"
    ),
}

MODELOS_FINAIS = {
    "Baseline 7 dias": "baseline_media_7",
    "RF original": "previsao_rf_original",
    "RF aprimorada": "previsao_rf_aprimorada",
}

CHAVES_BACKTESTING = [
    "janela",
    "id",
    "data_alvo",
    "horizonte",
]


# ============================================================
# 2. FUNCOES AUXILIARES
# ============================================================

def calcular_metricas(reais, previstos):
    reais = np.asarray(reais, dtype=float)
    previstos = np.asarray(previstos, dtype=float)

    erros = previstos - reais

    mae = np.abs(erros).mean()

    denominador = np.abs(reais).sum()

    wape = (
        100 * np.abs(erros).sum() / denominador
        if denominador > 0
        else np.nan
    )

    vies = erros.mean()

    erro_absoluto_total = np.abs(erros).sum()

    return {
        "mae": float(mae),
        "wape": float(wape),
        "vies_medio": float(vies),
        "erro_absoluto_total": float(
            erro_absoluto_total
        ),
    }


@pytest.fixture(scope="module")
def dados():
    """
    Carrega os resultados uma unica vez
    para todos os testes deste arquivo.
    """

    for nome, caminho in ARQUIVOS.items():
        assert caminho.is_file(), (
            f"Arquivo ausente ({nome}): {caminho}"
        )

    return {
        "backtesting": pd.read_csv(
            ARQUIVOS["backtesting"],
            parse_dates=[
                "data_origem",
                "data_alvo",
            ],
        ),
        "experimento": pd.read_csv(
            ARQUIVOS["experimento"],
            parse_dates=["data_alvo"],
        ),
        "final": pd.read_csv(
            ARQUIVOS["final"],
            parse_dates=["data_alvo"],
        ),
        "metricas": pd.read_csv(
            ARQUIVOS["metricas"]
        ),
        "horizontes": pd.read_csv(
            ARQUIVOS["horizontes"]
        ),
    }


# ============================================================
# 3. TESTES DO BACKTESTING
# ============================================================

def test_quantidade_previsoes_backtesting(dados):
    backtesting = dados["backtesting"]

    assert len(backtesting) == 840
    assert backtesting["janela"].nunique() == 4

    assert set(backtesting["janela"]) == {
        1, 2, 3, 4
    }

    for janela, grupo in backtesting.groupby("janela"):
        assert len(grupo) == 210
        assert grupo["id"].nunique() == 30

        assert (
            grupo.groupby("id")["horizonte"]
            .nunique()
            .eq(7)
            .all()
        )


def test_backtesting_sem_duplicatas(dados):
    backtesting = dados["backtesting"]

    assert not backtesting.duplicated(
        CHAVES_BACKTESTING
    ).any()


def test_backtesting_respeita_cronologia(dados):
    backtesting = dados["backtesting"]

    assert (
        backtesting["data_origem"]
        < backtesting["data_alvo"]
    ).all()

    diferenca = (
        backtesting["data_alvo"]
        - backtesting["data_origem"]
    ).dt.days

    assert (
        diferenca == backtesting["horizonte"]
    ).all()

    assert backtesting["horizonte"].between(
        1, 7
    ).all()

    for _, grupo in backtesting.groupby("janela"):
        assert (
            grupo["data_origem"].nunique() == 1
        )

        assert (
            grupo["data_alvo"].nunique() == 7
        )


# ============================================================
# 4. TESTES DO EXPERIMENTO
# ============================================================

def test_experimento_usa_mesmas_observacoes(dados):
    backtesting = dados["backtesting"]
    experimento = dados["experimento"]

    assert len(experimento) == 840

    assert not experimento.duplicated(
        CHAVES_BACKTESTING
    ).any()

    comparacao = backtesting.merge(
        experimento,
        on=CHAVES_BACKTESTING,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_anterior", "_experimento"),
    )

    assert len(comparacao) == 840

    assert comparacao["_merge"].eq("both").all()

    np.testing.assert_allclose(
        comparacao["vendas_reais_anterior"],
        comparacao["vendas_reais_experimento"],
        rtol=0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        comparacao["baseline_media_7_anterior"],
        comparacao["baseline_media_7_experimento"],
        rtol=0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        comparacao["previsao_random_forest"],
        comparacao["previsao_rf_original"],
        rtol=0,
        atol=1e-8,
    )


# ============================================================
# 5. TESTES DO CONJUNTO FINAL
# ============================================================

def test_quantidade_previsoes_finais(dados):
    final = dados["final"]

    assert len(final) == 210

    assert final["id"].nunique() == 30

    assert final["horizonte"].nunique() == 7

    assert not final.duplicated(
        ["id", "data_alvo"]
    ).any()

    assert (
        final.groupby("id")["horizonte"]
        .nunique()
        .eq(7)
        .all()
    )


def test_datas_do_teste_independente(dados):
    final = dados["final"]
    backtesting = dados["backtesting"]

    data_inicio = pd.Timestamp("2016-04-25")
    data_fim = pd.Timestamp("2016-05-01")
    ultima_data_treino = pd.Timestamp("2016-04-24")

    assert final["data_alvo"].min() == data_inicio
    assert final["data_alvo"].max() == data_fim

    assert (
        final["data_alvo"] > ultima_data_treino
    ).all()

    assert (
        backtesting["data_alvo"].max()
        < final["data_alvo"].min()
    )

    dias_previstos = (
        final["data_alvo"]
        - pd.to_timedelta(
            final["horizonte"],
            unit="D"
        )
    )

    assert (
        dias_previstos == ultima_data_treino
    ).all()

    assert set(final["data_alvo"]) == set(
        pd.date_range(
            data_inicio,
            data_fim,
            freq="D"
        )
    )


def test_series_finais_compativeis_com_ultima_janela(
    dados
):
    final = dados["final"]
    backtesting = dados["backtesting"]

    ultima_janela = backtesting.loc[
        backtesting["janela"] == 4
    ]

    assert set(final["id"]) == set(
        ultima_janela["id"]
    )


def test_previsoes_sem_valores_invalidos(dados):
    final = dados["final"]

    colunas = [
        "vendas_reais",
        "baseline_media_7",
        "previsao_rf_original",
        "previsao_rf_aprimorada",
    ]

    assert final[colunas].notna().all().all()

    valores = final[colunas].to_numpy(
        dtype=float
    )

    assert np.isfinite(valores).all()

    assert (valores >= 0).all()


# ============================================================
# 6. TESTES DAS METRICAS GLOBAIS
# ============================================================

def test_metricas_globais_consistentes(dados):
    final = dados["final"]
    metricas = dados["metricas"]

    assert len(metricas) == 3

    assert set(metricas["modelo"]) == set(
        MODELOS_FINAIS
    )

    assert metricas["modelo"].is_unique

    for nome, coluna in MODELOS_FINAIS.items():
        calculadas = calcular_metricas(
            final["vendas_reais"],
            final[coluna],
        )

        registrada = metricas.loc[
            metricas["modelo"] == nome
        ].iloc[0]

        assert int(
            registrada["observacoes"]
        ) == len(final)

        for indicador, valor in calculadas.items():
            assert float(
                registrada[indicador]
            ) == pytest.approx(
                valor,
                abs=1e-8
            )


# ============================================================
# 7. TESTES DAS METRICAS POR HORIZONTE
# ============================================================

def test_metricas_horizontes_consistentes(dados):
    final = dados["final"]
    horizontes = dados["horizontes"]

    assert len(horizontes) == 21

    assert not horizontes.duplicated(
        ["horizonte", "modelo"]
    ).any()

    for horizonte in range(1, 8):
        grupo = final.loc[
            final["horizonte"] == horizonte
        ]

        assert len(grupo) == 30

        for nome, coluna in MODELOS_FINAIS.items():
            calculadas = calcular_metricas(
                grupo["vendas_reais"],
                grupo[coluna],
            )

            registro = horizontes.loc[
                (
                    horizontes["horizonte"]
                    == horizonte
                )
                & (
                    horizontes["modelo"]
                    == nome
                )
            ]

            assert len(registro) == 1

            for indicador, valor in calculadas.items():
                assert float(
                    registro.iloc[0][indicador]
                ) == pytest.approx(
                    valor,
                    abs=1e-8
                )


# ============================================================
# 8. TESTE DA DOCUMENTACAO
# ============================================================

def test_relatorios_gerados():
    arquivos = [
        REPORTS_DIR / "decisao_modelo_aula15.md",
        REPORTS_DIR / "avaliacao_final.md",
    ]

    for arquivo in arquivos:
        assert arquivo.is_file(), (
            f"Relatorio ausente: {arquivo}"
        )

        conteudo = arquivo.read_text(
            encoding="utf-8"
        )

        assert len(conteudo.strip()) > 100
