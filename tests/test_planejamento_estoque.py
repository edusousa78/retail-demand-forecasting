
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARQUIVO_PLANO = (
    PROCESSED_DIR / "planejamento_reposicao.csv"
)

ARQUIVO_DIARIO = (
    PROCESSED_DIR / "planejamento_demanda_diaria.csv"
)

DATA_REFERENCIA = pd.Timestamp("2016-04-24")

HORIZONTE = 7
TOTAL_SERIES = 30


# ============================================================
# 2. CARREGAMENTO
# ============================================================

@pytest.fixture(scope="module")
def dados():

    assert ARQUIVO_PLANO.is_file(), (
        f"Arquivo ausente: {ARQUIVO_PLANO}"
    )

    assert ARQUIVO_DIARIO.is_file(), (
        f"Arquivo ausente: {ARQUIVO_DIARIO}"
    )

    plano = pd.read_csv(
        ARQUIVO_PLANO,
        parse_dates=["data_referencia"],
    )

    diario = pd.read_csv(
        ARQUIVO_DIARIO,
        parse_dates=["data_alvo"],
    )

    return plano, diario


# ============================================================
# 3. ESTRUTURA DO PLANEJAMENTO
# ============================================================

def test_quantidade_series(dados):

    plano, diario = dados

    assert len(plano) == TOTAL_SERIES

    assert plano["id"].nunique() == TOTAL_SERIES

    assert len(diario) == (
        TOTAL_SERIES * HORIZONTE
    )

    assert set(plano["id"]) == set(diario["id"])


def test_sem_duplicatas(dados):

    plano, diario = dados

    assert not plano["id"].duplicated().any()

    assert not diario.duplicated(
        ["id", "data_alvo"]
    ).any()


def test_periodo_planejamento(dados):

    plano, diario = dados

    assert (
        plano["data_referencia"]
        == DATA_REFERENCIA
    ).all()

    assert diario["data_alvo"].min() == (
        pd.Timestamp("2016-04-25")
    )

    assert diario["data_alvo"].max() == (
        pd.Timestamp("2016-05-01")
    )

    for _, grupo in diario.groupby("id"):

        assert len(grupo) == HORIZONTE

        assert set(grupo["horizonte"]) == set(
            range(1, HORIZONTE + 1)
        )

        datas_esperadas = (
            DATA_REFERENCIA
            + pd.to_timedelta(
                grupo["horizonte"],
                unit="D"
            )
        )

        assert (
            grupo["data_alvo"] == datas_esperadas
        ).all()


# ============================================================
# 4. VALIDACAO DAS PREVISOES
# ============================================================

def test_previsoes_validas(dados):

    plano, diario = dados

    colunas_plano = [
        "demanda_prevista_7d",
        "estoque_atual_simulado",
        "pedido_em_transito_simulado",
        "estoque_seguranca_simulado",
        "quantidade_reposicao_sugerida",
    ]

    assert plano[
        colunas_plano
    ].notna().all().all()

    assert diario[
        "demanda_prevista"
    ].notna().all()

    assert np.isfinite(
        plano[colunas_plano].to_numpy(
            dtype=float
        )
    ).all()

    assert np.isfinite(
        diario["demanda_prevista"].to_numpy(
            dtype=float
        )
    ).all()

    assert (
        plano[colunas_plano] >= 0
    ).all().all()

    assert (
        diario["demanda_prevista"] >= 0
    ).all()


def test_demanda_diaria_corresponde_ao_plano(dados):

    plano, diario = dados

    soma_diaria = (
        diario.groupby("id")["demanda_prevista"]
        .sum()
        .sort_index()
    )

    demanda_plano = (
        plano.set_index("id")[
            "demanda_prevista_7d"
        ]
        .sort_index()
    )

    np.testing.assert_allclose(
        soma_diaria,
        demanda_plano,
        rtol=0,
        atol=0.001,
    )


# ============================================================
# 5. MOVIMENTACAO DIARIA DO ESTOQUE
# ============================================================

def test_saldo_diario_consistente(dados):

    plano, diario = dados

    parametros = plano.set_index("id")

    for identificador, grupo in diario.groupby("id"):

        grupo = grupo.sort_values("horizonte")

        registro = parametros.loc[identificador]

        saldo = float(
            registro["estoque_atual_simulado"]
        )

        for linha in grupo.itertuples(index=False):

            assert linha.saldo_inicial_dia == (
                pytest.approx(saldo, abs=0.001)
            )

            saldo = (
                saldo
                + float(linha.recebimento_previsto)
                - float(linha.demanda_prevista)
            )

            assert linha.saldo_final_projetado == (
                pytest.approx(saldo, abs=0.001)
            )

        assert registro[
            "saldo_final_sem_nova_reposicao"
        ] == pytest.approx(
            saldo,
            abs=0.001
        )


# ============================================================
# 6. REGRAS DE RUPTURA
# ============================================================

def test_classificacao_de_risco(dados):

    plano, diario = dados

    for registro in plano.itertuples(index=False):

        grupo = diario.loc[
            diario["id"] == registro.id
        ].sort_values("horizonte")

        dias_ruptura = grupo.loc[
            grupo["risco_ruptura_dia"],
            "horizonte",
        ]

        existe_ruptura = not dias_ruptura.empty

        assert bool(
            registro.risco_ruptura
        ) == existe_ruptura

        if existe_ruptura:

            primeiro_dia = int(
                dias_ruptura.min()
            )

            assert int(
                registro.dia_primeira_ruptura
            ) == primeiro_dia

            if primeiro_dia <= (
                registro.prazo_entrega_dias
            ):
                assert (
                    registro.prioridade == "CRITICA"
                )
            else:
                assert (
                    registro.prioridade == "ALTA"
                )

        else:

            assert pd.isna(
                registro.dia_primeira_ruptura
            )

            assert registro.prioridade in {
                "MEDIA",
                "BAIXA",
            }


# ============================================================
# 7. QUANTIDADE DE REPOSICAO
# ============================================================

def test_calculo_reposicao(dados):

    plano, _ = dados

    for registro in plano.itertuples(index=False):

        chegada = registro.dia_chegada_pedido

        transito_no_horizonte = (
            registro.pedido_em_transito_simulado
            if (
                pd.notna(chegada)
                and chegada <= HORIZONTE
            )
            else 0
        )

        calculo = (
            registro.demanda_prevista_7d
            + registro.estoque_seguranca_simulado
            - registro.estoque_atual_simulado
            - transito_no_horizonte
        )

        esperado = max(
            0,
            math.ceil(calculo)
        )

        assert (
            registro.quantidade_reposicao_sugerida
            == esperado
        )


# ============================================================
# 8. CONSISTENCIA DOS INDICADORES
# ============================================================

def test_indicadores_gerenciais(dados):

    plano, _ = dados

    assert plano[
        "demanda_prevista_7d"
    ].sum() == pytest.approx(
        10739.69,
        abs=0.05,
    )

    assert int(
        plano["estoque_atual_simulado"].sum()
    ) == 6515

    assert int(
        plano[
            "quantidade_reposicao_sugerida"
        ].sum()
    ) == 4279

    assert int(
        plano["risco_ruptura"].sum()
    ) == 21

    assert int(
        (plano["prioridade"] == "CRITICA").sum()
    ) == 8

    assert int(
        (plano["prioridade"] == "ALTA").sum()
    ) == 13
