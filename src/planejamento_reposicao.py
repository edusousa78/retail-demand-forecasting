
from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ARQUIVO_PREVISOES = (
    PROCESSED_DIR / "avaliacao_final_previsoes.csv"
)

ARQUIVO_HISTORICO = (
    RAW_DIR / "sales_train_validation.csv"
)

ARQUIVO_PLANO = (
    PROCESSED_DIR / "planejamento_reposicao.csv"
)

ARQUIVO_DIARIO = (
    PROCESSED_DIR / "planejamento_demanda_diaria.csv"
)

ARQUIVO_RELATORIO = (
    REPORTS_DIR / "planejamento_estoque.md"
)

# Modelo candidato escolhido antes do teste independente.
# A RF original obteve menor WAPE no teste, mas a
# superioridade da candidata nao foi confirmada.
COLUNA_PREVISAO = "previsao_rf_aprimorada"

NOME_MODELO = "Random Forest aprimorada"

HORIZONTE = 7
QUANTIDADE_SERIES = 30

DIAS_HISTORICOS = [
    f"d_{dia}"
    for dia in range(1886, 1914)
]

COLUNAS_PREVISOES = [
    "id",
    "item_id",
    "store_id",
    "data_alvo",
    "horizonte",
    COLUNA_PREVISAO,
]


# ============================================================
# 2. FUNCOES AUXILIARES
# ============================================================

def criar_gerador_aleatorio(identificador):
    """
    Cada serie recebe uma semente estavel.

    Assim, os estoques simulados permanecem
    iguais em diferentes execucoes.
    """
    resumo = hashlib.sha256(
        str(identificador).encode("utf-8")
    ).digest()

    semente = int.from_bytes(
        resumo[:8],
        byteorder="big"
    )

    return np.random.default_rng(semente)


def carregar_previsoes():
    if not ARQUIVO_PREVISOES.is_file():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_PREVISOES}\n"
            "Execute primeiro src/avaliacao_final.py."
        )

    # Importante:
    # Nao carregamos a coluna vendas_reais.
    # O planejamento utiliza somente previsoes
    # que foram geradas antes do periodo de teste.
    previsoes = pd.read_csv(
        ARQUIVO_PREVISOES,
        usecols=COLUNAS_PREVISOES,
        parse_dates=["data_alvo"],
    )

    assert len(previsoes) == (
        QUANTIDADE_SERIES * HORIZONTE
    )

    assert previsoes["id"].nunique() == (
        QUANTIDADE_SERIES
    )

    assert not previsoes.duplicated(
        ["id", "data_alvo"]
    ).any()

    assert previsoes[
        COLUNAS_PREVISOES
    ].notna().all().all()

    valores = previsoes[
        COLUNA_PREVISAO
    ].to_numpy(dtype=float)

    assert np.isfinite(valores).all()
    assert (valores >= 0).all()

    data_referencia = (
        previsoes["data_alvo"].min()
        - pd.Timedelta(days=1)
    )

    for _, grupo in previsoes.groupby("id"):
        assert set(grupo["horizonte"]) == set(
            range(1, HORIZONTE + 1)
        )

        datas_esperadas = (
            data_referencia
            + pd.to_timedelta(
                grupo["horizonte"],
                unit="D"
            )
        )

        assert (
            grupo["data_alvo"] == datas_esperadas
        ).all()

    return (
        previsoes.sort_values(
            ["id", "horizonte"]
        ).reset_index(drop=True),
        data_referencia,
    )


def carregar_historico(ids_selecionados):
    if not ARQUIVO_HISTORICO.is_file():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_HISTORICO}"
        )

    historico = pd.read_csv(
        ARQUIVO_HISTORICO,
        usecols=["id"] + DIAS_HISTORICOS,
        dtype={
            dia: "int32"
            for dia in DIAS_HISTORICOS
        },
    )

    historico = historico.loc[
        historico["id"].isin(ids_selecionados)
    ].copy()

    assert len(historico) == QUANTIDADE_SERIES
    assert historico["id"].is_unique

    valores = historico[
        DIAS_HISTORICOS
    ].to_numpy(dtype=float)

    historico["media_historica"] = (
        valores.mean(axis=1)
    )

    historico["desvio_historico"] = (
        valores.std(axis=1, ddof=0)
    )

    return historico[
        [
            "id",
            "media_historica",
            "desvio_historico",
        ]
    ]


def simular_parametros_estoque(
    identificador,
    media_historica,
    desvio_historico,
):
    """
    Simulacao didatica de dados operacionais.

    Nao representa estoque real de nenhuma loja.
    """
    rng = criar_gerador_aleatorio(identificador)

    dias_cobertura_inicial = rng.uniform(
        1.5, 8.0
    )

    estoque_atual = math.ceil(
        media_historica * dias_cobertura_inicial
    )

    prazo_entrega = int(
        rng.integers(1, 6)
    )

    tem_pedido_em_transito = (
        rng.random() < 0.55
    )

    if tem_pedido_em_transito:
        pedido_em_transito = math.ceil(
            media_historica * rng.uniform(0.5, 2.5)
        )

        chegada_pedido = int(
            rng.integers(1, HORIZONTE + 1)
        )
    else:
        pedido_em_transito = 0
        chegada_pedido = None

    # Heuristica de demonstracao, nao um intervalo
    # de confianca calibrado para o modelo.
    estoque_seguranca = math.ceil(
        1.65
        * desvio_historico
        * math.sqrt(prazo_entrega)
    )

    return {
        "estoque_atual": estoque_atual,
        "pedido_em_transito": pedido_em_transito,
        "dia_chegada_pedido": chegada_pedido,
        "prazo_entrega_dias": prazo_entrega,
        "estoque_seguranca": estoque_seguranca,
    }


# ============================================================
# 3. CARREGAMENTO
# ============================================================

print("=" * 60)
print("PLANEJAMENTO DE REPOSICAO - NOVAVAREJO")
print("=" * 60)

print("\nCarregando previsoes...")

previsoes, data_referencia = carregar_previsoes()

print(
    "Data de referencia:",
    data_referencia.date()
)

print(
    "Periodo previsto:",
    previsoes["data_alvo"].min().date(),
    "ate",
    previsoes["data_alvo"].max().date()
)

print(f"Series: {previsoes['id'].nunique()}")
print(f"Previsoes: {len(previsoes)}")

print("\nCarregando historico conhecido...")

historico = carregar_historico(
    previsoes["id"].unique()
)

previsoes = previsoes.merge(
    historico,
    on="id",
    how="left",
    validate="many_to_one",
)

assert previsoes[
    ["media_historica", "desvio_historico"]
].notna().all().all()


# ============================================================
# 4. PLANEJAMENTO POR PRODUTO E LOJA
# ============================================================

print("\nCalculando planejamento de reposicao...")

registros_plano = []
registros_diarios = []

for identificador, grupo in previsoes.groupby(
    "id",
    sort=True
):
    grupo = grupo.sort_values("horizonte")

    primeiro = grupo.iloc[0]

    parametros = simular_parametros_estoque(
        identificador,
        primeiro["media_historica"],
        primeiro["desvio_historico"],
    )

    estoque_atual = parametros["estoque_atual"]

    pedido_em_transito = parametros[
        "pedido_em_transito"
    ]

    chegada_pedido = parametros[
        "dia_chegada_pedido"
    ]

    prazo_entrega = parametros[
        "prazo_entrega_dias"
    ]

    estoque_seguranca = parametros[
        "estoque_seguranca"
    ]

    demanda_7d = float(
        grupo[COLUNA_PREVISAO].sum()
    )

    demanda_prazo_entrega = float(
        grupo.loc[
            grupo["horizonte"] <= prazo_entrega,
            COLUNA_PREVISAO
        ].sum()
    )

    # Pedidos em transito que chegam dentro
    # do prazo de entrega considerado.
    transito_ate_prazo = (
        pedido_em_transito
        if (
            chegada_pedido is not None
            and chegada_pedido <= prazo_entrega
        )
        else 0
    )

    ponto_pedido = (
        demanda_prazo_entrega
        + estoque_seguranca
    )

    saldo_para_ponto_pedido = (
        estoque_atual
        + transito_ate_prazo
    )

    # Somente pedidos que chegam dentro dos
    # sete dias entram no calculo do horizonte.
    transito_7d = (
        pedido_em_transito
        if (
            chegada_pedido is not None
            and chegada_pedido <= HORIZONTE
        )
        else 0
    )

    quantidade_repor = max(
        0,
        math.ceil(
            demanda_7d
            + estoque_seguranca
            - estoque_atual
            - transito_7d
        )
    )

    media_prevista_diaria = (
        demanda_7d / HORIZONTE
    )

    cobertura_dias = (
        estoque_atual / media_prevista_diaria
        if media_prevista_diaria > 0
        else np.nan
    )

    # Simulacao diaria SEM considerar novas
    # compras recomendadas pelo sistema.
    saldo = float(estoque_atual)
    primeiro_dia_ruptura = None

    for linha in grupo.itertuples(index=False):
        horizonte = int(linha.horizonte)

        demanda = float(
            getattr(linha, COLUNA_PREVISAO)
        )

        recebimento = (
            pedido_em_transito
            if chegada_pedido == horizonte
            else 0
        )

        saldo_inicial_dia = saldo

        saldo += recebimento
        saldo -= demanda

        if (
            saldo < 0
            and primeiro_dia_ruptura is None
        ):
            primeiro_dia_ruptura = horizonte

        registros_diarios.append({
            "id": identificador,
            "item_id": linha.item_id,
            "store_id": linha.store_id,
            "data_alvo": linha.data_alvo,
            "horizonte": horizonte,
            "demanda_prevista": round(demanda, 4),
            "saldo_inicial_dia": round(
                saldo_inicial_dia, 4
            ),
            "recebimento_previsto": recebimento,
            "saldo_final_projetado": round(
                saldo, 4
            ),
            "risco_ruptura_dia": saldo < 0,
        })

    # Classificacao de prioridade.
    if (
        primeiro_dia_ruptura is not None
        and primeiro_dia_ruptura <= prazo_entrega
    ):
        prioridade = "CRITICA"

        acao = (
            "Avaliar transferencia urgente: "
            "risco antes da nova entrega"
        )

    elif primeiro_dia_ruptura is not None:
        prioridade = "ALTA"

        acao = (
            "Priorizar reposicao para "
            "evitar ruptura"
        )

    elif saldo_para_ponto_pedido <= ponto_pedido:
        prioridade = "MEDIA"

        acao = (
            "Programar reposicao conforme "
            "ponto de pedido"
        )

    else:
        prioridade = "BAIXA"

        acao = (
            "Monitorar estoque e demanda"
        )

    registros_plano.append({
        "id": identificador,
        "item_id": primeiro["item_id"],
        "store_id": primeiro["store_id"],
        "data_referencia": data_referencia,
        "modelo": NOME_MODELO,
        "demanda_prevista_7d": round(
            demanda_7d, 4
        ),
        "estoque_atual_simulado": estoque_atual,
        "pedido_em_transito_simulado": (
            pedido_em_transito
        ),
        "dia_chegada_pedido": chegada_pedido,
        "prazo_entrega_dias": prazo_entrega,
        "estoque_seguranca_simulado": (
            estoque_seguranca
        ),
        "ponto_pedido": round(
            ponto_pedido, 4
        ),
        "cobertura_dias": (
            round(cobertura_dias, 2)
            if np.isfinite(cobertura_dias)
            else np.nan
        ),
        "quantidade_reposicao_sugerida": (
            quantidade_repor
        ),
        "saldo_final_sem_nova_reposicao": round(
            saldo, 4
        ),
        "dia_primeira_ruptura": (
            primeiro_dia_ruptura
        ),
        "risco_ruptura": (
            primeiro_dia_ruptura is not None
        ),
        "prioridade": prioridade,
        "acao_recomendada": acao,
    })


# ============================================================
# 5. CONSOLIDACAO
# ============================================================

plano = pd.DataFrame(registros_plano)

diario = pd.DataFrame(registros_diarios)

ordem_prioridade = {
    "CRITICA": 0,
    "ALTA": 1,
    "MEDIA": 2,
    "BAIXA": 3,
}

plano["_ordem"] = plano["prioridade"].map(
    ordem_prioridade
)

plano = (
    plano.sort_values(
        [
            "_ordem",
            "quantidade_reposicao_sugerida",
        ],
        ascending=[True, False]
    )
    .drop(columns="_ordem")
    .reset_index(drop=True)
)

diario = diario.sort_values(
    ["store_id", "item_id", "data_alvo"]
).reset_index(drop=True)

assert len(plano) == QUANTIDADE_SERIES

assert len(diario) == (
    QUANTIDADE_SERIES * HORIZONTE
)

assert plano["id"].is_unique

assert not diario.duplicated(
    ["id", "data_alvo"]
).any()

assert (
    plano["quantidade_reposicao_sugerida"] >= 0
).all()

assert (
    plano["estoque_atual_simulado"] >= 0
).all()


# ============================================================
# 6. INDICADORES GERENCIAIS
# ============================================================

demanda_total = plano[
    "demanda_prevista_7d"
].sum()

estoque_total = plano[
    "estoque_atual_simulado"
].sum()

reposicao_total = plano[
    "quantidade_reposicao_sugerida"
].sum()

series_com_risco = int(
    plano["risco_ruptura"].sum()
)

criticas = int(
    (plano["prioridade"] == "CRITICA").sum()
)

altas = int(
    (plano["prioridade"] == "ALTA").sum()
)

print("\n" + "=" * 60)
print("RESUMO GERENCIAL")
print("=" * 60)

print(
    f"Demanda prevista em 7 dias: "
    f"{demanda_total:,.2f} unidades"
)

print(
    f"Estoque atual simulado: "
    f"{estoque_total:,} unidades"
)

print(
    f"Reposicao sugerida: "
    f"{reposicao_total:,} unidades"
)

print(
    f"Series com risco de ruptura: "
    f"{series_com_risco}"
)

print(f"Prioridade critica: {criticas}")
print(f"Prioridade alta: {altas}")


# ============================================================
# 7. PRIORIDADES POR PRODUTO
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 PRIORIDADES DE REPOSICAO")
print("=" * 60)

colunas_exibicao = [
    "item_id",
    "store_id",
    "demanda_prevista_7d",
    "estoque_atual_simulado",
    "quantidade_reposicao_sugerida",
    "prioridade",
]

print(
    plano[colunas_exibicao]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 8. RESULTADOS POR LOJA
# ============================================================

por_loja = (
    plano.groupby("store_id")
    .agg(
        produtos=("id", "count"),
        demanda_prevista=(
            "demanda_prevista_7d",
            "sum"
        ),
        estoque_simulado=(
            "estoque_atual_simulado",
            "sum"
        ),
        reposicao_sugerida=(
            "quantidade_reposicao_sugerida",
            "sum"
        ),
        produtos_em_risco=(
            "risco_ruptura",
            "sum"
        ),
    )
    .reset_index()
    .sort_values(
        "reposicao_sugerida",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("PLANEJAMENTO POR LOJA")
print("=" * 60)

print(
    por_loja.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.2f}"
    )
)


# ============================================================
# 9. EXPORTACAO
# ============================================================

plano.to_csv(
    ARQUIVO_PLANO,
    index=False
)

diario.to_csv(
    ARQUIVO_DIARIO,
    index=False
)

def tabela_markdown(df, colunas):
    cabecalho = "| " + " | ".join(colunas) + " |"

    separador = (
        "| " + " | ".join(
            ["---"] * len(colunas)
        ) + " |"
    )

    linhas = []

    for valores in df[colunas].itertuples(
        index=False,
        name=None
    ):
        linhas.append(
            "| "
            + " | ".join(str(v) for v in valores)
            + " |"
        )

    return "\n".join(
        [cabecalho, separador] + linhas
    )


top_10 = plano[
    [
        "item_id",
        "store_id",
        "prioridade",
        "quantidade_reposicao_sugerida",
    ]
].head(10)

relatorio = f"""# Planejamento de estoque - NovaVarejo

## Contexto

Demonstracao de apoio a decisao de reposicao
utilizando previsoes de demanda e estoque simulado.

**Data de referencia:** {data_referencia.date()}

**Periodo previsto:** {
    previsoes["data_alvo"].min().date()
} a {
    previsoes["data_alvo"].max().date()
}

**Modelo:** {NOME_MODELO}

## Indicadores

- Series avaliadas: {len(plano)}
- Demanda prevista: {demanda_total:.2f} unidades
- Estoque simulado: {estoque_total} unidades
- Reposicao sugerida: {reposicao_total} unidades
- Series com risco de ruptura: {series_com_risco}
- Prioridade critica: {criticas}
- Prioridade alta: {altas}

## Prioridades de reposicao

{tabela_markdown(
    top_10,
    [
        "item_id",
        "store_id",
        "prioridade",
        "quantidade_reposicao_sugerida",
    ]
)}

## Premissas

- As previsoes sao provenientes do modelo treinado.
- Os saldos de estoque sao simulados.
- Os pedidos em transito sao simulados.
- Os prazos de entrega sao simulados.
- O estoque de seguranca usa uma heuristica
  baseada na variabilidade historica.
- Os pedidos em transito sao recebidos no inicio
  do dia indicado.
- A simulacao diaria nao considera novas compras
  recomendadas pelo sistema.
- A quantidade sugerida considera demanda de
  sete dias, estoque de seguranca, saldo inicial
  e pedidos em transito no horizonte.
- Uma nova compra pode chegar tarde demais
  para evitar uma ruptura antecipada.
- As quantidades sugeridas nao consideram
  lotes minimos, custos ou restricoes de fornecedor.

## Limitacoes

Esta e uma simulacao historica para portfolio.

Nao representa estoque real ou uma recomendacao
operacional pronta para execucao.

A RF aprimorada foi a candidata selecionada
no desenvolvimento, mas nao superou a RF original
no teste independente.

A demonstracao nao utiliza vendas reais futuras
para calcular o planejamento.

## Proxima etapa

Construir dashboard gerencial com Streamlit.
"""

ARQUIVO_RELATORIO.write_text(
    relatorio,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("ARQUIVOS GERADOS")
print("=" * 60)

print(ARQUIVO_PLANO)
print(ARQUIVO_DIARIO)
print(ARQUIVO_RELATORIO)

print("\nPlanejamento concluido com sucesso.")
