
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_PLANO = (
    BASE_DIR
    / "data"
    / "processed"
    / "planejamento_reposicao.csv"
)

ARQUIVO_DIARIO = (
    BASE_DIR
    / "data"
    / "processed"
    / "planejamento_demanda_diaria.csv"
)

st.set_page_config(
    page_title="NovaVarejo | Gestao de Estoque",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. FUNCOES AUXILIARES
# ============================================================

def formatar_numero(valor, casas=0):
    texto = f"{valor:,.{casas}f}"

    return (
        texto.replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def carregar_dados():
    arquivos = [
        ARQUIVO_PLANO,
        ARQUIVO_DIARIO,
    ]

    for arquivo in arquivos:
        if not arquivo.is_file():
            st.error(
                "Arquivo necessario nao encontrado: "
                f"{arquivo.name}"
            )

            st.info(
                "Execute primeiro: "
                "python src/planejamento_reposicao.py"
            )

            st.stop()

    plano = pd.read_csv(
        ARQUIVO_PLANO,
        parse_dates=["data_referencia"],
    )

    diario = pd.read_csv(
        ARQUIVO_DIARIO,
        parse_dates=["data_alvo"],
    )

    colunas_plano = [
        "id",
        "item_id",
        "store_id",
        "data_referencia",
        "modelo",
        "demanda_prevista_7d",
        "estoque_atual_simulado",
        "pedido_em_transito_simulado",
        "prazo_entrega_dias",
        "estoque_seguranca_simulado",
        "quantidade_reposicao_sugerida",
        "cobertura_dias",
        "risco_ruptura",
        "prioridade",
        "acao_recomendada",
    ]

    colunas_diario = [
        "id",
        "item_id",
        "store_id",
        "data_alvo",
        "horizonte",
        "demanda_prevista",
        "saldo_final_projetado",
        "risco_ruptura_dia",
    ]

    faltantes_plano = [
        coluna
        for coluna in colunas_plano
        if coluna not in plano.columns
    ]

    faltantes_diario = [
        coluna
        for coluna in colunas_diario
        if coluna not in diario.columns
    ]

    if faltantes_plano or faltantes_diario:
        st.error(
            "Os arquivos CSV nao possuem "
            "todas as colunas esperadas."
        )

        st.write(
            "Colunas ausentes no plano:",
            faltantes_plano
        )

        st.write(
            "Colunas ausentes no diario:",
            faltantes_diario
        )

        st.stop()

    if plano.empty or diario.empty:
        st.error("Os arquivos de dados estao vazios.")
        st.stop()

    if plano["id"].duplicated().any():
        st.error(
            "Existem series duplicadas no plano."
        )
        st.stop()

    if diario.duplicated(
        ["id", "data_alvo"]
    ).any():
        st.error(
            "Existem previsoes diarias duplicadas."
        )
        st.stop()

    if not set(diario["id"]) == set(plano["id"]):
        st.error(
            "As series dos dois arquivos "
            "nao correspondem."
        )
        st.stop()

    # Confere se a soma da previsao diaria
    # corresponde ao plano de reposicao.
    soma_diaria = (
        diario.groupby("id")["demanda_prevista"]
        .sum()
    )

    demanda_plano = plano.set_index(
        "id"
    )["demanda_prevista_7d"]

    if not np.allclose(
        soma_diaria.sort_index(),
        demanda_plano.sort_index(),
        atol=0.01,
        rtol=0,
    ):
        st.error(
            "Inconsistencia entre as previsoes "
            "diarias e o plano de reposicao."
        )
        st.stop()

    return plano, diario


# ============================================================
# 3. CARREGAR DADOS
# ============================================================

plano, diario = carregar_dados()

data_referencia = (
    plano["data_referencia"].min()
)

data_inicio = diario["data_alvo"].min()
data_fim = diario["data_alvo"].max()

modelo = plano["modelo"].iloc[0]


# ============================================================
# 4. CABECALHO
# ============================================================

st.title("📦 NovaVarejo | Gestao de Estoque")

st.markdown(
    "### Planejamento de reposicao com "
    "previsao de demanda"
)

st.warning(
    "**Demonstracao historica:** "
    "as previsoes referem-se a "
    f"{data_inicio:%d/%m/%Y} ate "
    f"{data_fim:%d/%m/%Y}. "
    "Estoque, pedidos em transito e "
    "prazos de entrega sao simulados. "
    "Nao utilizar para compras reais."
)

st.caption(
    f"Data de referencia: "
    f"{data_referencia:%d/%m/%Y} | "
    f"Modelo: {modelo}"
)


# ============================================================
# 5. FILTROS
# ============================================================

st.sidebar.header("Filtros gerenciais")

lojas = sorted(
    plano["store_id"].unique().tolist()
)

lojas_selecionadas = st.sidebar.multiselect(
    "Lojas",
    options=lojas,
    default=lojas,
)

prioridades_disponiveis = [
    prioridade
    for prioridade in [
        "CRITICA",
        "ALTA",
        "MEDIA",
        "BAIXA",
    ]
    if prioridade in plano["prioridade"].unique()
]

prioridades_selecionadas = (
    st.sidebar.multiselect(
        "Prioridades",
        options=prioridades_disponiveis,
        default=prioridades_disponiveis,
    )
)

produtos = sorted(
    plano["item_id"].unique().tolist()
)

produto_selecionado = st.sidebar.selectbox(
    "Produto",
    options=["Todos os produtos"] + produtos,
)

filtrado = plano.loc[
    plano["store_id"].isin(
        lojas_selecionadas
    )
    & plano["prioridade"].isin(
        prioridades_selecionadas
    )
].copy()

if produto_selecionado != "Todos os produtos":
    filtrado = filtrado.loc[
        filtrado["item_id"] == produto_selecionado
    ].copy()

if filtrado.empty:
    st.info(
        "Nenhuma serie encontrada para os "
        "filtros selecionados."
    )
    st.stop()

diario_filtrado = diario.loc[
    diario["id"].isin(filtrado["id"])
].copy()

st.sidebar.divider()

st.sidebar.caption(
    f"{len(filtrado)} combinacoes "
    "de produto e loja selecionadas."
)


# ============================================================
# 6. INDICADORES PRINCIPAIS
# ============================================================

demanda_total = filtrado[
    "demanda_prevista_7d"
].sum()

estoque_total = filtrado[
    "estoque_atual_simulado"
].sum()

reposicao_total = filtrado[
    "quantidade_reposicao_sugerida"
].sum()

series_risco = int(
    filtrado["risco_ruptura"].sum()
)

series_criticas = int(
    (filtrado["prioridade"] == "CRITICA").sum()
)

st.subheader("Resumo gerencial")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Demanda prevista",
        formatar_numero(demanda_total, 2),
        help="Unidades previstas para sete dias.",
    )

with col2:
    st.metric(
        "Estoque simulado",
        formatar_numero(estoque_total),
        help="Saldo inicial simulado.",
    )

with col3:
    st.metric(
        "Reposicao sugerida",
        formatar_numero(reposicao_total),
        help=(
            "Quantidade sugerida considerando "
            "demanda, seguranca e pedidos em transito."
        ),
    )

with col4:
    st.metric(
        "Series em risco",
        f"{series_risco} de {len(filtrado)}",
        help=(
            "Combinacoes produto-loja com "
            "saldo projetado negativo."
        ),
    )

st.caption(
    f"Prioridade critica: {series_criticas} "
    "combinacoes produto-loja."
)


# ============================================================
# 7. GRAFICOS DE DEMANDA E SALDO
# ============================================================

st.divider()
st.subheader("Projecao para os proximos 7 dias")

demanda_diaria = (
    diario_filtrado.groupby("data_alvo")
    .agg(
        demanda_prevista=(
            "demanda_prevista",
            "sum"
        ),
        saldo_projetado=(
            "saldo_final_projetado",
            "sum"
        ),
    )
    .sort_index()
)

esquerda, direita = st.columns(2)

with esquerda:
    st.markdown("**Demanda prevista por dia**")

    st.bar_chart(
        demanda_diaria[["demanda_prevista"]],
        x_label="Data",
        y_label="Unidades",
    )

with direita:
    st.markdown(
        "**Saldo projetado sem novas compras**"
    )

    st.line_chart(
        demanda_diaria[["saldo_projetado"]],
        x_label="Data",
        y_label="Unidades",
    )

st.caption(
    "O saldo projetado pode ficar negativo. "
    "Nesse caso, representa demanda acumulada "
    "que excede o estoque disponível, e não "
    "estoque físico negativo."
)


# ============================================================
# 8. PRIORIDADES POR LOJA
# ============================================================

st.divider()
st.subheader("Necessidade de reposicao por loja")

lojas_resumo = (
    filtrado.groupby("store_id")
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
        series_em_risco=(
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

st.bar_chart(
    lojas_resumo.set_index(
        "store_id"
    )[["reposicao_sugerida"]],
    x_label="Loja",
    y_label="Unidades sugeridas",
)

st.dataframe(
    lojas_resumo.rename(
        columns={
            "store_id": "Loja",
            "produtos": "Series",
            "demanda_prevista": "Demanda prevista",
            "estoque_simulado": "Estoque simulado",
            "reposicao_sugerida": "Reposicao sugerida",
            "series_em_risco": "Series em risco",
        }
    ).round(2),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 9. PLANO DETALHADO DE REPOSICAO
# ============================================================

st.divider()
st.subheader("Plano detalhado de reposicao")

ordem_prioridade = {
    "CRITICA": 0,
    "ALTA": 1,
    "MEDIA": 2,
    "BAIXA": 3,
}

detalhes = filtrado.copy()

detalhes["_ordem"] = (
    detalhes["prioridade"]
    .map(ordem_prioridade)
)

detalhes = detalhes.sort_values(
    [
        "_ordem",
        "quantidade_reposicao_sugerida",
    ],
    ascending=[True, False],
)

colunas_tabela = [
    "item_id",
    "store_id",
    "prioridade",
    "demanda_prevista_7d",
    "estoque_atual_simulado",
    "pedido_em_transito_simulado",
    "prazo_entrega_dias",
    "estoque_seguranca_simulado",
    "cobertura_dias",
    "quantidade_reposicao_sugerida",
    "acao_recomendada",
]

tabela = detalhes[colunas_tabela].rename(
    columns={
        "item_id": "Produto",
        "store_id": "Loja",
        "prioridade": "Prioridade",
        "demanda_prevista_7d": "Demanda 7 dias",
        "estoque_atual_simulado": "Estoque atual",
        "pedido_em_transito_simulado": "Em transito",
        "prazo_entrega_dias": "Prazo entrega",
        "estoque_seguranca_simulado": "Estoque seguranca",
        "cobertura_dias": "Cobertura (dias)",
        "quantidade_reposicao_sugerida": "Repor (unid.)",
        "acao_recomendada": "Acao recomendada",
    }
)

st.dataframe(
    tabela.round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Prioridade CRITICA: ruptura projetada "
    "antes ou ate o prazo de entrega. "
    "Prioridade ALTA: ruptura posterior "
    "ao prazo de entrega, dentro do horizonte."
)


# ============================================================
# 10. EXPORTACAO
# ============================================================

st.divider()
st.subheader("Exportar plano de reposicao")

csv_exportacao = detalhes.drop(
    columns=["_ordem"]
).to_csv(
    index=False,
    sep=";",
    decimal=",",
).encode("utf-8-sig")

st.download_button(
    label="Baixar plano de reposicao (CSV)",
    data=csv_exportacao,
    file_name="novavarejo_plano_reposicao.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# 11. PREMISSAS E LIMITACOES
# ============================================================

with st.expander(
    "Metodologia, premissas e limitacoes"
):
    st.markdown(
        """
        ### Origem das previsoes

        As previsoes foram geradas pela Random Forest
        aprimorada, selecionada durante o desenvolvimento.

        No teste independente, a Random Forest original
        apresentou menor WAPE. Portanto, a superioridade
        da versao aprimorada nao foi confirmada.

        ### Dados operacionais

        - Estoques sao simulados.
        - Pedidos em transito sao simulados.
        - Prazos de entrega sao simulados.
        - Estoque de seguranca utiliza uma heuristica.
        - O horizonte de planejamento e de sete dias.

        ### Calculo de reposicao

        Reposicao sugerida = max(
            0,
            demanda prevista
            + estoque de seguranca
            - estoque atual
            - pedidos em transito no horizonte
        )

        ### Restricoes

        - Nao considera lotes minimos de fornecedores.
        - Nao considera custos de compra ou transporte.
        - Nao incorpora restricoes de capacidade.
        - Uma compra pode chegar tarde demais
          para evitar determinada ruptura.
        - Os resultados nao sao recomendacoes
          operacionais para uma empresa real.

        ### Interpretacao

        Este dashboard e uma demonstracao historica
        de apoio a decisao, construida para portfolio.
        """
    )

st.divider()

st.caption(
    "NovaVarejo Brasil | Projeto educacional "
    "de Ciencia de Dados e apoio a decisao"
)
