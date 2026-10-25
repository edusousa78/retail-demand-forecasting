from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PASTA_LOCAL = BASE_DIR / "data" / "processed"
PASTA_DEMO = BASE_DIR / "data" / "demo"
NOME_PLANO = "planejamento_reposicao.csv"
NOME_DIARIO = "planejamento_demanda_diaria.csv"

# Em desenvolvimento, utiliza os arquivos processados.
# No Streamlit Cloud, utiliza os arquivos versionados de demo.
if (
    (PASTA_LOCAL / NOME_PLANO).is_file()
    and (PASTA_LOCAL / NOME_DIARIO).is_file()
):
    PASTA_DADOS = PASTA_LOCAL
    MODO_DADOS = "Resultados locais"
else:
    PASTA_DADOS = PASTA_DEMO
    MODO_DADOS = "Demonstracao publicada"

ARQUIVO_PLANO = PASTA_DADOS / NOME_PLANO
ARQUIVO_DIARIO = PASTA_DADOS / NOME_DIARIO

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
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


@st.cache_data(show_spinner="Carregando planejamento...")
def ler_csv(caminho, coluna_data):
    return pd.read_csv(caminho, parse_dates=[coluna_data])


def carregar_dados():
    for arquivo in (ARQUIVO_PLANO, ARQUIVO_DIARIO):
        if not arquivo.is_file():
            st.error(f"Arquivo necessario nao encontrado: {arquivo.name}")
            st.info(
                "Gere os resultados locais com "
                "`python src/planejamento_reposicao.py` ou "
                "adicione os dois CSVs em `data/demo/` para o deploy."
            )
            st.stop()

    try:
        plano = ler_csv(ARQUIVO_PLANO, "data_referencia")
        diario = ler_csv(ARQUIVO_DIARIO, "data_alvo")
    except (OSError, ValueError, pd.errors.ParserError) as erro:
        st.error(f"Nao foi possivel ler os dados: {erro}")
        st.stop()

    colunas_plano = {
        "id", "item_id", "store_id", "data_referencia", "modelo",
        "demanda_prevista_7d", "estoque_atual_simulado",
        "pedido_em_transito_simulado", "prazo_entrega_dias",
        "estoque_seguranca_simulado", "quantidade_reposicao_sugerida",
        "cobertura_dias", "risco_ruptura", "prioridade",
        "acao_recomendada",
    }
    colunas_diario = {
        "id", "item_id", "store_id", "data_alvo", "horizonte",
        "demanda_prevista", "saldo_final_projetado", "risco_ruptura_dia",
    }

    faltantes_plano = sorted(colunas_plano - set(plano.columns))
    faltantes_diario = sorted(colunas_diario - set(diario.columns))
    if faltantes_plano or faltantes_diario:
        st.error("Os arquivos CSV nao possuem todas as colunas esperadas.")
        st.write("Colunas ausentes no plano:", faltantes_plano)
        st.write("Colunas ausentes no diario:", faltantes_diario)
        st.stop()

    if plano.empty or diario.empty:
        st.error("Os arquivos de dados estao vazios.")
        st.stop()
    if plano["id"].duplicated().any():
        st.error("Existem series duplicadas no plano.")
        st.stop()
    if diario.duplicated(["id", "data_alvo"]).any():
        st.error("Existem previsoes diarias duplicadas.")
        st.stop()
    if set(diario["id"]) != set(plano["id"]):
        st.error("As series dos dois arquivos nao correspondem.")
        st.stop()

    # O total diario por serie deve coincidir com o plano de 7 dias.
    soma_diaria = diario.groupby("id")["demanda_prevista"].sum().sort_index()
    demanda_plano = (
        plano.set_index("id")["demanda_prevista_7d"].sort_index()
    )
    if not np.allclose(soma_diaria, demanda_plano, atol=0.01, rtol=0):
        st.error("Inconsistencia entre previsoes diarias e plano de reposicao.")
        st.stop()

    return plano, diario


# ============================================================
# 3. CARREGAMENTO E CABECALHO
# ============================================================

plano, diario = carregar_dados()
data_referencia = plano["data_referencia"].min()
data_inicio = diario["data_alvo"].min()
data_fim = diario["data_alvo"].max()
modelo = plano["modelo"].iloc[0]

st.title("📦 NovaVarejo | Gestao de Estoque")
st.markdown("### Planejamento de reposicao com previsao de demanda")
st.warning(
    "**Demonstracao historica:** previsoes de "
    f"{data_inicio:%d/%m/%Y} a {data_fim:%d/%m/%Y}. "
    "Estoques, pedidos em transito e prazos de entrega sao simulados. "
    "Os resultados nao devem ser usados para compras reais."
)
st.caption(
    f"Dados: {MODO_DADOS} | Referencia: {data_referencia:%d/%m/%Y} "
    f"| Modelo: {modelo}"
)


# ============================================================
# 4. FILTROS GERENCIAIS
# ============================================================

st.sidebar.header("Filtros gerenciais")
lojas = sorted(plano["store_id"].unique().tolist())
lojas_selecionadas = st.sidebar.multiselect(
    "Lojas", options=lojas, default=lojas
)
prioridades_disponiveis = [
    prioridade
    for prioridade in ("CRITICA", "ALTA", "MEDIA", "BAIXA")
    if prioridade in plano["prioridade"].unique()
]
prioridades_selecionadas = st.sidebar.multiselect(
    "Prioridades",
    options=prioridades_disponiveis,
    default=prioridades_disponiveis,
)
produtos = sorted(plano["item_id"].unique().tolist())
produto_selecionado = st.sidebar.selectbox(
    "Produto", options=["Todos os produtos"] + produtos
)

filtrado = plano.loc[
    plano["store_id"].isin(lojas_selecionadas)
    & plano["prioridade"].isin(prioridades_selecionadas)
].copy()
if produto_selecionado != "Todos os produtos":
    filtrado = filtrado.loc[
        filtrado["item_id"] == produto_selecionado
    ].copy()
if filtrado.empty:
    st.info("Nenhuma serie encontrada para os filtros selecionados.")
    st.stop()

diario_filtrado = diario.loc[diario["id"].isin(filtrado["id"])].copy()
st.sidebar.divider()
st.sidebar.caption(
    f"{len(filtrado)} combinacoes de produto e loja selecionadas."
)
st.sidebar.caption(f"Fonte: {MODO_DADOS}")


# ============================================================
# 5. INDICADORES PRINCIPAIS
# ============================================================

st.subheader("Resumo gerencial")
demanda_total = filtrado["demanda_prevista_7d"].sum()
estoque_total = filtrado["estoque_atual_simulado"].sum()
reposicao_total = filtrado["quantidade_reposicao_sugerida"].sum()
series_risco = int(filtrado["risco_ruptura"].sum())
series_criticas = int((filtrado["prioridade"] == "CRITICA").sum())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Demanda prevista", formatar_numero(demanda_total, 2),
        help="Unidades previstas para sete dias.",
    )
with col2:
    st.metric(
        "Estoque simulado", formatar_numero(estoque_total),
        help="Saldo inicial simulado.",
    )
with col3:
    st.metric(
        "Reposicao sugerida", formatar_numero(reposicao_total),
        help="Demanda, estoque de seguranca e pedidos em transito.",
    )
with col4:
    st.metric(
        "Series em risco", f"{series_risco} de {len(filtrado)}",
        help="Combinacoes produto-loja com risco de ruptura.",
    )
st.caption(f"Prioridade critica: {series_criticas} combinacoes produto-loja.")


# ============================================================
# 6. GRAFICOS DE DEMANDA E ESTOQUE
# ============================================================

st.divider()
st.subheader("Projecao para os proximos 7 dias")
demanda_diaria = (
    diario_filtrado.groupby("data_alvo")
    .agg(
        demanda_prevista=("demanda_prevista", "sum"),
        saldo_projetado=("saldo_final_projetado", "sum"),
    )
    .sort_index()
)
esquerda, direita = st.columns(2)
with esquerda:
    st.markdown("**Demanda prevista por dia**")
    st.bar_chart(
        demanda_diaria[["demanda_prevista"]],
        x_label="Data", y_label="Unidades",
    )
with direita:
    st.markdown("**Saldo projetado sem novas compras**")
    st.line_chart(
        demanda_diaria[["saldo_projetado"]],
        x_label="Data", y_label="Unidades",
    )
st.caption(
    "Um saldo projetado negativo representa demanda nao atendida "
    "acumulada, e nao estoque fisico negativo."
)


# ============================================================
# 7. NECESSIDADE DE REPOSICAO POR LOJA
# ============================================================

st.divider()
st.subheader("Necessidade de reposicao por loja")
lojas_resumo = (
    filtrado.groupby("store_id")
    .agg(
        produtos=("id", "count"),
        demanda_prevista=("demanda_prevista_7d", "sum"),
        estoque_simulado=("estoque_atual_simulado", "sum"),
        reposicao_sugerida=("quantidade_reposicao_sugerida", "sum"),
        series_em_risco=("risco_ruptura", "sum"),
    )
    .reset_index()
    .sort_values("reposicao_sugerida", ascending=False)
)
st.bar_chart(
    lojas_resumo.set_index("store_id")[["reposicao_sugerida"]],
    x_label="Loja", y_label="Unidades sugeridas",
)
st.dataframe(
    lojas_resumo.rename(columns={
        "store_id": "Loja",
        "produtos": "Series",
        "demanda_prevista": "Demanda prevista",
        "estoque_simulado": "Estoque simulado",
        "reposicao_sugerida": "Reposicao sugerida",
        "series_em_risco": "Series em risco",
    }).round(2),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 8. PLANO DETALHADO DE REPOSICAO
# ============================================================

st.divider()
st.subheader("Plano detalhado de reposicao")
ordem_prioridade = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAIXA": 3}
detalhes = filtrado.copy()
detalhes["_ordem"] = detalhes["prioridade"].map(ordem_prioridade)
detalhes = detalhes.sort_values(
    ["_ordem", "quantidade_reposicao_sugerida"],
    ascending=[True, False],
)
colunas_tabela = [
    "item_id", "store_id", "prioridade", "demanda_prevista_7d",
    "estoque_atual_simulado", "pedido_em_transito_simulado",
    "prazo_entrega_dias", "estoque_seguranca_simulado",
    "cobertura_dias", "quantidade_reposicao_sugerida",
    "acao_recomendada",
]
tabela = detalhes[colunas_tabela].rename(columns={
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
})
st.dataframe(tabela.round(2), hide_index=True, use_container_width=True)
st.caption(
    "CRITICA: ruptura projetada antes ou ate o prazo de entrega. "
    "ALTA: ruptura posterior ao prazo, dentro dos sete dias."
)


# ============================================================
# 9. EXPORTACAO
# ============================================================

st.divider()
st.subheader("Exportar plano de reposicao")
csv_exportacao = (
    detalhes.drop(columns=["_ordem"])
    .to_csv(index=False, sep=";", decimal=",")
    .encode("utf-8-sig")
)
st.download_button(
    label="Baixar plano de reposicao (CSV)",
    data=csv_exportacao,
    file_name="novavarejo_plano_reposicao.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# 10. METODOLOGIA E LIMITACOES
# ============================================================

with st.expander("Metodologia, premissas e limitacoes"):
    st.markdown(
        """
        ### Origem das previsoes

        As previsoes foram geradas pela Random Forest aprimorada,
        selecionada durante o desenvolvimento. No teste independente,
        a Random Forest original apresentou menor WAPE; portanto,
        a superioridade da versao aprimorada nao foi confirmada.

        ### Dados operacionais

        - Estoques, pedidos em transito e prazos sao simulados.
        - Estoque de seguranca utiliza uma heuristica.
        - Horizonte de planejamento: sete dias.
        - Os CSVs de demonstracao contem resultados historicos
          derivados do M5, combinados com dados operacionais simulados.

        ### Calculo de reposicao

        Quantidade sugerida = max(0, teto(demanda prevista +
        estoque de seguranca - estoque atual - pedidos em transito
        considerados no horizonte)).

        ### Restricoes

        - Nao considera lotes minimos, custos ou capacidade.
        - Uma compra pode chegar tarde demais para evitar ruptura.
        - Nao ha integracao em tempo real com sistemas empresariais.
        - Os resultados nao sao recomendacoes de compras reais.

        ### Interpretacao

        Demonstracao historica de apoio a decisao, criada para portfolio.
        """
    )

st.divider()
st.caption(
    "NovaVarejo Brasil | Projeto educacional de Ciencia de Dados "
    "e apoio a decisao"
)
