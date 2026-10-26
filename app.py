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
    MODO_DADOS = "Demonstração publicada"

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


ESTADOS = {"CA": "Califórnia", "TX": "Texas", "WI": "Wisconsin"}
CATEGORIAS = {
    "FOODS": "Alimentos",
    "HOUSEHOLD": "Casa e utilidades",
    "HOBBIES": "Lazer e hobbies",
}
PRIORIDADES = {
    "CRITICA": "Crítica",
    "ALTA": "Alta",
    "MEDIA": "Média",
    "BAIXA": "Baixa",
}


def nome_loja(codigo):
    """Apresenta um nome amigável sem alterar o código M5."""
    partes = str(codigo).split("_")
    if len(partes) == 2 and partes[0] in ESTADOS:
        return f"Loja {ESTADOS[partes[0]]} {partes[1].zfill(2)}"
    return f"Loja {codigo}"


def nome_produto(codigo):
    """O M5 não traz descrições comerciais dos itens."""
    partes = str(codigo).split("_")
    if len(partes) == 3 and partes[0] in CATEGORIAS:
        categoria, grupo, numero = partes
        return f"{CATEGORIAS[categoria]} · Grupo {grupo} · Produto {numero}"
    return f"Produto {codigo}"


def nome_prioridade(codigo):
    return PRIORIDADES.get(str(codigo), str(codigo).capitalize())


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

st.title("📦 NovaVarejo | Gestão de Estoque")
st.markdown("### Planejamento de reposição com previsão de demanda")
st.warning(
    "**Demonstração histórica:** previsões de "
    f"{data_inicio:%d/%m/%Y} a {data_fim:%d/%m/%Y}. "
    "Estoques, pedidos em trânsito e prazos de entrega são simulados. "
    "Os resultados não devem ser usados para compras reais."
)
st.caption(
    f"Dados: {MODO_DADOS} | Referência: {data_referencia:%d/%m/%Y} "
    f"| Modelo: {modelo}"
)


# ============================================================
# 4. FILTROS GERENCIAIS
# ============================================================

st.sidebar.header("Filtros gerenciais")
st.sidebar.caption("Selecione lojas, nível de urgência e produtos.")
lojas = sorted(plano["store_id"].unique().tolist())
lojas_selecionadas = st.sidebar.multiselect(
    "Lojas", options=lojas, default=lojas, format_func=nome_loja,
    help="Os nomes exibidos representam as lojas identificadas no conjunto M5.",
)
prioridades_disponiveis = [
    prioridade
    for prioridade in ("CRITICA", "ALTA", "MEDIA", "BAIXA")
    if prioridade in plano["prioridade"].unique()
]
prioridades_selecionadas = st.sidebar.multiselect(
    "Urgência de reposição",
    options=prioridades_disponiveis,
    default=prioridades_disponiveis,
    format_func=nome_prioridade,
)
produtos = sorted(plano["item_id"].unique().tolist())
produto_selecionado = st.sidebar.selectbox(
    "Produto", options=[None] + produtos,
    format_func=lambda codigo: (
        "Todos os produtos" if codigo is None else nome_produto(codigo)
    ),
    help="Nomes descritivos gerados a partir dos códigos M5; não são nomes comerciais.",
)

filtrado = plano.loc[
    plano["store_id"].isin(lojas_selecionadas)
    & plano["prioridade"].isin(prioridades_selecionadas)
].copy()
if produto_selecionado is not None:
    filtrado = filtrado.loc[
        filtrado["item_id"] == produto_selecionado
    ].copy()
if filtrado.empty:
    st.info("Nenhum produto encontrado para os filtros selecionados.")
    st.stop()

diario_filtrado = diario.loc[diario["id"].isin(filtrado["id"])].copy()
st.sidebar.divider()
st.sidebar.caption(
    f"{len(filtrado)} combinações de produto e loja selecionadas."
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
        "Reposição sugerida", formatar_numero(reposicao_total),
        help="Demanda, estoque de segurança e pedidos em trânsito.",
    )
with col4:
    st.metric(
        "Produtos/lojas em risco", f"{series_risco} de {len(filtrado)}",
        help="Combinações de produto e loja com risco de ruptura.",
    )
st.caption(f"Prioridade crítica: {series_criticas} combinações produto-loja.")


# ============================================================
# 6. GRAFICOS DE DEMANDA E ESTOQUE
# ============================================================

st.divider()
st.subheader("Projeção para os próximos 7 dias")
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
    "Um saldo projetado negativo representa demanda não atendida "
    "acumulada, e não estoque físico negativo."
)


# ============================================================
# 7. NECESSIDADE DE REPOSIÇÃO POR LOJA
# ============================================================

st.divider()
st.subheader("Necessidade de reposição por loja")
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
lojas_resumo["Loja"] = lojas_resumo["store_id"].map(nome_loja)
st.bar_chart(
    lojas_resumo.set_index("Loja")[["reposicao_sugerida"]].rename(
        columns={"reposicao_sugerida": "Unidades para reposição"}
    ),
    x_label="Loja", y_label="Unidades sugeridas",
)
st.dataframe(
    lojas_resumo.rename(columns={
        "produtos": "Produtos/lojas",
        "demanda_prevista": "Demanda prevista",
        "estoque_simulado": "Estoque simulado",
        "reposicao_sugerida": "Reposição sugerida",
        "series_em_risco": "Produtos/lojas em risco",
    })[
        ["Loja", "Produtos/lojas", "Demanda prevista", "Estoque simulado",
         "Reposição sugerida", "Produtos/lojas em risco"]
    ].round(2),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 8. PLANO DETALHADO DE REPOSIÇÃO
# ============================================================

st.divider()
st.subheader("Plano detalhado de reposição")
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
tabela = detalhes[colunas_tabela].copy()
tabela["item_id"] = tabela["item_id"].map(nome_produto)
tabela["store_id"] = tabela["store_id"].map(nome_loja)
tabela["prioridade"] = tabela["prioridade"].map(nome_prioridade)
tabela = tabela.rename(columns={
    "item_id": "Produto",
    "store_id": "Loja",
    "prioridade": "Urgência",
    "demanda_prevista_7d": "Demanda em 7 dias",
    "estoque_atual_simulado": "Estoque disponível",
    "pedido_em_transito_simulado": "Pedidos a receber",
    "prazo_entrega_dias": "Prazo de entrega (dias)",
    "estoque_seguranca_simulado": "Estoque de segurança",
    "cobertura_dias": "Cobertura (dias)",
    "quantidade_reposicao_sugerida": "Repor (unidades)",
    "acao_recomendada": "Ação recomendada",
})
st.dataframe(tabela.round(2), hide_index=True, use_container_width=True)
st.caption(
    "**Crítica:** ruptura projetada antes ou até o prazo de entrega. "
    "**Alta:** ruptura posterior ao prazo, dentro dos sete dias."
)
st.caption(
    "Os nomes de produtos são rótulos descritivos dos códigos M5, "
    "não nomes comerciais cadastrados."
)


# ============================================================
# 9. EXPORTAÇÃO
# ============================================================

st.divider()
st.subheader("Exportar plano de reposição")

# Exportação para o gestor: rótulos legíveis, mantendo IDs para auditoria.
exportacao = detalhes.drop(columns=["_ordem"]).copy()
exportacao.insert(0, "nome_produto", exportacao["item_id"].map(nome_produto))
exportacao.insert(1, "nome_loja", exportacao["store_id"].map(nome_loja))
exportacao.insert(2, "urgencia", exportacao["prioridade"].map(nome_prioridade))
csv_exportacao = exportacao.to_csv(index=False, sep=";", decimal=",").encode(
    "utf-8-sig"
)
st.download_button(
    label="Baixar plano de reposição (CSV)",
    data=csv_exportacao,
    file_name="novavarejo_plano_reposicao.csv",
    mime="text/csv",
    use_container_width=True,
)
st.caption("O CSV inclui nomes amigáveis e códigos originais para rastreabilidade.")


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
    "NovaVarejo Brasil | Projeto educacional de Ciência de Dados "
    "e apoio à decisão"
)
