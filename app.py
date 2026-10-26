
from pathlib import Path

import altair as alt
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

# Prioriza os resultados locais.
# Na nuvem, utiliza os dados publicados de demonstracao.
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
    page_title="NovaVarejo | Gestão de Estoque",
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


ESTADOS = {
    "CA": "Califórnia",
    "TX": "Texas",
    "WI": "Wisconsin",
}

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
    """Converte o codigo M5 em um nome amigavel."""

    partes = str(codigo).split("_")

    if len(partes) == 2 and partes[0] in ESTADOS:
        estado = ESTADOS[partes[0]]
        numero = partes[1].zfill(2)

        return f"Loja {estado} {numero}"

    return f"Loja {codigo}"


def nome_produto(codigo):
    """
    Cria uma descricao amigavel para o produto.

    O M5 nao fornece nomes comerciais completos.
    """

    partes = str(codigo).split("_")

    if len(partes) == 3 and partes[0] in CATEGORIAS:
        categoria, grupo, numero = partes

        return (
            f"{CATEGORIAS[categoria]} · "
            f"Grupo {grupo} · "
            f"Produto {numero}"
        )

    return f"Produto {codigo}"


def nome_prioridade(codigo):
    return PRIORIDADES.get(
        str(codigo),
        str(codigo).capitalize(),
    )


@st.cache_data(show_spinner="Carregando planejamento...")
def ler_csv(caminho, coluna_data):
    return pd.read_csv(
        caminho,
        parse_dates=[coluna_data],
    )


def carregar_dados():
    """Carrega e valida os arquivos do planejamento."""

    for arquivo in (ARQUIVO_PLANO, ARQUIVO_DIARIO):
        if not arquivo.is_file():
            st.error(
                "Arquivo necessário não encontrado: "
                f"{arquivo.name}"
            )

            st.info(
                "Execute `python src/planejamento_reposicao.py` "
                "para gerar os dados locais ou disponibilize "
                "os dois arquivos CSV em `data/demo/`."
            )

            st.stop()

    try:
        plano = ler_csv(
            ARQUIVO_PLANO,
            "data_referencia",
        )

        diario = ler_csv(
            ARQUIVO_DIARIO,
            "data_alvo",
        )

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
    ) as erro:

        st.error(
            f"Não foi possível carregar os dados: {erro}"
        )

        st.stop()

    colunas_plano = {
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
    }

    colunas_diario = {
        "id",
        "item_id",
        "store_id",
        "data_alvo",
        "horizonte",
        "demanda_prevista",
        "saldo_final_projetado",
        "risco_ruptura_dia",
    }

    faltantes_plano = sorted(
        colunas_plano - set(plano.columns)
    )

    faltantes_diario = sorted(
        colunas_diario - set(diario.columns)
    )

    if faltantes_plano or faltantes_diario:
        st.error(
            "Os arquivos CSV não possuem todas "
            "as colunas esperadas."
        )

        st.write(
            "Colunas ausentes no plano:",
            faltantes_plano,
        )

        st.write(
            "Colunas ausentes no diário:",
            faltantes_diario,
        )

        st.stop()

    if plano.empty or diario.empty:
        st.error("Os arquivos de dados estão vazios.")
        st.stop()

    if plano["id"].duplicated().any():
        st.error(
            "Existem séries duplicadas no planejamento."
        )
        st.stop()

    if diario.duplicated(
        ["id", "data_alvo"]
    ).any():

        st.error(
            "Existem previsões diárias duplicadas."
        )
        st.stop()

    if set(diario["id"]) != set(plano["id"]):
        st.error(
            "As séries dos dois arquivos não correspondem."
        )
        st.stop()

    soma_diaria = (
        diario.groupby("id")["demanda_prevista"]
        .sum()
        .sort_index()
    )

    demanda_plano = (
        plano.set_index("id")["demanda_prevista_7d"]
        .sort_index()
    )

    if not np.allclose(
        soma_diaria,
        demanda_plano,
        atol=0.01,
        rtol=0,
    ):
        st.error(
            "Inconsistência entre as previsões diárias "
            "e o plano de reposição."
        )

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

st.markdown(
    "### Planejamento de reposição com previsão de demanda"
)

st.warning(
    "**Demonstração histórica:** previsões de "
    f"{data_inicio:%d/%m/%Y} a "
    f"{data_fim:%d/%m/%Y}. "
    "Estoques, pedidos em trânsito e prazos de entrega "
    "são simulados. Os resultados não devem ser "
    "utilizados para compras reais."
)

st.caption(
    f"Dados: {MODO_DADOS} | "
    f"Referência: {data_referencia:%d/%m/%Y} | "
    f"Modelo: {modelo}"
)


# ============================================================
# 4. FILTROS GERENCIAIS
# ============================================================

def limpar_filtros():
    """Restaura todos os filtros para a visao geral."""

    st.session_state["filtro_lojas"] = []
    st.session_state["filtro_urgencias"] = []
    st.session_state["filtro_produto"] = None


st.sidebar.header("Filtros gerenciais")

st.sidebar.caption(
    "Selecione apenas o que deseja filtrar. "
    "Sem seleção de lojas ou urgências, "
    "todas são consideradas."
)

lojas = sorted(
    plano["store_id"].unique().tolist()
)

lojas_selecionadas = st.sidebar.multiselect(
    "Lojas",
    options=lojas,
    default=[],
    key="filtro_lojas",
    format_func=nome_loja,
    placeholder="Todas as lojas",
    help=(
        "Deixe vazio para incluir todas as lojas. "
        "Selecione uma ou mais para restringir a análise."
    ),
)

prioridades_disponiveis = [
    prioridade
    for prioridade in (
        "CRITICA",
        "ALTA",
        "MEDIA",
        "BAIXA",
    )
    if prioridade in plano["prioridade"].unique()
]

prioridades_selecionadas = st.sidebar.multiselect(
    "Urgência de reposição",
    options=prioridades_disponiveis,
    default=[],
    key="filtro_urgencias",
    format_func=nome_prioridade,
    placeholder="Todas as urgências",
    help=(
        "Deixe vazio para considerar "
        "todas as prioridades."
    ),
)

produtos = sorted(
    plano["item_id"].unique().tolist()
)

produto_selecionado = st.sidebar.selectbox(
    "Produto",
    options=[None] + produtos,
    index=0,
    key="filtro_produto",
    format_func=lambda codigo: (
        "Todos os produtos"
        if codigo is None
        else nome_produto(codigo)
    ),
    help=(
        "As descrições são baseadas nos códigos M5, "
        "não em nomes comerciais."
    ),
)

st.sidebar.button(
    "Limpar filtros",
    on_click=limpar_filtros,
    use_container_width=True,
    help="Restaura a visão geral do planejamento.",
)

# Uma selecao vazia significa considerar todos os registros.

lojas_ativas = (
    lojas_selecionadas
    if lojas_selecionadas
    else lojas
)

prioridades_ativas = (
    prioridades_selecionadas
    if prioridades_selecionadas
    else prioridades_disponiveis
)

filtrado = plano.loc[
    plano["store_id"].isin(lojas_ativas)
    & plano["prioridade"].isin(prioridades_ativas)
].copy()

if produto_selecionado is not None:
    filtrado = filtrado.loc[
        filtrado["item_id"] == produto_selecionado
    ].copy()

if filtrado.empty:
    st.info(
        "Nenhuma combinação de produto e loja "
        "corresponde aos filtros selecionados. "
        "Altere as opções ou clique em Limpar filtros."
    )

    st.stop()

diario_filtrado = diario.loc[
    diario["id"].isin(filtrado["id"])
].copy()

st.sidebar.divider()

st.sidebar.caption(
    f"{len(filtrado)} combinações de produto "
    "e loja selecionadas."
)

st.sidebar.caption(
    f"Fonte: {MODO_DADOS}"
)


# ============================================================
# 5. INDICADORES GERENCIAIS
# ============================================================

st.subheader("Visão geral do estoque")

st.caption(
    "Indicadores para o período selecionado. "
    "Todos os valores respondem aos filtros "
    "de loja, urgência e produto."
)

demanda_total = float(
    filtrado["demanda_prevista_7d"].sum()
)

estoque_total = float(
    filtrado["estoque_atual_simulado"].sum()
)

reposicao_total = float(
    filtrado["quantidade_reposicao_sugerida"].sum()
)

quantidade_series = len(filtrado)

series_risco = int(
    filtrado["risco_ruptura"].sum()
)

series_criticas = int(
    (filtrado["prioridade"] == "CRITICA").sum()
)

series_altas = int(
    (filtrado["prioridade"] == "ALTA").sum()
)

series_sem_ruptura = (
    quantidade_series - series_risco
)

horizonte_dias = int(
    diario_filtrado["data_alvo"].nunique()
)

media_diaria = (
    demanda_total / horizonte_dias
    if horizonte_dias
    else 0.0
)

percentual_risco = (
    100.0 * series_risco / quantidade_series
    if quantidade_series
    else 0.0
)

col_demanda, col_estoque, col_repor, col_risco = (
    st.columns(4, gap="small")
)

with col_demanda:
    with st.container(border=True):

        st.metric(
            label="Demanda prevista · 7 dias",
            value=formatar_numero(
                demanda_total,
                2,
            ),
            help=(
                "Soma das previsões de demanda "
                "das séries selecionadas."
            ),
        )

        st.caption(
            f"**{formatar_numero(media_diaria, 2)} un./dia** "
            "em média no horizonte."
        )

with col_estoque:
    with st.container(border=True):

        st.metric(
            label="Estoque inicial simulado",
            value=formatar_numero(
                estoque_total
            ),
            help=(
                "Soma dos estoques simulados "
                "na data de referência."
            ),
        )

        st.caption(
            "Saldo disponível antes das vendas previstas; "
            "não representa estoque real."
        )

with col_repor:
    with st.container(border=True):

        st.metric(
            label="Reposição sugerida",
            value=formatar_numero(
                reposicao_total
            ),
            help=(
                "Soma das quantidades sugeridas "
                "por produto e loja."
            ),
        )

        st.caption(
            "Unidades sugeridas para o plano; "
            "o prazo de entrega pode impedir "
            "evitar rupturas."
        )

with col_risco:
    with st.container(border=True):

        st.metric(
            label="Séries com risco de ruptura",
            value=(
                f"{formatar_numero(percentual_risco, 1)}%"
            ),
            help=(
                "Percentual de combinações produto-loja "
                "com risco de ruptura projetado."
            ),
        )

        st.caption(
            f"**{series_risco} de {quantidade_series}** "
            "combinações produto-loja."
        )

st.progress(
    min(
        max(percentual_risco / 100.0, 0.0),
        1.0,
    ),
    text=(
        "Participação das combinações "
        "com risco de ruptura"
    ),
)

st.caption(
    f"**Prioridade crítica:** {series_criticas} · "
    f"**Prioridade alta:** {series_altas} · "
    f"**Sem ruptura projetada:** {series_sem_ruptura}"
)

st.caption(
    "A análise é histórica e utiliza estoque simulado. "
    "A soma de unidades não substitui a avaliação "
    "individual de cada produto e loja."
)


# ============================================================
# 6. GRAFICOS DE DEMANDA E ESTOQUE
# ============================================================

st.divider()

st.subheader(
    "Projeção histórica de demanda e estoque — 7 dias"
)

st.caption(
    f"Período analisado: "
    f"{data_inicio:%d/%m/%Y} a "
    f"{data_fim:%d/%m/%Y}. "
    "Os gráficos apresentam datas no formato DD/MM."
)

demanda_diaria = (
    diario_filtrado
    .groupby("data_alvo")
    .agg(
        demanda_prevista=(
            "demanda_prevista",
            "sum",
        ),
        saldo_projetado=(
            "saldo_final_projetado",
            "sum",
        ),
    )
    .sort_index()
    .reset_index()
)

# CORRECAO DOS ROTULOS DE DATA
#
# Em vez de usar diretamente uma data temporal,
# criamos categorias DD/MM.
#
# Assim, o Altair nao insere horarios nem
# abreviacoes dos dias em ingles.

demanda_diaria["Data"] = (
    demanda_diaria["data_alvo"]
    .dt.strftime("%d/%m")
)

ordem_datas = (
    demanda_diaria["Data"].tolist()
)

eixo_datas = alt.X(
    "Data:N",
    sort=ordem_datas,
    scale=alt.Scale(
        domain=ordem_datas,
    ),
    title="Data (DD/MM)",
    axis=alt.Axis(
        values=ordem_datas,
        labelAngle=0,
        labelOverlap=False,
    ),
)

esquerda, direita = st.columns(2)

with esquerda:

    st.markdown(
        "**Demanda prevista por dia**"
    )

    grafico_demanda = (
        alt.Chart(demanda_diaria)
        .mark_bar(size=35)
        .encode(
            x=eixo_datas,
            y=alt.Y(
                "demanda_prevista:Q",
                title="Unidades previstas",
            ),
            tooltip=[
                alt.Tooltip(
                    "Data:N",
                    title="Dia",
                ),
                alt.Tooltip(
                    "demanda_prevista:Q",
                    title="Demanda (un.)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=290)
    )

    st.altair_chart(
        grafico_demanda,
        use_container_width=True,
    )

with direita:

    st.markdown(
        "**Saldo projetado sem novas compras**"
    )

    grafico_saldo = (
        alt.Chart(demanda_diaria)
        .mark_line(
            point=True,
            strokeWidth=3,
        )
        .encode(
            x=eixo_datas,
            y=alt.Y(
                "saldo_projetado:Q",
                title="Saldo projetado (un.)",
            ),
            tooltip=[
                alt.Tooltip(
                    "Data:N",
                    title="Dia",
                ),
                alt.Tooltip(
                    "saldo_projetado:Q",
                    title="Saldo (un.)",
                    format=",.2f",
                ),
            ],
        )
        .properties(height=290)
    )

    st.altair_chart(
        grafico_saldo,
        use_container_width=True,
    )

st.caption(
    "Um saldo projetado negativo representa "
    "demanda não atendida acumulada, "
    "e não estoque físico negativo."
)


# ============================================================
# 7. NECESSIDADE DE REPOSICAO POR LOJA
# ============================================================

st.divider()

st.subheader(
    "Necessidade de reposição por loja"
)

lojas_resumo = (
    filtrado
    .groupby("store_id")
    .agg(
        produtos=("id", "count"),
        demanda_prevista=(
            "demanda_prevista_7d",
            "sum",
        ),
        estoque_simulado=(
            "estoque_atual_simulado",
            "sum",
        ),
        reposicao_sugerida=(
            "quantidade_reposicao_sugerida",
            "sum",
        ),
        series_em_risco=(
            "risco_ruptura",
            "sum",
        ),
    )
    .reset_index()
    .sort_values(
        "reposicao_sugerida",
        ascending=False,
    )
)

lojas_resumo["Loja"] = (
    lojas_resumo["store_id"].map(nome_loja)
)

dados_grafico_lojas = (
    lojas_resumo
    .set_index("Loja")[
        ["reposicao_sugerida"]
    ]
    .rename(
        columns={
            "reposicao_sugerida":
                "Unidades para reposição"
        }
    )
)

st.bar_chart(
    dados_grafico_lojas,
    x_label="Loja",
    y_label="Unidades sugeridas",
)

tabela_lojas = lojas_resumo.rename(
    columns={
        "produtos": "Produtos/lojas",
        "demanda_prevista": "Demanda prevista",
        "estoque_simulado": "Estoque simulado",
        "reposicao_sugerida": "Reposição sugerida",
        "series_em_risco": "Produtos/lojas em risco",
    }
)

st.dataframe(
    tabela_lojas[
        [
            "Loja",
            "Produtos/lojas",
            "Demanda prevista",
            "Estoque simulado",
            "Reposição sugerida",
            "Produtos/lojas em risco",
        ]
    ].round(2),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 8. PLANO DETALHADO DE REPOSICAO
# ============================================================

st.divider()

st.subheader(
    "Plano detalhado de reposição"
)

ordem_prioridade = {
    "CRITICA": 0,
    "ALTA": 1,
    "MEDIA": 2,
    "BAIXA": 3,
}

detalhes = filtrado.copy()

detalhes["_ordem"] = (
    detalhes["prioridade"].map(ordem_prioridade)
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

tabela = detalhes[colunas_tabela].copy()

tabela["item_id"] = (
    tabela["item_id"].map(nome_produto)
)

tabela["store_id"] = (
    tabela["store_id"].map(nome_loja)
)

tabela["prioridade"] = (
    tabela["prioridade"].map(nome_prioridade)
)

tabela = tabela.rename(
    columns={
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
    }
)

st.dataframe(
    tabela.round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "**Crítica:** ruptura projetada antes ou "
    "até o prazo de entrega. "
    "**Alta:** ruptura posterior ao prazo, "
    "dentro dos sete dias."
)

st.caption(
    "Os nomes dos produtos são descrições "
    "dos códigos M5, não nomes comerciais."
)


# ============================================================
# 9. EXPORTACAO DO PLANEJAMENTO
# ============================================================

st.divider()

st.subheader(
    "Exportar plano de reposição"
)

exportacao = detalhes.drop(
    columns=["_ordem"]
).copy()

exportacao.insert(
    0,
    "nome_produto",
    exportacao["item_id"].map(nome_produto),
)

exportacao.insert(
    1,
    "nome_loja",
    exportacao["store_id"].map(nome_loja),
)

exportacao.insert(
    2,
    "urgencia",
    exportacao["prioridade"].map(nome_prioridade),
)

csv_exportacao = (
    exportacao.to_csv(
        index=False,
        sep=";",
        decimal=",",
    )
    .encode("utf-8-sig")
)

st.download_button(
    label="Baixar plano de reposição (CSV)",
    data=csv_exportacao,
    file_name="novavarejo_plano_reposicao.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "O arquivo CSV inclui nomes amigáveis "
    "e códigos originais para rastreabilidade."
)


# ============================================================
# 10. METODOLOGIA E LIMITACOES
# ============================================================

with st.expander(
    "Metodologia, premissas e limitações"
):

    st.markdown(
        """
        ### Origem das previsões

        As previsões foram geradas pela Random Forest
        aprimorada, selecionada durante o desenvolvimento.

        No teste independente, a Random Forest original
        apresentou menor WAPE. Portanto, a superioridade
        da versão aprimorada não foi confirmada.

        ### Dados operacionais

        - Estoques, pedidos em trânsito e prazos são simulados.
        - O estoque de segurança utiliza uma heurística.
        - Horizonte de planejamento: sete dias.
        - Os CSVs de demonstração contêm previsões históricas
          derivadas do M5, combinadas com dados operacionais
          simulados.

        ### Cálculo de reposição

        Quantidade sugerida = máximo entre zero e o
        arredondamento para cima de:

        demanda prevista + estoque de segurança
        - estoque atual - pedidos em trânsito considerados
        no horizonte.

        ### Restrições

        - Não considera lotes mínimos, custos ou capacidade.
        - Uma compra pode chegar tarde demais para evitar ruptura.
        - Não existe integração em tempo real com sistemas empresariais.
        - Os resultados não representam recomendações de compras reais.

        ### Interpretação

        Demonstração histórica de apoio à decisão,
        desenvolvida para portfólio de Ciência de Dados.
        """
    )


# ============================================================
# 11. RODAPE
# ============================================================

st.divider()

st.caption(
    "NovaVarejo Brasil | Projeto educacional "
    "de Ciência de Dados e apoio à decisão"
)
