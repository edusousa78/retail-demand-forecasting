
from pathlib import Path

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
DIAS_VALIDACAO = 7

ULTIMO_DIA = 1913

PRIMEIRO_DIA = (
    ULTIMO_DIA - DIAS_HISTORICO + 1
)

colunas_id = [
    "id",
    "item_id",
    "store_id",
]

colunas_dias = [
    f"d_{dia}"
    for dia in range(PRIMEIRO_DIA, ULTIMO_DIA + 1)
]

colunas_treino = colunas_dias[:-DIAS_VALIDACAO]


# ============================================================
# 2. CARREGAMENTO EFICIENTE
# ============================================================

print("=" * 60)
print("PREPARACAO DA AMOSTRA - NOVAVAREJO")
print("=" * 60)

print("\nCarregando vendas...")

tipos = {
    coluna: "int32"
    for coluna in colunas_dias
}

for coluna in colunas_id:
    tipos[coluna] = "string"

vendas = pd.read_csv(
    ARQUIVO_VENDAS,
    usecols=colunas_id + colunas_dias,
    dtype=tipos,
)

calendario = pd.read_csv(
    ARQUIVO_CALENDARIO,
    usecols=["d", "date"],
)

print(f"Series disponiveis: {len(vendas)}")
print(f"Dias carregados: {len(colunas_dias)}")


# ============================================================
# 3. SELECAO DAS SERIES
# ============================================================

print("\nSelecionando series com vendas efetivas...")

vendas["volume_treino"] = (
    vendas[colunas_treino].sum(axis=1)
)

ranking = vendas.sort_values(
    ["volume_treino", "id"],
    ascending=[False, True],
)

selecionadas = (
    ranking.head(QUANTIDADE_SERIES)
    [colunas_id + colunas_dias]
    .copy()
)

assert len(selecionadas) == QUANTIDADE_SERIES

assert selecionadas["id"].is_unique

assert (
    ranking.head(QUANTIDADE_SERIES)["volume_treino"] > 0
).all(), "Existem series selecionadas sem vendas no treino."

print(f"Series selecionadas: {len(selecionadas)}")


# ============================================================
# 4. TRANSFORMACAO WIDE PARA LONG
# ============================================================

print("\nTransformando vendas para formato longo...")

dados = selecionadas.melt(
    id_vars=colunas_id,
    value_vars=colunas_dias,
    var_name="d",
    value_name="vendas",
)

dados = dados.merge(
    calendario,
    on="d",
    how="left",
    validate="many_to_one",
    indicator=True,
)

assert dados["_merge"].eq("both").all(), (
    "Existem dias sem correspondencia no calendario."
)

dados = dados.drop(columns=["_merge"])

dados["date"] = pd.to_datetime(
    dados["date"],
    errors="raise",
)

dados = dados.sort_values(
    ["id", "date"]
).reset_index(drop=True)


# ============================================================
# 5. VALIDACOES DE QUALIDADE
# ============================================================

total_esperado = (
    QUANTIDADE_SERIES * DIAS_HISTORICO
)

assert len(dados) == total_esperado

assert dados["date"].notna().all()

assert dados["vendas"].notna().all()

assert (dados["vendas"] >= 0).all()

assert not dados.duplicated(
    subset=["id", "date"]
).any()


# ============================================================
# 6. DIVISAO TEMPORAL
# ============================================================

datas = sorted(dados["date"].unique())

data_corte = datas[-DIAS_VALIDACAO]

treino = dados[
    dados["date"] < data_corte
].copy()

validacao = dados[
    dados["date"] >= data_corte
].copy()

assert treino["date"].max() < validacao["date"].min()

assert len(treino) == (
    QUANTIDADE_SERIES
    * (DIAS_HISTORICO - DIAS_VALIDACAO)
)

assert len(validacao) == (
    QUANTIDADE_SERIES * DIAS_VALIDACAO
)

assert treino["id"].nunique() == QUANTIDADE_SERIES

assert validacao["id"].nunique() == QUANTIDADE_SERIES


# ============================================================
# 7. INDICADORES DE QUALIDADE
# ============================================================

vendas_treino = treino["vendas"].sum()
vendas_validacao = validacao["vendas"].sum()

dias_com_vendas = (
    dados["vendas"] > 0
).sum()

percentual_atividade = (
    dias_com_vendas / len(dados)
) * 100

print("\n" + "=" * 60)
print("RESULTADOS DA PREPARACAO")
print("=" * 60)

print(f"Series selecionadas: {QUANTIDADE_SERIES}")
print(f"Historico por serie: {DIAS_HISTORICO} dias")

print(f"\nTotal de registros: {len(dados)}")
print(f"Registros de treino: {len(treino)}")
print(f"Registros de validacao: {len(validacao)}")

print(
    f"\nPeriodo de treino: "
    f"{treino['date'].min().date()} "
    f"ate {treino['date'].max().date()}"
)

print(
    f"Periodo de validacao: "
    f"{validacao['date'].min().date()} "
    f"ate {validacao['date'].max().date()}"
)

print(f"\nVendas totais no treino: {vendas_treino}")
print(f"Vendas totais na validacao: {vendas_validacao}")

print(
    f"Observacoes com vendas positivas: "
    f"{percentual_atividade:.2f}%"
)

if vendas_validacao == 0:
    print(
        "\nATENCAO: a validacao nao possui "
        "vendas positivas."
    )


# ============================================================
# 8. EXPORTACAO
# ============================================================

arquivo_completo = (
    PROCESSED_DIR / "vendas_modelagem_long.csv"
)

arquivo_treino = (
    PROCESSED_DIR / "treino_modelagem.csv"
)

arquivo_validacao = (
    PROCESSED_DIR / "validacao_modelagem.csv"
)

dados.to_csv(
    arquivo_completo,
    index=False,
)

treino.to_csv(
    arquivo_treino,
    index=False,
)

validacao.to_csv(
    arquivo_validacao,
    index=False,
)

print("\nArquivos gerados:")
print(f"- {arquivo_completo}")
print(f"- {arquivo_treino}")
print(f"- {arquivo_validacao}")

print("\nPreparacao concluida com sucesso.")
