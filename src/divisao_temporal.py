
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

entrada = PROCESSED_DIR / "amostra_features.csv"

dados = pd.read_csv(
    entrada,
    parse_dates=["date"]
)

dados = dados.sort_values(
    ["id", "date"]
).reset_index(drop=True)

# Reservar os ultimos sete dias para validacao
dias_validacao = 5

datas = sorted(dados["date"].unique())

assert len(datas) > dias_validacao, (
    "Historico insuficiente para divisao temporal."
)

data_corte = datas[-dias_validacao]

treino = dados[
    dados["date"] < data_corte
].copy()

validacao = dados[
    dados["date"] >= data_corte
].copy()

# Garantir separacao cronologica
assert treino["date"].max() < validacao["date"].min()

# Garantir que todas as series estejam presentes
assert treino["id"].nunique() == dados["id"].nunique()
assert validacao["id"].nunique() == dados["id"].nunique()

treino.to_csv(
    PROCESSED_DIR / "treino.csv",
    index=False
)

validacao.to_csv(
    PROCESSED_DIR / "validacao.csv",
    index=False
)

print("=" * 55)
print("DIVISAO TEMPORAL - NOVAVAREJO")
print("=" * 55)

print(f"Total de registros: {len(dados)}")
print(f"Registros de treino: {len(treino)}")
print(f"Registros de validacao: {len(validacao)}")

print(
    f"\nTreino: {treino['date'].min().date()} "
    f"ate {treino['date'].max().date()}"
)

print(
    f"Validacao: {validacao['date'].min().date()} "
    f"ate {validacao['date'].max().date()}"
)

print("\nValidacoes concluidas.")
