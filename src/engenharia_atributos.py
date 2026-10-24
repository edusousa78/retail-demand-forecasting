
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

entrada = PROCESSED_DIR / "amostra_vendas_long.csv"
saida = PROCESSED_DIR / "amostra_features.csv"

dados = pd.read_csv(
    entrada,
    parse_dates=["date"]
)

dados = dados.sort_values(
    ["id", "date"]
).reset_index(drop=True)

# Atributos conhecidos pelo calendario
dados["dia_semana"] = dados["date"].dt.day_of_week
dados["mes"] = dados["date"].dt.month
dados["fim_semana"] = (
    dados["dia_semana"] >= 5
).astype(int)

# Historico de vendas por produto e loja
grupo = dados.groupby("id")["vendas"]

dados["lag_1"] = grupo.shift(1)
dados["lag_3"] = grupo.shift(3)
dados["lag_7"] = grupo.shift(7)

# Media dos sete dias anteriores
dados["media_movel_7"] = (
    dados.groupby("id")["vendas"]
    .transform(
        lambda serie: (
            serie.shift(1)
            .rolling(window=7, min_periods=7)
            .mean()
        )
    )
)

# Validacoes
assert dados["lag_1"].isna().sum() == 5
assert dados["lag_3"].isna().sum() == 15
assert dados["lag_7"].isna().sum() == 35
assert dados["media_movel_7"].isna().sum() == 35

dados.to_csv(saida, index=False)

print("=" * 55)
print("ENGENHARIA DE ATRIBUTOS - NOVAVAREJO")
print("=" * 55)

print(f"Registros: {len(dados)}")
print(f"Produtos/lojas: {dados['id'].nunique()}")

colunas = [
    "date",
    "vendas",
    "dia_semana",
    "lag_1",
    "lag_3",
    "lag_7",
    "media_movel_7",
]

print("\nExemplo de atributos:")
print(dados[colunas].head(12).to_string(index=False))

print(f"\nArquivo gerado: {saida}")
print("Validações concluídas.")
