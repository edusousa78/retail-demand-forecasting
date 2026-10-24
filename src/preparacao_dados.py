
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Amostra controlada para validar a transformação
DIAS = 21
PRODUTOS_LOJAS = 5

colunas_id = [
    "id",
    "item_id",
    "store_id",
]

colunas_dias = [
    f"d_{dia}" for dia in range(1, DIAS + 1)
]

vendas = pd.read_csv(
    RAW_DIR / "sales_train_validation.csv",
    usecols=colunas_id + colunas_dias,
    nrows=PRODUTOS_LOJAS,
)

calendario = pd.read_csv(
    RAW_DIR / "calendar.csv",
    usecols=["d", "date"],
)

# Transformação wide -> long
vendas_long = vendas.melt(
    id_vars=colunas_id,
    value_vars=colunas_dias,
    var_name="d",
    value_name="vendas",
)

# Relacionamento entre vendas e calendário
dados = vendas_long.merge(
    calendario,
    on="d",
    how="left",
    validate="many_to_one",
)

dados["date"] = pd.to_datetime(dados["date"])

# Validações de integridade
esperado = PRODUTOS_LOJAS * DIAS

assert len(dados) == esperado, (
    "Quantidade inesperada de registros."
)

assert dados["date"].notna().all(), (
    "Existem dias sem correspondência no calendário."
)

assert (dados["vendas"] >= 0).all(), (
    "Existem vendas negativas."
)

dados = dados.sort_values(
    ["id", "date"]
).reset_index(drop=True)

saida = PROCESSED_DIR / "amostra_vendas_long.csv"

dados.to_csv(saida, index=False)

print("=" * 55)
print("PREPARAÇÃO DE DADOS - NOVAVAREJO")
print("=" * 55)

print(f"Registros originais: {len(vendas)}")
print(f"Dias analisados: {DIAS}")
print(f"Registros transformados: {len(dados)}")

print("\nPrimeiras observações:")
print(dados.head(10).to_string(index=False))

print(f"\nArquivo gerado: {saida}")
print("Validações concluídas.")
