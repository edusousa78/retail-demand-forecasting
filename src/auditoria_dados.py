
from pathlib import Path
import pandas as pd

# Localiza a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

arquivos = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
]

for arquivo in arquivos:
    caminho = DATA_DIR / arquivo

    print(f"\n{'=' * 60}")
    print(f"ARQUIVO: {arquivo}")
    print(f"{'=' * 60}")

    # Le apenas as 5 primeiras linhas
    df = pd.read_csv(caminho, nrows=5)

    print(f"Primeiras 10 colunas: {df.columns[:10].tolist()}")
    print(f"Últimas 5 colunas: {df.columns[-5:].tolist()}")

    print("\nPrimeiras linhas:")
    print(df.iloc[:, :8].to_string(index=False))


print("\n" + "=" * 60)
print("AUDITORIA QUANTITATIVA")
print("=" * 60)

for arquivo in arquivos:
    caminho = DATA_DIR / arquivo

    print(f"\nArquivo: {arquivo}")

    # Carrega o arquivo completo para auditoria
    df = pd.read_csv(caminho)

    print(f"Linhas: {df.shape[0]:,}")
    print(f"Colunas: {df.shape[1]:,}")

    # Valores ausentes
    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]

    print(f"Colunas com valores ausentes: {len(ausentes)}")
    print(f"Total de valores ausentes: {ausentes.sum():,}")

    # Duplicidade das chaves de negocio
    chaves = {
        "calendar.csv": ["d"],
        "sales_train_validation.csv": ["id"],
        "sell_prices.csv": [
            "store_id", "item_id", "wm_yr_wk"
        ],
    }

    duplicados = df.duplicated(
        subset=chaves[arquivo]
    ).sum()

    print(f"Chaves duplicadas: {duplicados:,}")

    del df

# Investigacao dos valores ausentes no calendario
print("\n" + "=" * 60)
print("VALORES AUSENTES POR COLUNA - CALENDARIO")
print("=" * 60)

calendario = pd.read_csv(DATA_DIR / "calendar.csv")

ausentes_calendario = calendario.isna().sum()
ausentes_calendario = ausentes_calendario[
    ausentes_calendario > 0
]

print(ausentes_calendario.to_string())

print("\n" + "=" * 60)
print("VALIDACAO DE PRECOS")
print("=" * 60)

precos = pd.read_csv(DATA_DIR / "sell_prices.csv")

precos_invalidos = (precos["sell_price"] <= 0).sum()

print(f"Preços inválidos: {precos_invalidos}")
