
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACOES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ARQUIVO_PREVISOES = (
    PROCESSED_DIR
    / "experimento_atributos_previsoes.csv"
)

ARQUIVO_METRICAS = (
    PROCESSED_DIR
    / "selecao_modelo_metricas.csv"
)

ARQUIVO_DECISAO = (
    REPORTS_DIR
    / "decisao_modelo_aula15.md"
)

ARQUIVO_AVALIACAO = (
    RAW_DIR
    / "sales_train_evaluation.csv"
)

COLUNA_REAL = "vendas_reais"

MODELOS = {
    "baseline_media_7": "Baseline de 7 dias",
    "previsao_rf_original": "Random Forest original",
    "previsao_rf_atributos": "Random Forest aprimorada",
}

MODELO_ORIGINAL = "previsao_rf_original"
MODELO_NOVO = "previsao_rf_atributos"

CHAVES = [
    "janela",
    "id",
    "data_alvo",
    "horizonte",
]


# ============================================================
# 2. FUNCOES DE AVALIACAO
# ============================================================

def calcular_metricas(grupo, coluna_modelo):
    real = grupo[COLUNA_REAL].to_numpy(dtype=float)

    previsto = grupo[coluna_modelo].to_numpy(
        dtype=float
    )

    erro = previsto - real
    erro_absoluto = np.abs(erro)

    denominador = np.abs(real).sum()

    wape = (
        100 * erro_absoluto.sum() / denominador
        if denominador > 0
        else np.nan
    )

    return {
        "observacoes": len(grupo),
        "vendas_reais": float(real.sum()),
        "mae": float(erro_absoluto.mean()),
        "wape": float(wape),
        "vies_medio": float(erro.mean()),
        "erro_absoluto_total": float(
            erro_absoluto.sum()
        ),
    }


def avaliar(dados, dimensao, identificador):
    registros = []

    for coluna, nome in MODELOS.items():
        metricas = calcular_metricas(
            dados,
            coluna
        )

        registros.append({
            "dimensao": dimensao,
            "identificador": identificador,
            "modelo": nome,
            "coluna_modelo": coluna,
            **metricas,
        })

    return registros


def calcular_wape(grupo, coluna):
    return calcular_metricas(
        grupo,
        coluna
    )["wape"]


# ============================================================
# 3. CARREGAMENTO E VALIDACAO
# ============================================================

print("=" * 60)
print("SELECAO DO MODELO CANDIDATO - NOVAVAREJO")
print("=" * 60)

dados = pd.read_csv(
    ARQUIVO_PREVISOES,
    parse_dates=["data_alvo"],
)

colunas_obrigatorias = (
    CHAVES
    + [COLUNA_REAL]
    + list(MODELOS.keys())
)

faltantes = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in dados.columns
]

assert not faltantes, (
    f"Colunas ausentes: {faltantes}"
)

assert len(dados) == 840

assert not dados.duplicated(CHAVES).any()

assert dados[colunas_obrigatorias].notna().all().all()

assert dados["janela"].nunique() == 4
assert dados["horizonte"].nunique() == 7

assert (
    dados.groupby(
        ["janela", "id"]
    )["horizonte"]
    .nunique()
    .eq(7)
    .all()
)

assert (
    dados[
        [COLUNA_REAL] + list(MODELOS.keys())
    ] >= 0
).all().all()

print(f"\nPrevisoes verificadas: {len(dados)}")
print(f"Janelas: {dados['janela'].nunique()}")
print(f"Horizontes: {dados['horizonte'].nunique()}")


# ============================================================
# 4. CALCULAR METRICAS
# ============================================================

registros = avaliar(
    dados,
    "global",
    "global"
)

for janela, grupo in dados.groupby("janela"):
    registros.extend(
        avaliar(
            grupo,
            "janela",
            f"janela_{janela}"
        )
    )

for horizonte, grupo in dados.groupby("horizonte"):
    registros.extend(
        avaliar(
            grupo,
            "horizonte",
            f"horizonte_{horizonte}"
        )
    )

metricas = pd.DataFrame(registros)

global_modelos = metricas.loc[
    metricas["dimensao"] == "global"
].copy()

print("\n" + "=" * 60)
print("RESULTADOS GLOBAIS")
print("=" * 60)

print(
    global_modelos[
        [
            "modelo",
            "mae",
            "wape",
            "vies_medio",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 5. COMPARAR VERSOES
# ============================================================

metricas_original = calcular_metricas(
    dados,
    MODELO_ORIGINAL
)

metricas_novo = calcular_metricas(
    dados,
    MODELO_NOVO
)

wape_original = metricas_original["wape"]
wape_novo = metricas_novo["wape"]

ganho_relativo = (
    100 * (wape_original - wape_novo)
    / wape_original
)

melhorias_janelas = 0
melhorias_horizontes = 0

resultado_janelas = []
resultado_horizontes = []

for janela, grupo in dados.groupby("janela"):
    anterior = calcular_wape(
        grupo,
        MODELO_ORIGINAL
    )

    novo = calcular_wape(
        grupo,
        MODELO_NOVO
    )

    melhorou = novo < anterior

    melhorias_janelas += int(melhorou)

    resultado_janelas.append({
        "janela": janela,
        "wape_original": anterior,
        "wape_novo": novo,
        "melhorou": melhorou,
    })

for horizonte, grupo in dados.groupby("horizonte"):
    anterior = calcular_wape(
        grupo,
        MODELO_ORIGINAL
    )

    novo = calcular_wape(
        grupo,
        MODELO_NOVO
    )

    melhorou = novo < anterior

    melhorias_horizontes += int(melhorou)

    resultado_horizontes.append({
        "horizonte": horizonte,
        "wape_original": anterior,
        "wape_novo": novo,
        "melhorou": melhorou,
    })


# ============================================================
# 6. ANALISAR VIES
# ============================================================

vies_original = abs(
    metricas_original["vies_medio"]
)

vies_novo = abs(
    metricas_novo["vies_medio"]
)

# Tolerancia de 10% na magnitude do vies.
# Se o vies original for praticamente zero,
# usamos uma tolerancia absoluta de 0,1 unidade.
limite_vies = max(
    vies_original * 1.10,
    0.10
)

vies_aceitavel = (
    vies_novo <= limite_vies
)


# ============================================================
# 7. APLICAR CRITERIOS DE SELECAO
# ============================================================

criterios = {
    "WAPE global menor": (
        wape_novo < wape_original
    ),
    "Melhoria nas 4 janelas": (
        melhorias_janelas == 4
    ),
    "Melhoria em pelo menos 5 horizontes": (
        melhorias_horizontes >= 5
    ),
    "Vies dentro da tolerancia": (
        vies_aceitavel
    ),
    "840 previsoes validas": (
        len(dados) == 840
    ),
}

aprovado = all(criterios.values())

modelo_selecionado = (
    "Random Forest aprimorada"
    if aprovado
    else "Random Forest original"
)

print("\n" + "=" * 60)
print("CRITERIOS DE SELECAO")
print("=" * 60)

for criterio, passou in criterios.items():
    status = "APROVADO" if passou else "REPROVADO"

    print(f"{criterio}: {status}")

print("\n" + "=" * 60)
print("DECISAO TECNICA")
print("=" * 60)

print(f"WAPE original: {wape_original:.4f}%")
print(f"WAPE aprimorado: {wape_novo:.4f}%")

print(
    f"Reducao relativa do erro: "
    f"{ganho_relativo:+.2f}%"
)

print(
    f"Janelas melhoradas: "
    f"{melhorias_janelas}/4"
)

print(
    f"Horizontes melhorados: "
    f"{melhorias_horizontes}/7"
)

print(
    f"Modelo candidato selecionado: "
    f"{modelo_selecionado}"
)


# ============================================================
# 8. VERIFICAR DISPONIBILIDADE DE TESTE FUTURO
# ============================================================

print("\n" + "=" * 60)
print("VERIFICACAO DO CONJUNTO DE TESTE FUTURO")
print("=" * 60)

colunas_teste = [
    f"d_{dia}"
    for dia in range(1914, 1942)
]

teste_disponivel = False

if ARQUIVO_AVALIACAO.exists():
    cabecalho = pd.read_csv(
        ARQUIVO_AVALIACAO,
        nrows=0
    )

    colunas_presentes = set(cabecalho.columns)

    teste_disponivel = all(
        coluna in colunas_presentes
        for coluna in colunas_teste
    )

    if teste_disponivel:
        print(
            "Arquivo de avaliacao encontrado "
            "com os 28 dias futuros."
        )
    else:
        print(
            "Arquivo encontrado, mas sem todas "
            "as colunas d_1914 a d_1941."
        )
else:
    print(
        "Arquivo sales_train_evaluation.csv "
        "ainda nao encontrado."
    )

print(
    "Dados futuros nao foram utilizados "
    "na selecao do modelo."
)


# ============================================================
# 9. GERAR RELATORIO DE DECISAO
# ============================================================

linhas_criterios = "\n".join(
    (
        f"- {'APROVADO' if passou else 'REPROVADO'}"
        f": {criterio}"
    )
    for criterio, passou in criterios.items()
)

linhas_janelas = "\n".join(
    (
        f"- Janela {item['janela']}: "
        f"{item['wape_original']:.2f}% -> "
        f"{item['wape_novo']:.2f}%"
    )
    for item in resultado_janelas
)

linhas_horizontes = "\n".join(
    (
        f"- Horizonte {item['horizonte']}: "
        f"{item['wape_original']:.2f}% -> "
        f"{item['wape_novo']:.2f}%"
    )
    for item in resultado_horizontes
)

status_teste = (
    "Arquivo futuro disponivel para a proxima etapa."
    if teste_disponivel
    else "Arquivo futuro ainda indisponivel."
)

relatorio = f"""# Decisao de modelagem - NovaVarejo Brasil

## Projeto
Retail Demand Forecasting

## Objetivo
Prever vendas por produto-loja para os proximos 7 dias.

## Resultado do experimento

- WAPE Random Forest original: {wape_original:.4f}%
- WAPE Random Forest aprimorada: {wape_novo:.4f}%
- Reducao relativa do erro: {ganho_relativo:.2f}%
- Janelas melhoradas: {melhorias_janelas}/4
- Horizontes melhorados: {melhorias_horizontes}/7
- Vies medio original: {metricas_original['vies_medio']:.4f}
- Vies medio aprimorado: {metricas_novo['vies_medio']:.4f}

## Criterios de selecao

{linhas_criterios}

## Resultados por janela

{linhas_janelas}

## Resultados por horizonte

{linhas_horizontes}

## Decisao

Modelo candidato: **{modelo_selecionado}**

Esta e uma selecao para a proxima etapa de avaliacao,
nao uma aprovacao para producao.

## Limitacoes

- A amostra considera 30 series de alto volume.
- As quatro janelas foram utilizadas no desenvolvimento.
- O ganho pode nao se repetir em outros produtos ou periodos.
- Ainda nao foi executado um teste futuro independente.
- O vies de previsao ainda exige acompanhamento.

## Proxima etapa

{status_teste}

Avaliar a versao candidata em dados futuros nao utilizados
para escolher atributos ou hiperparametros.

O conjunto de teste nao deve ser utilizado para novos
ajustes repetitivos do modelo.
"""

ARQUIVO_DECISAO.write_text(
    relatorio,
    encoding="utf-8"
)

metricas.to_csv(
    ARQUIVO_METRICAS,
    index=False
)

print("\nArquivos gerados:")
print(f"- {ARQUIVO_METRICAS}")
print(f"- {ARQUIVO_DECISAO}")

print("\nSelecao do modelo concluida com sucesso.")
