# Decisao de modelagem - NovaVarejo Brasil

## Projeto
Retail Demand Forecasting

## Objetivo
Prever vendas por produto-loja para os proximos 7 dias.

## Resultado do experimento

- WAPE Random Forest original: 22.0747%
- WAPE Random Forest aprimorada: 21.4952%
- Reducao relativa do erro: 2.63%
- Janelas melhoradas: 4/4
- Horizontes melhorados: 5/7
- Vies medio original: 1.9611
- Vies medio aprimorado: 1.9642

## Criterios de selecao

- APROVADO: WAPE global menor
- APROVADO: Melhoria nas 4 janelas
- APROVADO: Melhoria em pelo menos 5 horizontes
- APROVADO: Vies dentro da tolerancia
- APROVADO: 840 previsoes validas

## Resultados por janela

- Janela 1: 26.19% -> 24.97%
- Janela 2: 18.90% -> 17.97%
- Janela 3: 19.03% -> 18.95%
- Janela 4: 24.69% -> 24.56%

## Resultados por horizonte

- Horizonte 1: 25.10% -> 25.15%
- Horizonte 2: 23.12% -> 23.05%
- Horizonte 3: 24.16% -> 24.53%
- Horizonte 4: 26.19% -> 24.37%
- Horizonte 5: 20.96% -> 19.39%
- Horizonte 6: 18.28% -> 17.68%
- Horizonte 7: 19.95% -> 19.72%

## Decisao

Modelo candidato: **Random Forest aprimorada**

Esta e uma selecao para a proxima etapa de avaliacao,
nao uma aprovacao para producao.

## Limitacoes

- A amostra considera 30 series de alto volume.
- As quatro janelas foram utilizadas no desenvolvimento.
- O ganho pode nao se repetir em outros produtos ou periodos.
- Ainda nao foi executado um teste futuro independente.
- O vies de previsao ainda exige acompanhamento.

## Proxima etapa

Arquivo futuro ainda indisponivel.

Avaliar a versao candidata em dados futuros nao utilizados
para escolher atributos ou hiperparametros.

O conjunto de teste nao deve ser utilizado para novos
ajustes repetitivos do modelo.
