# Planejamento de estoque - NovaVarejo

## Contexto

Demonstracao de apoio a decisao de reposicao
utilizando previsoes de demanda e estoque simulado.

**Data de referencia:** 2016-04-24

**Periodo previsto:** 2016-04-25 a 2016-05-01

**Modelo:** Random Forest aprimorada

## Indicadores

- Series avaliadas: 30
- Demanda prevista: 10739.69 unidades
- Estoque simulado: 6515 unidades
- Reposicao sugerida: 4279 unidades
- Series com risco de ruptura: 21
- Prioridade critica: 8
- Prioridade alta: 13

## Prioridades de reposicao

| item_id | store_id | prioridade | quantidade_reposicao_sugerida |
| --- | --- | --- | --- |
| FOODS_3_090 | CA_3 | CRITICA | 680 |
| FOODS_3_090 | WI_3 | CRITICA | 406 |
| FOODS_3_090 | TX_2 | CRITICA | 280 |
| FOODS_3_586 | CA_2 | CRITICA | 233 |
| FOODS_3_252 | CA_3 | CRITICA | 228 |
| FOODS_3_282 | CA_3 | CRITICA | 216 |
| FOODS_3_252 | TX_3 | CRITICA | 196 |
| FOODS_3_226 | WI_1 | CRITICA | 136 |
| FOODS_3_090 | CA_2 | ALTA | 289 |
| FOODS_3_090 | TX_3 | ALTA | 222 |

## Premissas

- As previsoes sao provenientes do modelo treinado.
- Os saldos de estoque sao simulados.
- Os pedidos em transito sao simulados.
- Os prazos de entrega sao simulados.
- O estoque de seguranca usa uma heuristica
  baseada na variabilidade historica.
- Os pedidos em transito sao recebidos no inicio
  do dia indicado.
- A simulacao diaria nao considera novas compras
  recomendadas pelo sistema.
- A quantidade sugerida considera demanda de
  sete dias, estoque de seguranca, saldo inicial
  e pedidos em transito no horizonte.
- Uma nova compra pode chegar tarde demais
  para evitar uma ruptura antecipada.
- As quantidades sugeridas nao consideram
  lotes minimos, custos ou restricoes de fornecedor.

## Limitacoes

Esta e uma simulacao historica para portfolio.

Nao representa estoque real ou uma recomendacao
operacional pronta para execucao.

A RF aprimorada foi a candidata selecionada
no desenvolvimento, mas nao superou a RF original
no teste independente.

A demonstracao nao utiliza vendas reais futuras
para calcular o planejamento.

## Proxima etapa

Construir dashboard gerencial com Streamlit.
