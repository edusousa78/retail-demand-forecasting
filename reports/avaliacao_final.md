# Avaliacao final - NovaVarejo Brasil

## Projeto

Retail Demand Forecasting

## Protocolo

- Treinamento: 28/10/2015 a 24/04/2016.
- Teste: 25/04/2016 a 01/05/2016.
- Series: 30.
- Horizonte: 7 dias.
- Previsoes: 210.
- Selecao de series: volume no treino.
- Modelos: baseline, RF original e RF aprimorada.

## Resultados

| Modelo | WAPE |
|---|---:|
| Baseline 7 dias | 24.39% |
| RF original | 20.15% |
| RF aprimorada | 20.72% |

## Comparacao

Variacao relativa da RF aprimorada contra a
RF original: -2.85%.

## Conclusao

A RF aprimorada superou o baseline, mas nao superou a RF original.

## Limitacoes

- Avaliacao concentrada em 30 series de alto volume.
- Apenas uma semana de teste independente.
- O teste nao comprova desempenho em producao.
- Novos ajustes nao devem ser selecionados usando
  repetidamente esta mesma semana de teste.

## Proximos passos

- Revisar testes automatizados.
- Documentar metodologia e resultados.
- Preparar README e publicacao no GitHub.
