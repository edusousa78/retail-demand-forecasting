# Retail Demand Forecasting

## Previsão de demanda no varejo com Machine Learning

Projeto de Ciência de Dados voltado à previsão de vendas
diárias por produto e loja, utilizando Python, Pandas,
Scikit-learn e técnicas de validação temporal.

**Status:** estudo de caso concluído e avaliado fora da
amostra. Não implantado em produção.

## 1. Contexto de negócio

A NovaVarejo Brasil é uma empresa fictícia utilizada
para contextualizar este estudo de caso.

O desafio consiste em estimar a quantidade de produtos
que será vendida nos próximos sete dias em cada loja.

Previsões de demanda podem apoiar decisões relacionadas
a estoque, reposição, compras e planejamento operacional.

## 2. Objetivo

Desenvolver e avaliar modelos capazes de prever vendas
diárias por combinação de produto e loja, com horizonte
de sete dias.

O projeto compara:

- Baseline baseado na média dos últimos sete dias.
- Random Forest com atributos temporais básicos.
- Random Forest com engenharia de atributos adicional.

## 3. Dados

Fonte: competição M5 Forecasting - Accuracy, Kaggle.

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

O conjunto original contém 30.490 séries temporais.

Para manter o experimento computacionalmente acessível,
foram selecionadas 30 séries de maior volume de vendas,
sempre considerando apenas o histórico disponível no
momento da seleção.

O estudo utiliza:

- Histórico de modelagem: 28/10/2015 a 24/04/2016.
- Backtesting: quatro janelas consecutivas de sete dias.
- Teste independente: 25/04/2016 a 01/05/2016.
- Horizonte: sete dias.
- Granularidade: produto, loja e dia.

Os arquivos originais não são distribuídos neste
repositório. Devem ser obtidos na fonte indicada.

## 4. Metodologia

### 4.1 Preparação dos dados

Os dados foram convertidos do formato largo para
o formato longo, com uma observação por produto,
loja e data.

### 4.2 Engenharia de atributos

Foram utilizados atributos como:

- Vendas observadas na data de origem.
- Defasagens temporais (lags).
- Médias móveis de vendas.
- Desvio-padrão móvel.
- Tendência entre médias de diferentes períodos.
- Dia da semana e mês da data prevista.
- Indicador de fim de semana.
- Horizonte de previsão.
- Identificação da série temporal.

Os atributos foram calculados com informações
disponíveis até a data de origem da previsão.

### 4.3 Modelagem

O modelo principal é um RandomForestRegressor,
implementado com Scikit-learn.

Configuração utilizada:

- 100 árvores.
- Profundidade máxima de 12.
- Mínimo de três amostras por folha.
- Random state igual a 42.

Foi utilizada uma estratégia de previsão direta
para os sete horizontes.

### 4.4 Validação temporal

O projeto utiliza backtesting com quatro janelas
cronológicas, evitando divisão aleatória entre
treinamento e validação.

A seleção das séries é refeita com informações
disponíveis em cada corte temporal.

As métricas principais são MAE e WAPE.

WAPE = soma dos erros absolutos /
soma das vendas reais absolutas × 100

Quanto menor o MAE e o WAPE, melhor o desempenho.

## 5. Resultados de desenvolvimento

Foram avaliadas 840 previsões em quatro janelas.


| Modelo                   |     MAE |   WAPE |
| ------------------------ | ------: | -----: |
| Baseline de 7 dias       | 14,0357 | 27,52% |
| Random Forest original   | 11,2586 | 22,07% |
| Random Forest aprimorada | 10,9631 | 21,50% |

A engenharia de atributos adicional reduziu o WAPE
em aproximadamente 2,63% relativamente à Random
Forest original durante o desenvolvimento.

A versão aprimorada foi selecionada como candidata
para o teste independente.

## 6. Teste independente

O teste foi realizado no período de 25/04/2016
a 01/05/2016, com 210 previsões.


| Modelo                   |     MAE |   WAPE |
| ------------------------ | ------: | -----: |
| Baseline de 7 dias       | 12,4735 | 24,39% |
| Random Forest original   | 10,3016 | 20,15% |
| Random Forest aprimorada | 10,5947 | 20,72% |

### Interpretação

Os dois modelos de Machine Learning superaram
o baseline no período de teste.

Entretanto, a Random Forest aprimorada não
confirmou a vantagem observada no backtesting.

A Random Forest original obteve o menor erro
nesse período independente.

A comparação demonstra a importância de avaliar
modelos em dados não utilizados para selecionar
atributos e ajustar decisões de modelagem.

Não é adequado afirmar que a versão aprimorada
é superior apenas com base no desenvolvimento.

Também não é adequado escolher retroativamente
um novo modelo usando repetidamente o mesmo
conjunto de teste.

## 7. Testes automatizados

O projeto possui testes automatizados com Pytest,
incluindo verificações de:

- Quantidade e unicidade das previsões.
- Integridade das janelas temporais.
- Separação entre treinamento e teste.
- Consistência dos experimentos.
- Valores ausentes ou inválidos.
- Recálculo das métricas.
- Existência dos relatórios técnicos.

Resultado da execução local:

**11 testes aprovados, nenhuma falha.**

## 8. Estrutura do projeto

```text
retail-demand-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── decisao_modelo_aula15.md
│   └── avaliacao_final.md
├── src/
│   ├── preparacao_amostra_modelagem.py
│   ├── avaliacao_baselines.py
│   ├── backtesting_temporal.py
│   ├── modelo_ml_inicial.py
│   ├── backtesting_ml.py
│   ├── diagnostico_modelo.py
│   ├── experimento_atributos.py
│   ├── selecao_modelo.py
│   └── avaliacao_final.py
├── tests/
│   └── test_integridade_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

Outros scripts exploratórios das primeiras etapas
também podem estar presentes em `src/`.

## 9. Como executar

### Pré-requisitos

- Python 3.13 ou versão compatível.
- Ambiente virtual Python.
- Dependências listadas em requirements.txt.
- Arquivos do M5 Forecasting.

### Criar ambiente virtual no Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Preparar os dados

Baixe no Kaggle os arquivos:

- sales_train_validation.csv
- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

Coloque-os em `data/raw/`.

### Executar as etapas principais

Execute na ordem:

```powershell
python src/preparacao_amostra_modelagem.py
python src/avaliacao_baselines.py
python src/backtesting_temporal.py
python src/modelo_ml_inicial.py
python src/backtesting_ml.py
python src/diagnostico_modelo.py
python src/experimento_atributos.py
python src/selecao_modelo.py
python src/avaliacao_final.py
```

### Executar os testes

```powershell
python -m pytest tests/test_integridade_pipeline.py -v
```

Os testes de integridade dependem dos CSVs gerados
pelas etapas anteriores.

## 10. Limitações

- Foram utilizadas apenas 30 séries de alto volume.
- O teste independente abrange somente sete dias.
- Os resultados não representam necessariamente
  o desempenho em produtos de baixo volume.
- Não foram realizados testes operacionais em produção.
- Não foram incorporados todos os possíveis fatores
  comerciais, promocionais e de abastecimento.
- A estabilidade em períodos futuros ainda precisa
  ser investigada.
- A avaliação final não deve ser reutilizada
  indefinidamente para selecionar novos modelos.

## 11. Possíveis melhorias futuras

- Avaliação em mais séries e categorias de produtos.
- Incorporação de preços e eventos do calendário.
- Comparação com outros algoritmos.
- Avaliação de previsões probabilísticas.
- Monitoramento de deriva e desempenho.
- Construção de pipeline de inferência.

## 12. Tecnologias

Python, Pandas, NumPy, Scikit-learn, Pytest,
Git e GitHub.

## 13. Considerações finais

O projeto demonstra um fluxo de trabalho de
Ciência de Dados que parte de um problema de
negócio e avança por preparação, modelagem,
experimentação, backtesting, teste independente
e validação automatizada.

Um dos principais aprendizados foi verificar
que uma melhoria observada no desenvolvimento
nem sempre se mantém em dados futuros.

Essa conclusão foi preservada na documentação
como parte da análise técnica.

## 14. Dashboard gerencial de estoque

O projeto inclui uma aplicação Streamlit para demonstrar
como previsões de demanda podem apoiar decisões de estoque.

### Funcionalidades

- Indicadores de demanda prevista para sete dias.
- Estoque disponível e pedidos em trânsito simulados.
- Identificação de produtos com risco de ruptura.
- Classificação de prioridades de reposição.
- Quantidade sugerida para reposição por produto e loja.
- Gráficos de demanda diária e saldo projetado.
- Filtros por loja, produto e prioridade.
- Exportação do plano de reposição para CSV.

### Exemplo de resultado

Na simulação histórica de 25/04/2016 a 01/05/2016:


| Indicador                    |          Resultado |
| ---------------------------- | -----------------: |
| Demanda prevista             | 10.739,69 unidades |
| Estoque simulado             |     6.515 unidades |
| Reposição sugerida         |     4.279 unidades |
| Séries com risco de ruptura |                 21 |
| Prioridades críticas        |                  8 |

Esses números representam uma demonstração histórica
com dados operacionais simulados, não uma recomendação
para compras reais.

### Como executar o dashboard

Depois de criar o ambiente virtual, instalar as
dependências e obter os arquivos originais do M5,
execute as etapas de preparação e avaliação descritas
anteriormente neste README.

Em seguida, gere o planejamento de estoque:

```powershell
python src/planejamento_reposicao.py
```

Inicie a aplicação:

```powershell
python -m streamlit run app.py
```

Acesse o endereço local informado pelo Streamlit,
normalmente http://localhost:8501.

### Executar os testes

```powershell
python -m pytest tests/ -v
```

Resultado da última execução local:

**20 testes aprovados, nenhuma falha.**

### Limitações operacionais

- Os saldos de estoque são simulados.
- Os pedidos em trânsito e prazos de entrega são simulados.
- O estoque de segurança utiliza uma heurística.
- O dashboard utiliza previsões históricas previamente
  geradas, não previsões atualizadas automaticamente.
- Os CSVs são gerados localmente e não são publicados
  no GitHub.
- A aplicação ainda não possui integração com ERP,
  banco de dados operacional ou fornecedores.
- Não deve ser utilizada para compras reais.

### Próximas evoluções

- Integração com dados reais de estoque.
- Geração automatizada de previsões.
- Regras de reposição com lead time e lote mínimo.
- Monitoramento do desempenho do modelo.
- Implantação da aplicação em ambiente apropriado.

---
