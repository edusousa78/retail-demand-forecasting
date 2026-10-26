# Retail Demand Forecasting

## Previsão de demanda e gestão de estoque com Machine Learning

Projeto de Ciência de Dados que utiliza previsões de demanda para apoiar o planejamento de reposição de estoque no varejo.

O estudo combina modelagem preditiva, validação temporal, testes automatizados e um dashboard gerencial interativo desenvolvido com Streamlit.

**Status:** projeto de portfólio concluído, com dashboard de demonstração publicado. Não é um sistema operacional em produção.

### Acessar o projeto

- **Dashboard interativo:** https://novavarejo-estoque.streamlit.app/
- **Código-fonte:** https://github.com/edusousa78/retail-demand-forecasting

---

## 1. Contexto de negócio

A NovaVarejo Brasil é uma empresa fictícia criada para contextualizar o estudo de caso.

O problema de negócio envolve rupturas de estoque e excesso de mercadorias, que podem prejudicar vendas, custos e operações.

O projeto busca responder a três perguntas:

1. Qual será a demanda estimada por produto e loja nos próximos sete dias?
2. Quais combinações de produto e loja apresentam risco de ruptura?
3. Quais quantidades de reposição podem ser consideradas no planejamento?

As respostas são apresentadas em um dashboard interativo, permitindo explorar diferentes lojas, produtos e níveis de urgência.

## 2. Objetivos

### Objetivo empresarial

Apoiar o planejamento de estoque e as decisões de reposição.

### Objetivo analítico

Prever a demanda diária por produto e loja, com horizonte de sete dias, utilizando técnicas de Machine Learning.

### Entregáveis

- Pipeline de preparação de dados.
- Engenharia de atributos temporais.
- Modelos de previsão e baseline.
- Backtesting com validação temporal.
- Avaliação em teste independente.
- Planejamento demonstrativo de reposição.
- Dashboard interativo.
- Exportação de relatórios CSV.
- Testes automatizados.
- Documentação técnica e gerencial.

## 3. Fonte dos dados

O projeto utiliza dados da competição **M5 Forecasting - Accuracy**, disponibilizados no Kaggle.

Fonte:

https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

O conjunto original contém 30.490 séries temporais.

Para manter o experimento computacionalmente acessível, foram selecionadas 30 séries de maior volume de vendas, respeitando a disponibilidade histórica dos dados em cada etapa.

### Períodos utilizados


| Etapa                   | Período                    |
| ----------------------- | --------------------------- |
| Histórico de modelagem | 28/10/2015 a 24/04/2016     |
| Backtesting             | Quatro janelas de sete dias |
| Teste independente      | 25/04/2016 a 01/05/2016     |
| Horizonte de previsão  | Sete dias                   |
| Granularidade           | Produto, loja e dia         |

Os arquivos brutos do M5 não são distribuídos neste repositório.

A pasta `data/demo/` contém apenas os resultados necessários para executar a demonstração pública do dashboard.

## 4. Metodologia

### 4.1 Preparação dos dados

Os dados de vendas foram transformados do formato largo para o formato longo, permitindo analisar cada observação por produto, loja e data.

### 4.2 Engenharia de atributos

Foram utilizados atributos como:

- Vendas históricas.
- Defasagens temporais (*lags*).
- Médias móveis.
- Desvio-padrão móvel.
- Tendência entre diferentes períodos.
- Dia da semana.
- Mês.
- Indicador de fim de semana.
- Horizonte de previsão.
- Identificação da série temporal.

Os atributos foram construídos com informações disponíveis até a origem de cada previsão, evitando o uso indevido de dados futuros.

### 4.3 Modelagem

Foram comparados três métodos:

1. Baseline baseado na média dos últimos sete dias.
2. Random Forest com atributos temporais básicos.
3. Random Forest com engenharia de atributos adicional.

O modelo de Machine Learning utiliza `RandomForestRegressor`, do Scikit-learn.

Configuração principal:

- 100 árvores.
- Profundidade máxima de 12.
- Mínimo de três amostras por folha.
- `random_state=42`.

Foi utilizada uma estratégia de previsão direta para os sete horizontes.

### 4.4 Validação temporal

O projeto utiliza backtesting com quatro janelas cronológicas.

Essa abordagem evita a divisão aleatória dos registros e respeita a ordem temporal das vendas.

As métricas principais são:

- **MAE:** erro absoluto médio.
- **WAPE:** erro absoluto ponderado pelo volume observado.

Quanto menores os valores dessas métricas, melhor o desempenho preditivo.

## 5. Resultados dos modelos

### 5.1 Backtesting de desenvolvimento

Foram avaliadas 840 previsões distribuídas em quatro janelas.


| Modelo                   |     MAE |   WAPE |
| ------------------------ | ------: | -----: |
| Baseline de sete dias    | 14,0357 | 27,52% |
| Random Forest original   | 11,2586 | 22,07% |
| Random Forest aprimorada | 10,9631 | 21,50% |

A Random Forest aprimorada apresentou o menor erro durante o desenvolvimento e foi selecionada como candidata para o teste independente.

### 5.2 Teste independente

O teste foi realizado de 25/04/2016 a 01/05/2016, com 210 previsões.


| Modelo                   |     MAE |   WAPE |
| ------------------------ | ------: | -----: |
| Baseline de sete dias    | 12,4735 | 24,39% |
| Random Forest original   | 10,3016 | 20,15% |
| Random Forest aprimorada | 10,5947 | 20,72% |

### Interpretação

Os dois modelos de Machine Learning superaram o baseline no teste independente.

Entretanto, a Random Forest aprimorada não confirmou sua vantagem observada durante o desenvolvimento.

A Random Forest original apresentou o menor WAPE no período independente.

Esse resultado demonstra a importância de preservar um conjunto de teste não utilizado na seleção do modelo.

O dashboard utiliza as previsões da versão aprimorada, selecionada antes da avaliação independente. A aplicação representa uma demonstração da solução gerencial, e não uma comprovação de superioridade desse modelo.

## 6. Planejamento de reposição

As previsões foram utilizadas para construir um planejamento demonstrativo de estoque.

O módulo considera:

- Demanda prevista para sete dias.
- Estoque inicial simulado.
- Pedidos em trânsito simulados.
- Prazo de entrega simulado.
- Estoque de segurança estimado.
- Saldo projetado.
- Risco de ruptura.
- Quantidade sugerida de reposição.

### Regra de reposição

A quantidade sugerida é calculada a partir da demanda prevista, do estoque de segurança, do estoque disponível e dos pedidos em trânsito considerados no horizonte.

O resultado é limitado a valores não negativos e arredondado para cima em unidades inteiras.

### Classificação de urgência


| Prioridade | Interpretação                                                         |
| ---------- | ----------------------------------------------------------------------- |
| Crítica   | Ruptura projetada antes ou até o prazo de entrega                      |
| Alta       | Ruptura posterior ao prazo de entrega, dentro do horizonte              |
| Média     | Necessidade de atenção ao ponto de reposição, sem ruptura projetada |
| Baixa      | Sem necessidade imediata identificada pela regra                        |

A classificação e as quantidades são demonstrativas. Não substituem políticas reais de compras ou abastecimento.

## 7. Dashboard gerencial

**Aplicação pública:**

https://novavarejo-estoque.streamlit.app/

O dashboard foi desenvolvido com Streamlit e apresenta uma interface de apoio à decisão.

### Funcionalidades

- Indicadores de demanda prevista, estoque e reposição.
- Percentual de combinações produto-loja com risco de ruptura.
- Classificação de urgência.
- Filtros automáticos por loja, produto e prioridade.
- Nomes amigáveis para lojas e produtos.
- Gráfico de demanda diária.
- Gráfico de saldo projetado.
- Reposição sugerida por loja.
- Tabela detalhada de planejamento.
- Exportação de relatório CSV.
- Informações sobre metodologia e limitações.

Os filtros começam sem restrições, exibindo todos os registros. Quando uma opção é selecionada, os indicadores e gráficos são atualizados automaticamente.

### Indicadores da demonstração

Período histórico: 25/04/2016 a 01/05/2016.


| Indicador                             |          Resultado |
| ------------------------------------- | -----------------: |
| Demanda prevista                      | 10.739,69 unidades |
| Estoque inicial simulado              |     6.515 unidades |
| Reposição sugerida                  |     4.279 unidades |
| Combinações produto-loja analisadas |                 30 |
| Combinações com risco de ruptura    |                 21 |
| Percentual em risco                   |              70,0% |
| Prioridades críticas                 |                  8 |
| Prioridades altas                     |                 13 |

**Importante:** os estoques e parâmetros operacionais são simulados. Os resultados não devem ser utilizados para compras reais.

## 8. Arquitetura do projeto

```text
retail-demand-forecasting/
|
|-- app.py
|-- data/
|   |-- demo/
|   |   |-- planejamento_reposicao.csv
|   |   |-- planejamento_demanda_diaria.csv
|   |-- raw/
|   |-- processed/
|
|-- docs/
|   |-- 01_visao_negocio.md
|   |-- 02_regras_negocio.md
|
|-- reports/
|   |-- avaliacao_final.md
|   |-- decisao_modelo_aula15.md
|   |-- planejamento_estoque.md
|
|-- src/
|   |-- auditoria_dados.py
|   |-- preparacao_dados.py
|   |-- engenharia_atributos.py
|   |-- divisao_temporal.py
|   |-- preparacao_amostra_modelagem.py
|   |-- avaliacao_baselines.py
|   |-- backtesting_temporal.py
|   |-- modelo_ml_inicial.py
|   |-- backtesting_ml.py
|   |-- diagnostico_modelo.py
|   |-- experimento_atributos.py
|   |-- selecao_modelo.py
|   |-- avaliacao_final.py
|   |-- planejamento_reposicao.py
|
|-- tests/
|   |-- test_integridade_pipeline.py
|   |-- test_planejamento_estoque.py
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

A estrutura acima destaca os principais componentes. Outros scripts auxiliares podem estar presentes no repositório.

As pastas `data/raw/` e `data/processed/` são utilizadas localmente e não precisam conter dados para executar o dashboard em modo de demonstração.

## 9. Como executar

### 9.1 Pré-requisitos

- Python 3.13 ou versão compatível.
- Git.
- Ambiente virtual Python.
- Dependências de `requirements.txt`.

### 9.2 Clonar o repositório

```powershell
git clone https://github.com/edusousa78/retail-demand-forecasting.git
cd retail-demand-forecasting
```

### 9.3 Criar o ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 9.4 Executar o dashboard de demonstração

Os arquivos em `data/demo/` permitem executar a aplicação sem baixar os dados brutos do Kaggle.

```powershell
python -m streamlit run app.py
```

Abra o endereço informado pelo Streamlit, normalmente:

http://localhost:8501

A aplicação prioriza os resultados em `data/processed/` quando ambos os arquivos necessários estão disponíveis. Caso contrário, utiliza os dados de `data/demo/`.

### 9.5 Reproduzir a análise completa

Para reproduzir o pipeline de modelagem, baixe os dados originais do M5 no Kaggle.

Arquivos utilizados:

- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Coloque os arquivos em `data/raw/`.

Execute as principais etapas na ordem:

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
python src/planejamento_reposicao.py
```

Essas etapas geram os resultados necessários para a análise e para o planejamento demonstrativo de reposição.

## 10. Testes automatizados

O projeto utiliza Pytest para verificar a integridade do pipeline e do planejamento de estoque.

Os testes incluem:

- Quantidade e unicidade das previsões.
- Respeito à cronologia.
- Integridade dos experimentos.
- Consistência das métricas.
- Existência dos relatórios.
- Quantidade de séries e previsões do planejamento.
- Validade das demandas previstas.
- Consistência do saldo projetado.
- Classificação de risco.
- Cálculo das quantidades de reposição.
- Consistência dos indicadores gerenciais.

### Executar

```powershell
python -m pytest tests/ -q
```

**Último resultado informado: 20 testes aprovados, nenhuma falha.**

Os testes dependem dos arquivos de saída gerados pelo pipeline.

## 11. Tecnologias

- Python.
- Pandas.
- NumPy.
- Scikit-learn.
- Altair.
- Streamlit.
- Pytest.
- Git.
- GitHub.
- Streamlit Community Cloud.

## 12. Implantação

O dashboard está hospedado no Streamlit Community Cloud.

- **Repositório:** `edusousa78/retail-demand-forecasting`
- **Branch:** `main`
- **Arquivo principal:** `app.py`
- **Python configurado:** 3.13

A aplicação utiliza os arquivos versionados em `data/demo/` para permitir acesso público sem depender dos dados brutos do M5.

O deploy disponibiliza a interface interativa, mas não executa automaticamente o treinamento dos modelos ou a atualização das previsões.

## 13. Limitações

Este é um projeto demonstrativo, não uma ferramenta pronta para operação empresarial.

Principais limitações:

- Uso de apenas 30 séries temporais.
- Teste independente limitado a sete dias.
- Estoques e pedidos em trânsito simulados.
- Prazos de entrega simulados.
- Estoque de segurança baseado em heurística.
- Ausência de integração com ERP ou fornecedores.
- Ausência de atualização automática das previsões.
- Não considera todos os custos, restrições logísticas e lotes mínimos.
- Uma reposição sugerida pode chegar tarde demais para evitar ruptura.
- Os resultados históricos não representam a situação atual de nenhuma loja.

## 14. Evoluções futuras

- Integração com dados reais de estoque.
- Atualização automática das previsões.
- Ampliação para mais produtos e lojas.
- Previsões probabilísticas e intervalos de incerteza.
- Regras de reposição com custos e restrições logísticas.
- Monitoramento da qualidade das previsões.
- Integração com sistemas empresariais.
- Autenticação e perfis de acesso.
- Pipeline automatizado de treinamento e inferência.

## 15. Considerações finais

O projeto demonstra um fluxo de Ciência de Dados que começa com um problema de negócio e avança por preparação dos dados, modelagem, validação temporal, teste independente e entrega de uma aplicação interativa.

Além de comparar modelos, o trabalho transforma previsões de demanda em informações gerenciais para apoiar o planejamento de reposição.

O principal aprendizado técnico foi que uma melhoria observada no desenvolvimento nem sempre se mantém em dados independentes.

O principal aprendizado de negócio foi que um modelo preditivo precisa ser apresentado de forma compreensível e útil para apoiar decisões.

---

**NovaVarejo Brasil — Projeto fictício de Ciência de Dados e Gestão de Estoque.**
