# trAIn Health

> **Clinical ML Studio** - Plataforma profissional de experimentação em Machine Learning para saúde com foco em reprodutibilidade, rigor científico e governança.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Sobre o Projeto

**trAIn Health** é uma aplicação desktop completa e profissional para experimentação com aprendizado de máquina em contextos clínicos e de saúde. O sistema combina:

- 🎨 **Interface moderna** com design healthcare-inspired e temas claro/escuro
- 🔬 **Rigor científico** com literatura completa para cada algoritmo
- 📊 **Pipeline completo** desde carregamento até avaliação e relatórios
- ⚙️ **Reprodutibilidade total** com controle de seeds e versionamento de experimentos
- 📚 **Governança** com histórico, exportação e documentação detalhada

### Diferenciais

- **Literatura Científica Integrada**: Cada modelo possui documentação completa com fundamentação matemática, estudos clínicos, hiperparâmetros explicados e boas práticas
- **Design Healthcare Premium**: Paleta de cores profissional inspirada em saúde (teal/green accents)
- **Configuração Centralizada**: Arquivo `config.py` com todas as constantes da aplicação
- **Código Profissional**: Google-style docstrings, type hints completos, logging estruturado, tratamento robusto de erros
- **Arquitetura Modular**: Estrutura `src/` limpa com separação clara de responsabilidades (core, models, ui, utils)
- **Testes Automatizados**: Suite de validação do sistema e testes de conteúdo de literatura

## 🚀 Instalação

### Pré-requisitos

- **Python 3.9+** (testado até 3.14)
- pip e virtualenv
- Windows, Linux ou macOS

### Instalação Rápida

1. **Clone o repositório**:
```bash
git clone <repository-url>
cd trAIn
```

2. **Crie um ambiente virtual**:
```bash
python -m venv .venv
```

3. **Ative o ambiente virtual**:
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

4. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

5. **Execute a aplicação**:
```bash
python main.py
```

## 🎯 Funcionalidades

### Modelos de Machine Learning

#### Classificação (8 modelos)
- **Logistic Regression** - Baseline linear probabilístico
- **K-Nearest Neighbors (KNN)** - Aprendizado baseado em similaridade
- **Naive Bayes** - Classificador probabilístico bayesiano
- **Support Vector Machine (SVM)** - Separação por hiperplanos ótimos
- **Decision Tree** - Aprendizado baseado em regras interpretáveis
- **Random Forest** - Ensemble de árvores de decisão
- **Gradient Boosting** - Boosting sequencial de weak learners
- **XGBoost** - Gradient boosting otimizado e regularizado

#### Regressão (7 modelos)
- **Linear Regression** - Regressão linear por mínimos quadrados (OLS)
- **Ridge Regression** - Regressão linear com regularização L2
- **Support Vector Regression (SVR)** - Regressão com margens epsilon
- **Decision Tree Regressor** - Árvore de regressão CART
- **Random Forest Regressor** - Ensemble de árvores de regressão
- **Gradient Boosting Regressor** - Boosting para problemas de regressão
- **XGBoost Regressor** - XGBoost para tarefas de regressão

### Pré-processamento

#### Normalização (Scaling)
- **StandardScaler** - Padronização (média 0, desvio 1)
- **MinMaxScaler** - Normalização min-max [0, 1]
- **RobustScaler** - Escala robusta a outliers (usa mediana e IQR)
- **None** - Sem normalização

#### Balanceamento de Classes (Apenas Classificação)
- **RandomOverSampler** - Sobreamostragem aleatória da classe minoritária
- **RandomUnderSampler** - Sub-amostragem aleatória da classe majoritária  
- **SMOTE** - Synthetic Minority Over-sampling Technique
- **None** - Sem balanceamento

### Métricas de Avaliação

#### Classificação
- **Acurácia**: Accuracy, Balanced Accuracy
- **Discriminação**: Precision, Recall, F1-Score (Macro e Weighted)
- **Especificidade**: True Negative Rate (TNR)
- **Correlação**: Matthews Correlation Coefficient (MCC)
- **Curvas ROC/PR**: ROC AUC, PR AUC (problemas binários)
- **Probabilísticas**: Log Loss
- **Matriz de Confusão**: Visualização completa de erros

#### Regressão
- **Erros Absolutos**: MAE, MedAE, MAPE
- **Erros Quadráticos**: MSE, RMSE
- **Variância Explicada**: R² Score, Explained Variance
- **Visualizações**: Scatter de predições, análise de resíduos

### Formatos de Dados

- ✅ **CSV** (Comma-Separated Values)
- ✅ **Parquet** (Apache Parquet para grandes volumes)

### Tipo de Problema (Detecção Automática)

O sistema identifica automaticamente se é **Classificação** ou **Regressão** baseado na variável target:
- **Classificação**: Target categórico ou numérico com < 20 valores únicos
- **Regressão**: Target numérico contínuo com >= 20 valores únicos

## 📖 Como Usar

### Workflow Completo

1. **Carregar Dados**
   - Clique em "Carregar Arquivo"
   - Selecione CSV ou Parquet
   - Sistema mostra preview automático

2. **Selecionar Variável Target**
   - Escolha a coluna alvo no dropdown
   - Sistema detecta automaticamente o tipo de problema
   - Modelos disponíveis são filtrados

3. **Configurar Parâmetros Metodológicos**
   - **Proporção Teste**: % de dados reservados para validação (padrão: 20%)
   - **Seed Fixa**: Semente aleatória para reprodutibilidade (padrão: 42)
   - **Scaler**: Técnica de normalização de features
   - **Sampler**: Técnica de balanceamento (apenas classificação)

4. **Selecionar e Configurar Modelos**
   - Escolha até **5 modelos** simultaneamente
   - Cada modelo pode ter múltiplas configurações de hiperparâmetros
   - Interface dinâmica mostra apenas parâmetros relevantes
   - Consulte a **literatura integrada** para entender cada hiperparâmetro

5. **Executar Experimento**
   - Clique em "Executar Experimento"
   - Sistema treina todos os modelos em thread separada (UI responsiva)
   - Progresso é exibido em tempo real

6. **Analisar Resultados**
   - **Aba de Métricas**: Tabela comparativa de todas as execuções
   - **Aba de Gráficos**: Confusion Matrix, ROC Curve, PR Curve, Residuals
   - **Aba de Relatório**: Documento TXT completo com todas as informações
   - **Aba de Histórico**: Todos os experimentos passados com filtros

7. **Exportar e Compartilhar**
   - Download de pipelines treinados (.joblib)
   - Exportação de relatórios (.txt)
   - Exportação de histórico completo (.json)

## 🏗️ Arquitetura do Projeto

```
trAIn/
│
├── config.py                   # Configuração centralizada (cores, paths, constantes)
├── main.py                     # Ponto de entrada com logging configurado
├── main_gui.py                 # Interface PyQt6 e lógica de UI
├── requirements.txt            # Dependências Python
├── README.md                   # Documentação completa (este arquivo)
├── history.json                # Histórico de experimentos
├── .gitignore                  # Exclusões do Git
│
├── src/                        # Código fonte modular
│   ├── core/                   # Lógica de negócio central
│   │   ├── data_handler.py     # Carregamento, split e detecção de problema
│   │   └── pipeline_builder.py # Construção de pipelines sklearn/imblearn
│   │
│   ├── models/                 # Implementações de modelos ML
│   │   ├── specs.py            # TypedDicts para especificações
│   │   ├── registry.py         # Registro central de modelos
│   │   ├── logistic_regression.py
│   │   ├── knn_classifier.py
│   │   ├── naive_bayes.py
│   │   ├── svm.py
│   │   ├── decision_tree_classifier.py
│   │   ├── random_forest_classifier.py
│   │   ├── gradient_boosting_classifier.py
│   │   ├── xgboost_classifier.py
│   │   ├── linear_regression.py
│   │   ├── ridge_regression.py
│   │   ├── svr_regressor.py
│   │   ├── decision_tree_regressor.py
│   │   ├── random_forest_regressor.py
│   │   ├── gradient_boosting_regressor.py
│   │   └── xgboost_regressor.py
│   │
│   ├── ui/                     # Componentes de interface
│   │   └── literature.py       # Carregamento e theming de literatura HTML
│   │
│   └── utils/                  # Utilidades
│       ├── evaluator.py        # Treinamento, avaliação e geração de gráficos
│       └── reporter.py         # Geração de relatórios formatados
│
├── literature/                 # Documentação científica dos modelos (HTML)
│   ├── logistic_regression/
│   ├── knn/
│   ├── naive_bayes/
│   ├── svm/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── gradient_boosting/
│   ├── xgboost/
│   ├── linear_regression/
│   ├── ridge_regression/
│   ├── svr/
│   ├── decision_tree_regressor/
│   ├── random_forest_regressor/
│   ├── gradient_boosting_regressor/
│   └── xgboost_regressor/
│
├── sample_data/                # Dados de exemplo para testes
│   ├── heart_statlog_cleveland_hungary_final.csv
│   └── insurance_encoded.csv
│
├── tests/                      # Testes automatizados
│   ├── validate_system.py      # Validação completa do sistema
│   ├── test_linear_regression_literature.py
│   ├── test_xgboost_literature.py
│   └── README.md               # Documentação dos testes
│
└── docs/                       # Documentação adicional
    └── REFACTORING_SUMMARY.md  # Resumo da refatoração profissional
```

### Princípios de Design

- **SOLID**: Single Responsibility, Open/Closed, Dependency Inversion
- **DRY**: Don't Repeat Yourself - configuração centralizada
- **Separation of Concerns**: Core, UI, Models, Utils isolados
- **Type Safety**: Type hints completos em todo o código
- **Documentation**: Google-style docstrings para todas as funções públicas
- **Error Handling**: Try/except com mensagens específicas e logging
- **PEP 8**: Código seguindo padrões Python

## 🔬 Literatura Científica Integrada

Cada modelo possui documentação HTML completa com:

✅ **Fundamentação Matemática**: Equações, otimizações e derivações  
✅ **Quando Usar / Quando Evitar**: Orientações práticas baseadas em evidências  
✅ **Hiperparâmetros Explicados**: O que cada parâmetro faz e como ajustar  
✅ **Estudos Clínicos**: Referências a aplicações em saúde (Framingham, MIMIC-III, APACHE)  
✅ **Mitos e Boas Práticas**: Desmistificação de conceitos comuns  
✅ **Pipeline Clínico**: Checklist para deploy em produção médica  
✅ **Análises Avançadas**: Ablation studies, fairness, interpretabilidade  

## 🧪 Testes Automatizados

### Suite de Testes

Execute a validação completa do sistema:

```bash
python tests/validate_system.py
```

**Testes incluídos**:
1. ✅ Importação de todos os módulos
2. ✅ Criação de dados sintéticos
3. ✅ Detecção automática de tipo de problema
4. ✅ Split de dados com estratificação
5. ✅ Construção de pipelines
6. ✅ Treinamento e avaliação de modelos

### Testes de Literatura

Validam qualidade do conteúdo científico:

```bash
python tests/test_linear_regression_literature.py
python tests/test_xgboost_literature.py
```

Consulte [tests/README.md](tests/README.md) para mais detalhes.

## ⚙️ Configuração

Todas as constantes da aplicação estão centralizadas em `config.py`:

- **Metadados**: Nome, versão, descrição
- **Paths**: Literatura, histórico, dados
- **Defaults**: Test size, random state, scaler, sampler
- **UI Settings**: Dimensões de janela, sidebar
- **Temas**: Cores completas para dark/light mode
- **Options**: Scalers e samplers disponíveis

## 🎨 Temas

### Tema Escuro (Dark Mode)
- Background: `#0f1720` (deep blue-gray)
- Surface: `#14212b` (card background)
- Accent: `#49b9a6` (teal green)
- Text: `#e7f2ef` (light mint)

### Tema Claro (Light Mode)
- Background: `#f4f8f7` (very light mint)
- Surface: `#ffffff` (white cards)
- Accent: `#2f8f83` (darker teal)
- Text: `#1f2d2a` (dark gray)

Paleta inspirada em ambientes de saúde: limpo, confiável, profissional.

## 📊 Exemplo de Relatório

```
=====================================================
 RELATORIO DE EXPERIMENTO DE MACHINE LEARNING (TCC) 
=====================================================
Timestamp da Execução: 2026-02-20 14:35:22
Seed (Semente) Fixa: 42
-----------------------------------------------------

[1] INFORMACOES DO DATASET
  Nome do Arquivo: heart_disease.csv
  Variável Target: target
  Tipo de Problema: Classification
  Total de Amostras: 918
  Total de Features: 11
  Distribuicao Classes (Treino): {0: 368, 1: 366}
  Distribuicao Classes (Teste): {0: 92, 1: 92}

[2] PARAMETROS E TECNICAS ESCOLHIDAS
  Proporção Treino/Teste: 80% / 20%
  Técnica de Normalização (Scaler): StandardScaler
  Técnica de Balanceamento (Sampler): SMOTE (Aplicado apenas no Treino)
  Algoritmo de Treinamento: Random Forest #1
  Parametros do Modelo:
    n_estimators: 100
    max_depth: None
    ...

[3] AVALIACAO DE DESEMPENHO (Conjunto de Teste Intocado)
  Metricas de Classificacao:
    Accuracy: 0.8750
    Balanced Accuracy: 0.8723
    Specificity (TNR): 0.8913
    ...
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Siga o fluxo:

1. **Fork** o projeto
2. Crie uma **branch** para sua feature:
   ```bash
   git checkout -b feature/MinhaFeature
   ```
3. **Commit** com mensagens descritivas:
   ```bash
   git commit -m "feat: adiciona suporte a LSTM para séries temporais"
   ```
4. **Push** para sua branch:
   ```bash
   git push origin feature/MinhaFeature
   ```
5. Abra um **Pull Request** com descrição detalhada

### Padrões de Código

- Siga **PEP 8**
- Adicione **type hints** em funções públicas
- Escreva **docstrings** no estilo Google
- Mantenha **testes** para novas funcionalidades
- Use **logging** ao invés de print()

## 📝 Licença

Este projeto está sob a licença **MIT**. Veja `LICENSE` para detalhes.

## 👥 Autores

Desenvolvido com foco em **qualidade**, **rigor científico** e **usabilidade** para aplicações em saúde.

## 🙏 Agradecimentos

- **[Scikit-learn](https://scikit-learn.org/)** - Biblioteca robusta de ML
- **[Imbalanced-learn](https://imbalanced-learn.org/)** - Técnicas de balanceamento
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** - Framework GUI moderno
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting otimizado
- **[Matplotlib](https://matplotlib.org/)** & **[Seaborn](https://seaborn.pydata.org/)** - Visualizações científicas
- **Comunidade científica** pelos estudos e referências em saúde

## 📚 Referências Científicas

A literatura integrada cita dezenas de estudos, incluindo:

- **Framingham Heart Study** - Predição de risco cardiovascular
- **MIMIC-III** - Banco de dados de cuidados intensivos
- **APACHE II/III** - Scores de gravidade em UTI
- **eICU Collaborative Research Database** - Dados multicêntricos de UTI
- Papers seminais de cada algoritmo (Breiman, Friedman, Chen & Guestrin, etc.)

---

<div align="center">

**trAIn Health** - Onde **treino** e **inteligência artificial** encontram a **saúde**.

🩺 💚 🤖

*Desenvolvido para TCC e aplicações profissionais em Machine Learning Clínico*

</div>
