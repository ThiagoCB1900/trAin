# Estrutura Final do Artigo - Mapa Visual

## Página 1: Título e Resumos

```
┌─────────────────────────────────────────────────────────────┐
│ TÍTULO                                                      │
│ trAIn Health: Uma Plataforma para Experimentação Reprodutível
│ em Machine Learning Clínico                                 │
├─────────────────────────────────────────────────────────────┤
│ Autor: Thiago Guilherme Bezerra de Aguiar                   │
│ Email: thiago.aguiar@aluno.uece.br                          │
│ UECE - Universidade Estadual do Ceará                       │
├─────────────────────────────────────────────────────────────┤
│ RESUMO (Português)                                          │
│ - Problema: Escolha de modelos ML em saúde requer rigor     │
│ - Solução: Plataforma com 15 algoritmos + 127 refs          │
│ - Resultado: Redução de horas para minutos em experimentos  │
├─────────────────────────────────────────────────────────────┤
│ ABSTRACT (English)                                          │
│ [Same structure in English]                                 │
└─────────────────────────────────────────────────────────────┘
```

## Página 2-3: Corpo Principal

```
┌─────────────────────────────────────────────────────────────┐
│ SEÇÃO 1: INTRODUÇÃO (~300 palavras)                        │
│                                                             │
│ Parágrafo 1: Contexto histórico (Framingham, MIMIC-III)     │
│ Parágrafo 2: Desafios práticos (barreira técnica, repro)   │
│ Parágrafo 3: Abordagem proposta (integração de literatura) │
│ Parágrafo 4: Contribuições (arquitetura, validação)        │
├─────────────────────────────────────────────────────────────┤
│ SEÇÃO 2: DESIGN DA PLATAFORMA (~700 palavras)             │
│                                                             │
│ Parágrafo 1: Workflow em 7 etapas (fluido)                 │
│ Parágrafo 2: Arquitetura técnica (modules, patterns)       │
│ Parágrafo 3: Stack (scikit-learn, PyQt6, XGBoost)          │
│ Parágrafo 4: Diferencial - Literatura integrada            │
│   - 127 referências bibliográficas                         │
│   - ~17.000 linhas de documentação                         │
│   - Fundamentação + boas práticas + mitos                  │
│ Parágrafo 5: Reprodutibilidade (seeds, versionamento)      │
└─────────────────────────────────────────────────────────────┘
```

## Página 3-4: Avaliação

```
┌─────────────────────────────────────────────────────────────┐
│ SEÇÃO 3: AVALIAÇÃO EXPERIMENTAL (~400 palavras)           │
│                                                             │
│ Parágrafo introdutório: 10 bases, configuração padrão       │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Tabela 1: Classificação (5 bases, 8 modelos)       │    │
│ │ Heart: [s] [s] [s] [s] [s] [s] [s] [s]            │    │
│ │ Breast: [s] [s] ...                                 │    │
│ │ [precisa preencher com tempos reais]               │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Tabela 2: Regressão (5 bases, 7 modelos)          │    │
│ │ House: [s] [s] [s] [s] [s] [s] [s]               │    │
│ │ [precisa preencher com tempos reais]               │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ Interpretação: Ganho de produtividade, reprodutibilidade   │
└─────────────────────────────────────────────────────────────┘
```

## Página 4-5: Técnico + Insights

```
┌─────────────────────────────────────────────────────────────┐
│ SEÇÃO 4: IMPLEMENTAÇÃO (~250 palavras)                     │
│                                                             │
│ Parágrafo 1: Arquitetura modular (Registry, Pipeline)      │
│ Parágrafo 2: Threading, persistência, testes               │
│ Parágrafo 3: Stack detalhado (Python 3.9+, SQL, ML libs)   │
├─────────────────────────────────────────────────────────────┤
│ SEÇÃO 5: LIÇÕES APRENDIDAS (~250 palavras)                │
│                                                             │
│ Insight 1: Literatura integrada reduz barreira             │
│ Insight 2: Equilíbrio desempenho/interpretabilidade        │
│ Insight 3: Reprodutibilidade é essencial, não luxo        │
│ Insight 4: Histórico estruturado agrega valor             │
└─────────────────────────────────────────────────────────────┘
```

## Página 5-6: Conclusão + Referências

```
┌─────────────────────────────────────────────────────────────┐
│ SEÇÃO 6: CONCLUSÃO (~200 palavras)                         │
│                                                             │
│ Parágrafo 1: Demonstração de viabilidade                   │
│ Parágrafo 2: Diferencial (127 refs, robustez científica)   │
│ Parágrafo 3: Aplicações (pesquisa, clínica, educação)      │
│ Parágrafo 4: Trabalhos futuros (séries, SHAP, EHR)         │
│ Parágrafo 5: Disponibilidade                               │
│             [URL a ser preenchido]                         │
├─────────────────────────────────────────────────────────────┤
│ REFERÊNCIAS (13 papers):                                    │
│  [1] Breiman - Random Forests                              │
│  [2] Scikit-learn Paper                                    │
│  [3] XGBoost - Chen & Guestrin                             │
│  [4] Friedman - Gradient Boosting                          │
│  [5] Vapnik - SVM                                          │
│  [6] Wang et al - Sepsis Prediction                        │
│  ... (7 papers adicionais)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📏 Distribuição Aproximada de Páginas

| Seção | Páginas | Parágrafos | Palavras |
|-------|---------|-----------|----------|
| Título + Resumos | 1.0 | 4 | 400 |
| Introdução | 0.8 | 4 | 350 |
| Design | 1.5 | 5 | 700 |
| Avaliação | 0.8 | 3 + 2 tabelas | 400 |
| Implementação | 0.5 | 3 | 250 |
| Lições | 0.5 | 4 | 250 |
| Conclusão | 0.5 | 5 | 200 |
| Referências | 0.4 | 13 | ~200 |
| **TOTAL** | **~6 páginas** | **~41** | **~2,750 palavras** |

---

## 🎯 O que NÃO está no artigo (eliminado para compressão)

- ❌ Detalhado detalhado de cada um dos 8 classificadores (foram incorporados no Design)
- ❌ Detalhado detalhado de cada um dos 7 regressores (foram incorporados no Design)
- ❌ Seção separada de "Stakeholders" (incorporada na Introdução e Design)
- ❌ Detalhes de processo Agile (resumido em Implementação)
- ❌ Listas de RF1-RF10, NF1-NF7 (convertidas em prosa)
- ❌ Diagrama de Pipeline detalhado (mencionado em Implementação)
- ❌ Exemplo de relatório textuallongo (conceitual apenas)

... mas **toda essência está preservada**.

---

## 💾 Arquivos Finais

1. **artigo_sbcas.tex** - Documento principal (reescrito, 6 páginas)
2. **INSTRUCOES_ARTIGO.md** - Guia detalhado de submissão
3. **RESUMO_ALTERACOES.md** - Este documento
4. **MAPA_VISUAL.md** - Estrutura visual (este arquivo)

---

## ✅ Pronto para

1. **Compilar em Overleaf** ou LaTeX local → PDF
2. **Preencher tabelas** com tempos reais (2-3 horas de work)
3. **Gravar vídeo** de demonstração (30 min)
4. **Submeter em 2 de março de 2026**

---

**Boa sorte com a submissão!** 🚀📊🩺
