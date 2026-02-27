# ✅ CHECKLIST FINAL DE SUBMISSÃO

## 🔵 FASE 1: Artigo Escrito (100% COMPLETO ✅)

### 1.1 Documento Principal
- [x] `artigo_sbcas.tex` criado e reescrito
- [x] 6 páginas máximo (atual: ~5.8 páginas)
- [x] Formato SBC compatível (12pt, margens 1.5cm)
- [x] Seções estruturadas:
  - [x] Título + Resumo PT/EN
  - [x] Introdução (problema + solução)
  - [x] Design da Plataforma (arquitetura + 127 refs + 17k docs)
  - [x] Avaliação Experimental (2 tabelas, 10 bases)
  - [x] Implementação (SOLID, patterns, stack)
  - [x] Lições Aprendidas (4 insights)
  - [x] Conclusão + Trabalhos Futuros
  - [x] Referências (13 papers)
- [x] Linguagem humanizada (parágrafos fluidos, sem bullets excessivos)
- [x] Autor preenchido: Thiago Guilherme Bezerra de Aguiar
- [x] Email preenchido: thiago.aguiar@aluno.uece.br
- [x] Afiliação preenchida: UECE - Universidade Estadual do Ceará

### 1.2 Conteúdo Técnico Validado
- [x] 127 referências bibliográficas mencionadas ✓
- [x] ~17.000 linhas de documentação citadas ✓
- [x] 15 modelos (8 classificadores + 7 regressores) nomeados ✓
- [x] 10 datasets especificados (5 classificação + 5 regressão) ✓
- [x] Arquitetura corretamente descrita (Registry, Pipeline, Threading) ✓
- [x] Stack tecnológico correto (Python 3.9+, PyQt6, scikit-learn, XGBoost) ✓
- [x] SBCAS requirements cobertos:
  - [x] Descrição do problema ✓
  - [x] Stakeholders (integrados naturalmente) ✓
  - [x] Processo de desenvolvimento ✓
  - [x] Testes e avaliação ✓
  - [x] Lições aprendidas ✓
  - [x] Trabalhos futuros ✓

### 1.3 Documentação de Suporte
- [x] `INSTRUCOES_ARTIGO.md` - Guia completo de submissão
- [x] `RESUMO_ALTERACOES.md` - Auditoria de mudanças
- [x] `MAPA_VISUAL.md` - Estrutura visual (recém-criado)

---

## 🟡 FASE 2: Dados Faltantes (PRÓXIMA AÇÃO - ~3 horas)

### 2.1 Tabelas de Benchmark (CRÍTICA)
**Status:** Estrutura pronta, dados faltando

#### Tabela 1: Classificação
- Dataset: Heart, Breast Cancer, Diabetes, Iris, Titanic
- Modelos: 8 (LR, KNN, NB, SVM, DT, RF, GB, XGB)
- Placeholder: `[s]` = segundos de execução
- Linhas a preencher: 5 × 8 = 40 células

**Como obter:**
```bash
1. Abra trAIn Health (main_gui.py)
2. Carregue base "Heart"
3. Selecione "Treinar Todos"
4. Anote tempo total em segundos da aba Comparação
5. Repita para Breast Cancer, Diabetes, Iris, Titanic
6. Repita procedimento anterior para cada modelo individualmente se necessário
```

**Tempo estimado:** ~1-2 horas (10 min por base)

#### Tabela 2: Regressão
- Dataset: House Prices, Boston Housing, Diabetes, Insurance, Medical
- Modelos: 7 (LR, Ridge, SVM-R, DT-R, RF-R, GB-R, XGB-R)
- Placeholder: `[s]` = segundos de execução
- Linhas a preencher: 5 × 7 = 35 células

**Tempo estimado:** ~1-2 horas (10 min por base)

**Como editar o arquivo:**
```latex
% Abra artigo_sbcas.tex, procure por:
% Linha ~135: \begin{tabular}{|l|...|r|r|...|r|}
% Replace [s] com valores reais, ex: 2.34, 3.12, etc.
```

### 2.2 Vídeo de Demonstração (OBRIGATÓRIO)
**Status:** Não gravado ainda
**Requisito SBCAS:** 5-10 minutos, demonstração funcional

**Roteiro sugerido (7-8 minutos):**
- **0:00-0:30** - Apresentação e instalação (pip install, requirements)
- **0:30-1:00** - Estrutura de dados esperada (CSV, colunas)
- **1:00-2:00** - Carregar base de dados (interface)
- **2:00-3:00** - Selecionar features, preprocessadores, balanceadores
- **3:00-4:30** - Treinar modelos (destacar que é automático, 15 modelos)
- **4:30-5:30** - Aba de Comparação (mostrar métricas, ROC curves)
- **5:30-7:00** - Aba de Literatura (DESTAQUE - mostrar 127 refs, fundamentação teórica)
- **7:00-8:00** - Exportar histórico, reprodutibilidade

**Ferramentas recomendadas:**
- OBS Studio (grátis, open-source)
- Camtasia (pago, profissional)
- Windows Game Bar (Win+G - grátis para Win10+)

**Upload obrigatório:**
- YouTube (public ou unlisted)
- Vimeo, ou
- Outro host de vídeo com URL pública

**Tempo estimado:** 30-45 min (gravação + edição básica)

### 2.3 URL do Repositório (CRÍTICA)
**Status:** Placeholder `[URL a ser preenchido]` no Conclusão

**Requisitos:**
- GitHub, GitLab, Gitea, ou similar
- Incluir:
  - `README.md` com instruções de instalação
  - `requirements.txt` com dependências
  - Código-fonte completo (`main.py`, `main_gui.py`, pasta `src/`)
  - Datasets de exemplo (mínimo 1-2 pequenos)
  - Documentação técnica

**Onde está no artigo:**
```latex
% Linha ~180 (no final da Conclusão):
\noindent \textbf{Disponibilidade:} Código, vídeo de demonstração funcional e datasets 
de teste estão em [URL a ser preenchido].
```

**Como editar:**
```latex
% Replace [URL a ser preenchido] com:
% Exemplo: https://github.com/seu-usuario/train-health
% Exemplo: https://gitea.seu-servidor.com/seu-usuario/train-health
```

**Tempo estimado:** 10 min (se repo já existe) ou 30 min (criar repo + upload)

---

## 🟢 FASE 3: Compilação e Submissão (FINAL - ~1 hora)

### 3.1 Compilação Local
```bash
# Se tiver LaTeX instalado localmente:
pdflatex artigo_sbcas.tex
pdflatex artigo_sbcas.tex  # 2x para resolver referências
# Output: artigo_sbcas.pdf
```

### 3.2 Compilação em Overleaf (RECOMENDADO)
1. Vá para https://www.overleaf.com
2. Crie novo projeto (New Project → Upload)
3. Upload `artigo_sbcas.tex`
4. Preencha placeholders `[s]` e `[URL]` no editor
5. Compile (botão verde "Recompile")
6. Download PDF

### 3.3 Validação Pré-Submissão
- [ ] PDF compilado sem erros
- [ ] 6 páginas exatas (verificar zoom 100%)
- [ ] Todas as tabelas preenchidas (sem `[s]`)
- [ ] URL no final está correta e acessível
- [ ] Autor: "Thiago Guilherme Bezerra de Aguiar"
- [ ] Afiliação: "UECE"
- [ ] Resumo em português ✓
- [ ] Abstract em English ✓
- [ ] Referências como última página

### 3.4 Submissão no SBCAS
1. Vá para https://www.sbcas2026.com
2. Faça login (crie conta se necessário)
3. Vá para trilha "Tools & Applications"
4. Botão "Submit Paper"
5. Upload `artigo_sbcas.pdf`
6. Upload vídeo (YouTube link ou upload direto)
7. Preencha metadados (título, autores, keywords)
8. Confirmar submissão

**DEADLINE: 2 de março de 2026**

---

## 📋 TIMELINE SUGERIDA

| Fase | Tarefa | Duração | Prazo |
|------|--------|---------|-------|
| **Agora** | Executar benchmarks (10 bases × 8-7 modelos) | 2h | Hoje |
| **+2h** | Preencher placeholders nas tabelas | 20min | Hoje |
| **+2.5h** | Gravar vídeo (planejamento + gravação + edição) | 1h | Amanhã |
| **+3.5h** | Upload vídeo e obter URL | 20min | Amanhã |
| **+4h** | Posso fazer setup repositório se necessário | 30min | Amanhã |
| **+4.5h** | Compilar PDF final em Overleaf | 10min | Amanhã |
| **+5h** | Submeter no SBCAS | 15min | Amanhã |

**Total:** ~5 horas de trabalho prático

---

## 🚨 CHECKLIST PRÉ-SUBMISSÃO (Final)

### Artigo
- [ ] Arquivo `artigo_sbcas.tex` pronto
- [ ] Sem placeholders `[s]` (todos preenchidos)
- [ ] Sem placeholders `[n]` (todos preenchidos, se houver)
- [ ] URL do repositório preenchida (não `[URL a ser preenchido]`)
- [ ] Autor correto: Thiago Guilherme Bezerra de Aguiar
- [ ] Email correto: thiago.aguiar@aluno.uece.br
- [ ] Afiliação correta: UECE

### PDF
- [ ] Compilado sem errors ou warnings
- [ ] Exatamente 6 páginas
- [ ] Todas imagens/tabelas visíveis
- [ ] Referências funcionando (hyperlinks)
- [ ] Fonte legível (12pt minimum)

### Vídeo
- [ ] Gravado e editado (5-10 min)
- [ ] Áudio claro
- [ ] Demonstração clara de funcionalidades
- [ ] Literatura tab bem explicada (127 refs)
- [ ] Hospedado em URL pública (YouTube/Vimeo/etc)

### Repositório
- [ ] Código-fonte completo incluído
- [ ] `README.md` com instruções
- [ ] `requirements.txt` atualizado
- [ ] Datasets de teste inclusos
- [ ] URL acessível e funcionando

### SBCAS
- [ ] Conta criada em sbcas2026.com
- [ ] Trilha correta: "Tools & Applications"
- [ ] Metadados preenchidos (título, keywords)
- [ ] PDF upload confirmado
- [ ] Vídeo link confirmado
- [ ] Submissão finalizada ANTES de 2 março 2026

---

## 📞 Próximos Passos

### Imediatamente (Hoje):
```
1. Execute benchmarks nas 10 bases
2. Preencha [s] nas Tabelas 1 e 2
3. Recompile PDF em Overleaf para validar
```

### Amanhã:
```
1. Grave e edite vídeo (30-45 min)
2. Upload em YouTube/Vimeo (obtenha URL)
3. Setup/push repositório GitHub (se não existir)
4. Preencha [URL a ser preenchido] com link real
5. Final compilation em Overleaf
6. Submeta no SBCAS
```

---

**Artigo está 95% pronto. Só faltam dados e vídeo.** 🎯
