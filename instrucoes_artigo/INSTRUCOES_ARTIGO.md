# Instruções para Completar o Artigo SBCAS 2026 (Versão Revisada)

## Visão Geral

O arquivo `artigo_sbcas.tex` foi **completamente reescrito** para garantir:
- ✅ **Linguagem científica profissional** com tom humanizado
- ✅ **Máximo 6 páginas** (incluindo referências)
- ✅ **Dados pessoais preenchidos**: Thiago Guilherme Bezerra de Aguiar, UECE
- ✅ **Estrutura fluida** com texto corrido (sem listas excessivas)

## Informações Já Preenchidas ✅

1. **Autor e Afiliação**
   - Nome: Thiago Guilherme Bezerra de Aguiar
   - Email: thiago.aguiar@aluno.uece.br
   - Instituição: Universidade Estadual do Ceará (UECE)

2. **Estrutura do Artigo**
   - Introdução: Motivação, desafios e escopo
   - Design: Workflow, arquitetura, modelos (8+7), literatura integrada
   - Avaliação Experimental: 2 tabelas (classificação + regressão)
   - Implementação: Stack tecnológico, considerações técnicas
   - Lições Aprendidas: 4 insights principais
   - Conclusão: Impacto e trabalhos futuros
   - Referências: 13 papers fundamentais

3. **Números Integrados**
   - 15 algoritmos (8 classificação, 7 regressão)
   - 127 referências na literatura integrada
   - ~17.000 linhas de documentação educacional
   - 10 bases de dados para validação

## Placeholders Ainda a Preencher

### 1. **Tabelas de Benchmarks** (IMPORTANTE)

Linhas ~95-140 do arquivo .tex contêm duas tabelas com placeholders `[s]` e `[n]`:

**Tabela 1: Classificação**
```latex
Heart & 918 & [s] & [s] & [s] & ... (preencha com tempos em segundos)
```

**Tabela 2: Regressão** 
```latex
House & [n] & [s] & [s] & [s] & ... (preencha com número de amostras + tempos)
```

**Como obter:**
1. Execute a ferramenta com cada base de dados
2. Treine todos os 8 modelos (classificação) ou 7 (regressão)
3. Anote o tempo total da aba de comparação
4. Divida pelo número de modelos (se não houver timestamp individual)
5. Cole os valores nas tabelas

### 2. **URL da Ferramenta** (CRÍTICO)

Procure na seção de Conclusão (última página):
```latex
\noindent \textbf{Disponibilidade:} Código, vídeo de demonstração funcional 
e datasets de teste estão em [URL a ser preenchido].
```

Substitua por:
```latex
\noindent \textbf{Disponibilidade:} Código, vídeo de demonstração funcional 
e datasets de teste estão em \url{https://seu-repositorio-aqui}. 
Documentação completa acompanha o repositório.
```

### 3. **Screenshot (Opcional, mas Recomendado)**

Se quiser incluir uma figura da interface, o template está pronto. Basta:
1. Capturar screenshot da ferramenta
2. Salvar como `screenshot_app.png` no mesmo diretório
3. O documento está preparado para incluir via `\includegraphics`

---

## Checklist de Revisão Antes de Submeter

- [ ] Todas as tabelas de benchmarks foram preenchidas com dados reais
- [ ] URL do repositório está correta e funcional
- [ ] Email de contato está correto (thiago.aguiar@aluno.uece.br)
- [ ] Documento foi compilado sem erros em Overleaf ou LaTeX local
- [ ] PDF final tem **exatamente 6 páginas** (contar manualmente)
- [ ] Todas as referências estão citadas no texto
- [ ] Fontes estão embarcadas no PDF (importante para revisores)
- [ ] Vídeo de demonstração foi enviado para YouTube/Vimeo (obrigatório)
- [ ] URL do vídeo está pronta para colar no formulário de submissão

---

## Passo-a-Passo: Compilação em Overleaf (Recomendado)

1. Acesse **overleaf.com** e crie nova conta (se necessário)
2. Clique em **"New Project"** → **"Blank Project"**
3. Delete o arquivo `main.tex` padrão
4. Crie novo arquivo chamado `artigo_sbcas.tex`
5. Cole todo o conteúdo do arquivo .tex deste projeto
6. Clique em **"Recompile"** (botão verde)
7. Após compilação bem-sucedida, baixe o PDF: **"Download"** → **"PDF"**

---

## Passo-a-Passo: Compilação Local (Windows/Linux/Mac)

### Windows (PowerShell)

```powershell
cd c:\Users\thiag\trAin
pdflatex artigo_sbcas.tex
pdflatex artigo_sbcas.tex  # Execute 2x para referências
# Resultado: artigo_sbcas.pdf
```

### Linux/Mac

```bash
cd /caminho/para/trAin
pdflatex artigo_sbcas.tex
pdflatex artigo_sbcas.tex
```

**Pré-requisitos:** TeX Live ou MiKTeX instalado.

---

## Importante: O Vídeo é Obrigatório!

Prepare um vídeo **5-10 minutos** mostrando:
1. **Instalação** (~30 seg): Clonar repo, instalar requirements, rodar app
2. **Carregamento de Dados** (~1 min): Upload CSV, visualizar preview
3. **Seleção de Modelos** (~1 min): Escolher até 5 algoritmos, ajustar hiperparâmetros
4. **Execução** (~1 min): Clicar "Executar", mostrar progresso real-time
5. **Análise Completa** (~3-5 min): Métricas, gráficos, relatórios
6. **Aba de Literatura** (~1-2 min): Destacar diferencial educacional

**Ferramentas livres para gravar:**
- OBS Studio (excelente, multiplataforma)
- ScreenFlow (Mac)
- Camtasia (pago, mas polido)

**Upload:**
- YouTube (recomendado) - Pode ser "Unlisted"
- Vimeo (alternativa)

---

## Submissão Final (Prazo: 2 de março de 2026)

**Plataforma:** https://jems3.sbc.org.br/sbcas_fa2026

**Arquivos necessários:**
1. `artigo_sbcas.pdf` (máximo 6 páginas)
2. URL do vídeo (copiar no campo de submissão)

**Metadados do formulário:**
- Título: "trAIn Health: Uma Plataforma para Experimentação Reprodutível em Machine Learning Clínico"
- Autor: Thiago Guilherme Bezerra de Aguiar
- Email: thiago.aguiar@aluno.uece.br
- Afiliação: Universidade Estadual do Ceará
- Resumo: Cole do abstract do artigo
- Palavras-chave: Machine Learning, Saúde Digital, Reprodutibilidade Científica, Literatura Integrada, Experimentação Automizada

---

## Dicas de Qualidade Final

### Para o Artigo:
- Leia em voz alta para detectar problemas de fluidez
- Verifique referências (todos os [X] citados estão em `\thebibliography`?)
- Confirm margins, font, espaçamento (deve parecer profissional)
- Teste compilação final 48h antes da submissão

### Para o Vídeo:
- **Áudio claro**: Use microfone decente ou gravador de sistema com boa captura
- **Resolução 1080p mínimo**: Fonte legível é crítica
- **Edição simples**: Cortes básicos, sem efeitos complexos
- **Apresentação calma**: Pausas para leitura, não mude telas muito rápido

---

## Contatos Úteis

- **SBCAS 2026 Trilha FA:** romuere(at)ufpi(dot)edu(dot)br, fsumika(at)ufop(dot)edu(dot)br
- **JEMS3 Suporte:** https://jems3.sbc.org.br (chat ou email na plataforma)

---

**Status do Artigo:** 90% pronto para submissão
**Tempo estimado para finalizar:** 2-3 horas (tabelas + vídeo)
**Data recomendada de submissão:** Até 28 de fevereiro (margem de segurança)

Boa sorte! 🚀📊🩺

