# Gate C1 — estudo de composição do corpus

## Objetivo

Este documento define a próxima etapa científica do Gate C1 do
`cambacica-base-180m`: comparar composições candidatas do corpus antes de
congelar o manifesto final.

A infraestrutura de amostragem e inspeção já foi validada. O problema agora é
decidir **qual papel cada fonte deve desempenhar no corpus do Cambacica**.

O estudo não deve começar escolhendo um número de tokens e tentando preenchê-lo.
Primeiro deve estabelecer quais dados merecem entrar; o volume final será uma
consequência das fontes, filtros, deduplicação e medições posteriores.

## 1. Princípios

O estudo deve preservar as regras de `docs/CORPUS.md`:

- pretraining nativamente em português;
- qualidade e proveniência acima de volume bruto;
- exclusão deliberada de datasets produzidos por tradução automática;
- preservação de diversidade de domínio e de variedades do português;
- reprodutibilidade das decisões;
- nenhuma hipótese de que um modelo maior será treinado depois;
- custo de retreinamento compatível com a infraestrutura da Orion.

Não devem ser copiados automaticamente pesos de Manacá, Tucano ou qualquer
outro modelo.

## 2. Papel das fontes já caracterizadas

### Corpus Carolina

Papel esperado: núcleo PT-BR contemporâneo e de alta proveniência.

Forças:

- taxonomias explícitas;
- forte presença de dados estruturados e contemporâneos;
- proveniência auditável;
- variedade PT-BR documentada pela própria fonte.

Riscos:

- domínios extremamente desbalanceados em volume;
- documentos legislativos muito longos;
- necessidade de licença por componente e deduplicação global.

### Wikipédia em português

Papel esperado: componente enciclopédico e factual pluricêntrico.

Forças:

- estrutura relativamente limpa;
- cobertura temática ampla;
- bom sinal factual e lexical.

Riscos:

- variedade nacional não rotulada por artigo;
- potencial overlap com agregadores como GigaVerbo;
- tradução humana ou automática pode existir sem marcação confiável por página.

### ParlamentoPT

Papel esperado: âncora de PT-PT formal e institucional.

Forças:

- português europeu nativo;
- linguagem política, administrativa e parlamentar;
- grande volume disponível.

Riscos:

- fórmulas repetitivas e boilerplate;
- viés de domínio se super-representado;
- necessidade de deduplicação e controle de peso.

### Literatura em domínio público

Papel esperado: diversidade estilística, lexical, histórica e literária.

Forças:

- escrita nativamente em português;
- diversidade de época e estilo;
- potencial para ampliar cobertura lexical além de web contemporânea.

Riscos:

- ortografia histórica;
- obras extremamente longas ou de referência;
- status de domínio público precisa continuar documentado por obra/jurisdição.

### GigaVerbo-v2 residual

Papel esperado: reservatório de escala, diversidade temática e linguagem web.

Forças:

- grande disponibilidade;
- scores upstream de qualidade educacional;
- diversidade de fontes e estilos;
- capacidade de fornecer volume sem novo crawl amplo.

Riscos:

- tradução latente em web crawl;
- subsets explicitamente traduzidos ou sintéticos;
- heterogeneidade de licença;
- overlap com fontes primárias;
- classificadores upstream podem introduzir vieses;
- composição física dos Parquets não deve ser confundida com distribuição
  científica desejada.

## 3. Famílias de composição a comparar

O estudo deve comparar pelo menos três famílias conceituais. Elas não possuem
ainda percentuais ou budgets congelados.

### A — Curated-heavy

Núcleo formado majoritariamente por fontes de proveniência forte e diretamente
interpretáveis:

- Carolina;
- Wikipédia;
- ParlamentoPT;
- literatura;
- eventualmente fontes acadêmicas/institucionais aprovadas em segunda rodada.

GigaVerbo residual entra apenas para cobrir lacunas de domínio, registro e
vocabulário.

Pergunta científica: quanto desempenho geral é possível obter mantendo a web
residual como componente minoritário?

### B — Balanced

Mantém o mesmo núcleo curado, mas usa GigaVerbo residual como componente
relevante de diversidade e escala.

Pergunta científica: qual equilíbrio melhora generalização sem diluir a
identidade nativa e a auditabilidade do corpus?

### C — Residual-heavy control

Mantém um núcleo mínimo garantido de fontes curadas, mas aumenta
substancialmente a participação do GigaVerbo residual.

Essa família existe como controle experimental, não como escolha presumida.

Pergunta científica: quanto ganho vem simplesmente de mais diversidade web e
quanto desse ganho justifica o custo e a perda de controle sobre proveniência?

## 4. Eixos que devem orientar a decisão

Cada composição candidata deve ser comparada segundo:

- proporção de documentos e bytes por fonte;
- proporção de texto PT-BR, PT-PT e variedade desconhecida;
- distribuição de domínio;
- distribuição de comprimento de documento;
- presença de literatura, texto científico, institucional, factual e informal;
- qualidade upstream e qualidade medida localmente;
- taxa de documentos removidos por idioma e integridade;
- deduplicação exata e aproximada intra/inter-fonte;
- overlap com benchmarks de avaliação;
- completude de metadados e proveniência;
- restrições de licença e redistribuição;
- volume único após filtros e deduplicação;
- custo de armazenamento e reconstrução;
- fertilidade do tokenizer quando o Gate C2 fornecer candidatos.

Nenhum único eixo deve ser tratado como substituto de qualidade global.

## 5. Dados que ainda precisam ser obtidos antes de congelar pesos

### Fontes atuais

Para cada fonte, precisamos medir sobre a população final elegível, não apenas
sobre amostras diagnósticas:

- volume bruto;
- volume após filtros de integridade;
- volume após deduplicação exata;
- volume após deduplicação aproximada;
- volume removido por overlap com outras fontes;
- distribuição de domínio e variedade quando disponível.

### Segunda rodada

Antes de fechar o C1, devem ser avaliados pelo menos como candidatos:

- Ulysses/Tesemõ;
- SciELO.

A inclusão não é obrigatória. O objetivo é decidir com evidência se eles
acrescentam informação que não está suficientemente representada nas fontes já
aceitas.

BDTD direto e FineWeb-2 direto permanecem fora do caminho principal enquanto
não houver uma necessidade concreta.

## 6. Relação com o Gate C2

C1 e C2 possuem uma dependência controlada.

C1 deve congelar:

- fontes elegíveis;
- políticas de filtro;
- política de deduplicação;
- famílias de mistura candidatas;
- manifests de documentos.

C2 deve então medir tokenização real dessas famílias.

Antes de C2, números em palavras, caracteres ou tokens de um tokenizer externo
podem ser usados apenas como diagnóstico. O budget oficial em tokens do
Cambacica deve ser medido com o tokenizer escolhido pelo próprio projeto.

Se C2 revelar que uma composição fragmenta sistematicamente uma variedade ou
domínio importante, C1 pode ser reaberto de forma explícita e documentada.

## 7. Experimentos mínimos para escolher a mistura

A escolha final não deve depender somente de estatísticas offline.

Após C2 e antes do treinamento principal, as famílias A/B/C devem ser reduzidas
a um pequeno número de recipes concretas e comparadas em runs curtos e baratos,
com o mesmo modelo, tokenizer e budget visto.

Métricas mínimas:

- validation loss/perplexity por domínio;
- benchmarks em português que apresentem sinal em modelos pequenos;
- estabilidade de treinamento;
- sensibilidade a variedades do português;
- qualidade em texto factual, institucional, literário e web;
- throughput e custo real na Orion.

O objetivo não é otimizar cada fonte isoladamente, e sim selecionar uma mistura
que seja boa para um modelo geral pequeno em português.

## 8. Critérios para encerrar C1

C1 poderá ser marcado como PASS quando houver:

- fontes finais selecionadas e revisões pinned;
- política de idioma e qualidade congelada;
- política de licença documentada;
- deduplicação global definida e executável;
- decontaminação de benchmarks definida;
- split train/validation/test reproduzível;
- manifests finais dos documentos;
- estatísticas de composição antes/depois dos filtros;
- famílias de mistura reduzidas a uma decisão científica justificável;
- interface clara para o Gate C2 medir os tokens reais.

Até lá, C1 permanece **IN PROGRESS**.