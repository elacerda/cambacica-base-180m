# Corpus do Cambacica

## Status

Gate: **C1 — corpus**

Status: **EM DEFINIÇÃO**

Este documento registra a política científica inicial para o corpus de
pretraining do `cambacica-base-180m`.

O objetivo do C1 não é maximizar o número de tokens disponíveis. O objetivo é
construir um corpus auditável, nativamente em português e adequado a um modelo
pequeno, com no máximo aproximadamente 200 milhões de parâmetros, que possa ser
treinado e retreinado rapidamente na Orion.

Nenhuma decisão neste documento pressupõe uma futura versão maior do
Cambacica. Um eventual modelo de outra escala será uma decisão independente.

## 1. Princípio central

O Cambacica será treinado **do zero e nativamente em português**.

Isso significa que o corpus deve ser escolhido pelo valor que oferece a um
modelo pequeno em português, e não por compatibilidade com modelos estrangeiros,
por disponibilidade de grandes volumes multilíngues ou por um objetivo futuro
de escala.

Para esta etapa:

- qualidade tem prioridade sobre volume bruto;
- proveniência tem prioridade sobre conveniência;
- dados traduzidos automaticamente não constituem uma fonte deliberada de
  pretraining;
- diversidade linguística e de domínio deve ser preservada sem transformar o
  corpus em uma coleta indiscriminada da web;
- toda transformação relevante deve ser reproduzível e auditável.

## 2. O que significa "português nativo"

A política do Cambacica distingue quatro situações.

### 2.1 Texto originalmente escrito em português

É o alvo principal do corpus.

Inclui, por exemplo:

- textos jornalísticos;
- documentos científicos e acadêmicos;
- literatura;
- textos enciclopédicos;
- documentos governamentais e institucionais;
- textos de web genuinamente escritos em português;
- textos informais ou conversacionais quando a proveniência e a qualidade forem
  aceitáveis.

### 2.2 Texto traduzido automaticamente para português

Datasets ou coleções produzidos deliberadamente por tradução automática de
outras línguas **não devem ser incluídos como fonte normal de pretraining**.

Isso inclui conjuntos originalmente em inglês que tenham recebido uma versão
portuguesa por tradução automática em massa.

Uma futura ablação científica pode estudar esse tipo de dado, mas ela deve ser
explicitamente separada do corpus principal.

### 2.3 Páginas multilíngues classificadas incorretamente como português

Devem ser rejeitadas quando o conteúdo predominante não for português.

Um rótulo upstream como `pt` ou `por_Latn` é evidência útil, mas não deve ser
considerado prova suficiente de que o documento é adequado ao Cambacica.

### 2.4 Texto português com conteúdo estrangeiro natural

Deve ser preservado.

São exemplos normais:

- nomes próprios;
- citações;
- termos técnicos;
- nomes de software;
- expressões latinas;
- nomenclatura científica;
- pequenos trechos em outras línguas dentro de documentos originalmente
  escritos em português.

A política não busca uma porcentagem artificial de zero palavras estrangeiras.
Ela busca evitar dependência deliberada de conteúdo não nativo ou traduzido.

## 3. Variedades do português

O Cambacica não deve tratar `por_Latn` como sinônimo de português brasileiro.

O corpus pode e deve preservar, quando houver dados adequados e verificáveis:

- português brasileiro;
- português europeu;
- variedades africanas do português;
- textos históricos ou anteriores ao Acordo Ortográfico de 1990.

Neste estágio não há pesos fixados para essas variedades.

A origem geográfica ou variedade linguística deve ser registrada quando puder
ser inferida de forma confiável a partir da própria fonte ou de metadados. Não
devemos fabricar rótulos nacionais quando a fonte apenas garante que o texto é
português.

## 4. Lições do Manacá

A auditoria local do Manacá mostrou que o corpus publicado foi muito mais
restrito do que o framework geral de aquisição presente no repositório.

O treinamento reportado usa três componentes principais:

- GigaVerbo;
- Wikipedia em português;
- Ulysses/Tesemõ.

O paper reporta aproximadamente 20,1 bilhões de tokens únicos, com proporções
aproximadas de 73,1% GigaVerbo, 23,9% Ulysses e 3,1% Wikipedia.

Essas proporções **não são uma receita para o Cambacica**. O Manacá é um modelo
de outra escala e o peso elevado de textos jurídicos pode ser inadequado para
um modelo geral com menos de 200 milhões de parâmetros.

### 4.1 Práticas úteis

São candidatos fortes a reutilização conceitual:

- manter documentos explícitos até o fim do preprocessing;
- usar marcador de fim de documento;
- separar aquisição, limpeza, deduplicação, validação e preparação para treino;
- caracterizar quantitativamente o corpus antes do treinamento;
- versionar as fontes externas;
- manter uma trilha de auditoria científica.

### 4.2 Práticas que não serão copiadas automaticamente

Não devem ser herdados sem nova evidência:

- a proporção 73/24/3;
- o uso de aproximadamente 24% de texto jurídico;
- a seleção sequencial de shards de GigaVerbo usada na reprodução local;
- o objetivo amplo de 1T tokens presente no framework de corpus do Manacá;
- o tokenizer de 64k como escolha padrão;
- o uso de `por_Latn` ou de um rótulo upstream como prova de variedade
  linguística;
- a implementação atual de deduplicação do upstream;
- estimativas de tokens baseadas somente em caracteres.

### 4.3 Melhoria de reprodutibilidade

A auditoria do Manacá encontrou lacunas entre o procedimento descrito no paper
e o caminho executável disponível para deduplicação e preparação dos dados.

O Cambacica deve evitar esse tipo de ambiguidade.

O corpus congelado para treinamento deverá possuir um manifesto que identifique
exatamente os arquivos e transformações usados.

## 5. Lições do GigaVerbo e Tucano

GigaVerbo e GigaVerbo-v2 demonstram que já existe um grande reservatório de
dados em português que pode ser usado como matéria-prima.

Para o Cambacica, a principal consequência é que **não precisamos construir um
Common Crawl português completo para obter volume suficiente**.

GigaVerbo-v2 é especialmente interessante porque oferece metadados de qualidade
educacional, toxicidade e proveniência que permitem amostragem e análise mais
controladas.

Entretanto:

- o dataset é muito maior do que o Cambacica necessita;
- subconjuntos traduzidos ou de origem inadequada devem ser removidos;
- classificadores de qualidade upstream podem introduzir vieses;
- uma partição considerada educacional não deve ser aceita sem caracterização
  local;
- os números de composição e qualidade devem ser obtidos da revisão específica
  do dataset usada pelo projeto, não de estimativas secundárias.

## 6. Fontes candidatas

A lista abaixo é uma lista de investigação, não uma composição aprovada.

### 6.1 Candidatos de alta prioridade

**Wikipedia em português**

- texto enciclopédico e relativamente estruturado;
- licença e revisão devem ser fixadas no manifesto;
- mistura de variedades do português deve ser preservada, não rotulada
  artificialmente como PT-BR.

**Corpus Carolina**

- candidato a núcleo de alta proveniência para português brasileiro;
- metadados e licença por componente devem ser auditados antes da inclusão.

**GigaVerbo-v2**

- candidato a fonte principal de diversidade;
- deve ser amostrado de forma estratificada, não simplesmente concatenado;
- devem ser priorizadas partições de alta qualidade;
- fontes explicitamente produzidas por tradução automática devem ser excluídas
  do corpus principal.

**Ulysses/Tesemõ**

- fonte valiosa de português institucional, legislativo e jurídico;
- candidata a participação menor do que no Manacá;
- seu peso deve ser medido para evitar viés excessivo de domínio.

### 6.2 Candidatos que exigem auditoria de acesso e licença

**SciELO**

- potencial fonte de português científico revisado;
- volume utilizável, acesso programático, licença por artigo e possibilidade de
  redistribuição precisam ser verificados antes de contar tokens.

**BDTD e repositórios acadêmicos**

- potencial fonte de dissertações e teses;
- licenças, duplicação entre repositórios, PDFs/OCR e metadados precisam ser
  caracterizados.

**Domínio Público e literatura em domínio público**

- potencial fonte de literatura brasileira, portuguesa e lusófona;
- versões, OCR, ortografia histórica e duplicatas precisam ser controlados.

### 6.3 Fontes de web ampla

FineWeb-2 PT, HPLT e outras coleções de Common Crawl não são prioridades do C1
neste momento.

Elas podem ser úteis se a caracterização mostrar que o corpus compacto carece
de diversidade, mas não devem ser adicionadas apenas para aumentar o número de
tokens.

## 7. Duas estratégias candidatas

O C1 trabalhará inicialmente com duas famílias de corpus.

Nenhuma é ainda a configuração final.

### Estratégia A — núcleo curado compacto

Ordem de grandeza inicial para investigação: **aproximadamente 4–6 bilhões de
tokens únicos**.

Possíveis componentes:

- Wikipedia em português;
- Corpus Carolina;
- texto científico e acadêmico com licença adequada;
- literatura em domínio público;
- conteúdo institucional e governamental;
- subconjunto controlado de Ulysses/Tesemõ;
- fontes lusófonas adicionais quando forem verificáveis.

Objetivo: maximizar proveniência, português nativo e densidade de informação,
mantendo o treinamento barato e repetível.

### Estratégia B — núcleo curado + web educacional

Ordem de grandeza inicial para investigação: **aproximadamente 6–10 bilhões de
tokens únicos**, podendo ser revisada depois da medição real.

Parte do mesmo núcleo da Estratégia A e adiciona uma amostra estratificada de
alta qualidade do GigaVerbo-v2.

Objetivo: ganhar diversidade lexical, temática e estilística sem transformar o
Cambacica em um modelo treinado predominantemente em web bruta.

### Escalas maiores

Misturas de dezenas de bilhões de tokens de web ampla não são candidatas
principais neste estágio.

O Cambacica é deliberadamente pequeno e deve continuar barato o suficiente para
permitir experiências repetidas. Volume adicional só será considerado se uma
medição demonstrar benefício claro para esse modelo.

## 8. Corpus único versus tokens vistos pelo treinamento

O projeto deve manter separadas três grandezas:

1. **tokens únicos do corpus congelado**;
2. **tokens efetivamente vistos durante o treinamento**;
3. **dados usados para treinar o tokenizer**.

Elas não precisam ser iguais.

É aceitável repetir de forma controlada dados de alta qualidade durante o
pretraining, desde que a política de repetição seja explícita e medida.

O número final de épocas ou tokens vistos pelo modelo não será decidido no C1
somente a partir de scaling laws genéricas. Ele dependerá também dos benchmarks
reais de treinamento do Cambacica na Orion.

## 9. Pipeline obrigatório

A implementação poderá mudar, mas o corpus final deverá conceitualmente seguir
esta ordem:

```text
aquisição versionada
        ↓
normalização mínima e preservação de metadados
        ↓
validação de idioma
        ↓
filtros de integridade e qualidade
        ↓
remoção de boilerplate quando aplicável
        ↓
deduplicação exata
        ↓
deduplicação aproximada
        ↓
decontaminação dos benchmarks
        ↓
caracterização final
        ↓
split de treino / validação / teste
        ↓
manifesto congelado
        ↓
tokenização / formato de treinamento
```

A ordem poderá ser refinada se houver uma justificativa científica explícita,
mas duplicatas não devem poder atravessar os splits.

## 10. Identificação de idioma

A política inicial é:

- fontes nativamente portuguesas e curadas podem aproveitar a garantia da
  própria fonte, mas ainda devem ser caracterizadas;
- fontes web ou multilíngues exigem identificação de idioma por documento;
- classificações upstream devem ser preservadas como metadado, não tratadas
  como verdade absoluta;
- documentos curtos ou tecnicamente densos não devem ser rejeitados somente
  porque um classificador de idioma tem baixa confiança;
- a decisão deve ser testada em amostras manualmente inspecionadas.

FastText, GlotLID ou outras ferramentas são candidatos; nenhuma combinação está
congelada neste documento.

A classificação PT-BR/PT-PT/PALOP deve ser separada da decisão "é português?".
Não devemos transformar heurísticas lexicais simples em rótulos científicos sem
validação.

## 11. Filtragem de qualidade

Filtros simples e transparentes devem existir desde o início:

- texto vazio ou inválido;
- comprimentos extremos;
- razão anormal de caracteres alfabéticos;
- excesso de símbolos;
- repetição interna extrema;
- páginas compostas majoritariamente de menus, templates ou boilerplate.

Classificadores educacionais ou de qualidade, como os fornecidos pelo
GigaVerbo-v2, podem ser usados como sinais adicionais.

Eles não substituem caracterização independente.

Filtros por perplexidade, toxicity classifiers ou classificadores semânticos
não são requisitos desta primeira especificação. Se usados, seus efeitos devem
ser medidos e documentados.

## 12. Deduplicação

A deduplicação do Cambacica deve ser explicitamente reproduzível.

O baseline deve incluir:

1. deduplicação exata por hash de conteúdo normalizado;
2. deduplicação aproximada baseada em MinHash/LSH ou método equivalente;
3. deduplicação global entre fontes, e não somente dentro de cada fonte.

Os parâmetros exatos ainda não estão congelados.

A implementação deverá produzir estatísticas de:

- documentos antes e depois;
- tokens antes e depois;
- duplicatas intra-fonte;
- duplicatas inter-fonte;
- clusters de deduplicação por origem.

Deduplicação semântica por embeddings não é requisito inicial. Pode ser testada
como ablação em subconjuntos menores.

## 13. Decontaminação

Benchmarks usados para avaliar o Cambacica devem ser registrados antes do
congelamento final do corpus.

O corpus deve ser analisado contra seus conjuntos de validação e teste antes da
criação dos splits finais.

O método exato de matching ainda será definido, mas deve ser suficientemente
reproduzível para que possamos responder quais documentos foram removidos ou
marcados e por quê.

A caracterização de overlap, por si só, não será considerada equivalente a uma
etapa de decontaminação.

## 14. Split de treino, validação e teste

O split final deve ocorrer **depois da deduplicação global e da decontaminação**.

Esse requisito evita que near-duplicates de um documento apareçam ao mesmo
tempo em treino e validação/teste.

Os splits deverão ser determinísticos e reproduzíveis a partir do manifesto.

A política exata de proporções será definida depois que conhecermos o número e o
tamanho real dos documentos candidatos.

## 15. Metadados e manifesto

A meta é preservar, quando disponível, pelo menos:

```text
source
source_revision
source_id
original_url
license
language
language_score
language_variety
quality_score
content_hash
dedup_cluster
split
```

Nem toda fonte fornecerá todos os campos. Valores ausentes devem permanecer
explicitamente ausentes, em vez de serem inventados.

O corpus congelado deverá registrar adicionalmente:

- revisão exata de cada dataset upstream;
- lista de arquivos/shards usados;
- checksums;
- versão do código de preprocessing;
- configuração de filtros;
- configuração de deduplicação;
- seed(s);
- contagens por fonte antes e depois de cada estágio;
- contagem com o tokenizer final quando o C2 estiver concluído.

## 16. Tokenizer e C1

O C1 deve produzir texto limpo e metadados sem depender de uma decisão precoce
sobre o tokenizer.

Contagens preliminares de tokens podem usar um tokenizer provisório somente
como estimativa e devem ser identificadas como tal.

O orçamento final em tokens será recalculado com o tokenizer congelado no C2.

A amostra usada para treinar o tokenizer deverá ser estratificada por fonte e
por variedade linguística quando possível. Ela não precisa reproduzir
mecanicamente os pesos finais de treinamento.

## 17. Medições necessárias antes de congelar C1

Antes de marcar C1 como PASS, precisamos medir em amostras reais das fontes
candidatas:

- volume em documentos e bytes;
- distribuição de comprimentos;
- proporção de documentos claramente em português;
- erros dos detectores de idioma em amostra manual;
- presença de tradução automática quando detectável;
- duplicação intra e inter-fonte;
- distribuição por origem/domínio;
- distribuição de escores de qualidade onde disponíveis;
- cobertura de variedades do português onde for confiavelmente mensurável;
- custo de armazenamento e preprocessing;
- contagem de tokens com tokenizers candidatos quando C2 começar.

## 18. Critérios provisórios para C1 PASS

C1 poderá ser fechado quando:

- todas as fontes escolhidas estiverem identificadas e versionadas;
- licença e condições de uso estiverem registradas;
- a política de português nativo estiver implementada e validada por amostra;
- os filtros de qualidade estiverem documentados;
- a deduplicação global estiver implementada e medida;
- o conjunto de benchmarks para decontaminação estiver definido;
- o split final estiver especificado;
- o manifesto do corpus puder ser reproduzido a partir do código e das
  revisões registradas;
- o tamanho real do corpus estiver conhecido;
- a estratégia A ou B, ou uma variante delas, tiver sido escolhida com base nas
  medições.

Até lá, C1 permanece **EM DEFINIÇÃO**.

## 19. Questões abertas

1. Qual subconjunto de GigaVerbo-v2 oferece melhor compromisso entre qualidade,
   diversidade e português originalmente escrito?
2. Qual participação de Ulysses melhora linguagem formal sem dominar o estilo
   do modelo?
3. Quais fontes científicas e acadêmicas podem ser adquiridas e redistribuídas
   de forma reproduzível?
4. Quais fontes abertas permitem representar melhor PT-PT e variedades
   africanas sem recorrer a tradução automática?
5. Qual combinação de LangID minimiza falsos positivos sem destruir documentos
   portugueses técnicos ou multilíngues naturais?
6. Quais parâmetros de MinHash são adequados para o tamanho real do corpus?
7. Qual orçamento de tokens únicos permite a melhor relação entre qualidade e
   frequência de retreinamento para um modelo <=200M?
8. Quantos tokens efetivamente vistos durante o treino são úteis antes que a
   curva de ganho deixe de justificar tempo adicional na Orion?

## 20. Próxima ação

A próxima etapa do C1 é construir um **inventário mensurável das fontes
candidatas** e obter amostras pequenas e reproduzíveis.

Essas amostras serão usadas para substituir estimativas secundárias por medidas
reais antes de escolher a composição final do corpus.
