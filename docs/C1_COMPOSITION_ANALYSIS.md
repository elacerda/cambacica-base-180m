# Gate C1 — Estudo Científico de Composição do Corpus e Especificação de Misturas

## Status

Gate: **C1 — corpus**  
Status: **IN PROGRESS**  
Data de referência: **2026-10-01**  
Commit de referência da amostragem: `065c460`  

---

## 1. Resumo de Evidências e Infraestrutura Medida

Este documento formaliza o primeiro estudo científico de composição do corpus de pretraining do **Cambacica** (`cambacica-base-180m`), transformando as propostas conceituais em uma **especificação rigorosa de misturas candidatas**.

### 1.1 Escopo do Modelo e Restrições Científicas
O Cambacica possui identidade e restrições técnicas fixadas no Gate C0:
- **Treinado do zero** (*from scratch*), exclusivamente com inicialização aleatória de pesos;
- **Decodificador causal autorregressivo** (*decoder-only* causal LM);
- **Nativamente em português**, priorizando textos originalmente gerados em língua portuguesa e rejeitando corpora deliberadamente traduzidos por máquina;
- **Escala compacta**: $\le 200\text{M}$ parâmetros (alvo nominal de ~180M parâmetros);
- **Treinamento barato e reprodutivo**: projetado para rodar confortavelmente em ~40 GB de VRAM de uma única GPU NVIDIA H100 no cluster Orion;
- **Modelo base**: focado em densidade informacional, coerência sintática e conhecimento enciclopédico/factual na base, sem alinhamento prévio ou instruct;
- **Modelo autônomo**: não constitui um *stepping stone* ou versão reduzida de uma família maior planejada.

Em consonância com [`docs/CORPUS.md`](CORPUS.md) e [`docs/C1_CORPUS_COMPOSITION_STUDY.md`](C1_CORPUS_COMPOSITION_STUDY.md), **a composição não começa a partir de uma meta abstrata de contagem de tokens**. A primeira pergunta científica que este estudo responde é:
> *"O que o corpus do Cambacica deve conter e qual papel cada fonte deve desempenhar para um modelo causal de 180M parâmetros?"*

### 1.2 Medição Real do Armazenamento no Sistema
A infraestrutura de armazenamento local sob `/mnt/data` foi aferida diretamente via comandos de sistema operacional em 2026-10-01:

```text
Filesystem:           10.10.10.1:/srv/data (NFSv4) montado em /mnt/data
Capacidade total:     66 TB
Espaço em uso:        1,6 TB
Espaço livre real:    64 TB (3% de utilização)
Ocupação atual:       /mnt/data/cambacica-base-180m consome atualmente 776 MB
```

Essa medição confirma que o espaço físico não é um gargalo para o projeto. O princípio de parcimônia nos downloads decorre de **disciplina científica e controle de proveniência**, e não de restrição de disco.

### 1.3 Base de Evidências Medidas nas Amostras Persistentes
A análise apoia-se em sete amostras persistentes normalizadas sob o esquema comum Arrow/Parquet (`src/cambacica/corpus/schema.py`) e armazenadas localmente no storage `/mnt/data/cambacica-base-180m/samples/gate_c1/`:

| Amostra / Arquivo | Modo | Documentos | Caracteres | Palavras | Chars/Doc (Média) | Words/Doc (Média) | Duplicatas Exatas | Docs Quase-Vazios (<100c/<20w) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `carolina/representative.parquet` | `representative` | 10.000 | 49.614.232 | 7.857.679 | 4.961,4 | 785,8 | 0 (0,0%) | 2.472 (24,72%) |
| `carolina/diagnostic.parquet` | `diagnostic` | 10.000 | 2.032.363.446 | 321.716.862 | 203.236,3 | 32.171,7 | 0 (0,0%) | 60 (0,60%) |
| `wikipedia_pt/representative.parquet` | `representative` | 10.000 | 75.408.485 | 11.952.658 | 7.540,9 | 1.195,3 | 0 (0,0%) | 91 (0,91%) |
| `parlamento_pt/representative.parquet` | `representative` | 10.000 | 38.502.428 | 5.890.694 | 3.850,2 | 589,1 | 1.145 (11,45%) | 1.478 (14,78%) |
| `gutenberg_pt/representative.parquet` | `representative` | 100 | 33.963.334 | 5.135.100 | 339.633,3 | 51.351,0 | 0 (0,0%) | 0 (0,0%) |
| `gigaverbo_v2/audit.parquet` | `audit` | 25.000 | 206.484.383 | 32.536.803 | 8.259,4 | 1.301,5 | 1 (0,0%) | 0 (0,0%) |
| `gigaverbo_v2/candidate.parquet` | `candidate` | 25.000 | 207.905.563 | 32.779.867 | 8.316,2 | 1.311,2 | 1 (0,0%) | 0 (0,0%) |

Principais constatações empíricas da auditoria:
1. **Disparidade extrema de tamanho por domínio no Corpus Carolina**: Documentos legislativos (`leg`) e judiciais (`jud`) contêm textos integrais com até milhões de palavras por registro TEI, enquanto o subconjunto social/web (`dat`) apresenta 24,7% de registros muito curtos (tweets, comentários breves).
2. **Redundância de fórmulas no ParlamentoPT**: 11,45% da amostra representativa de debates parlamentares consiste em repetições exatas de fórmulas regimentais, despachos e cabeçalhos editoriais curtos.
3. **Composição e filtragem no GigaVerbo-v2**: Na partição `edu_high`, a exclusão formal eliminou 55.519 registros de tradução sintética (`ultrachat`) durante a coleta dos 25.000 documentos elegíveis. Os 11 subconjuntos sobreviventes possuem distribuições de tamanho e escores de qualidade heterogêneos, destacando-se `finepdfs_por_Latn` como fonte de textos formais e longos.
4. **Isolamento de Overlap nas Amostras**: 0 colisões exatas e 0 near-duplicates com Jaccard $\ge 0,80$ foram encontrados entre as amostras persistentes das diferentes fontes, embora a sobreposição latente na população total permaneça um risco a ser mitigado na deduplicação global.

---

## 2. Perfil Científico das Cinco Famílias de Fontes

Para cada uma das fontes candidatas, constrói-se um perfil detalhado com base nas evidências coletadas, rotulando expressamente a natureza de cada afirmação:
- `[DOCUMENTADO]`: fato verificado em especificações formais, repositórios oficiais ou artigos acadêmicos das fontes.
- `[MEDIDO]`: valor quantificado diretamente a partir das amostras persistentes e manifests do C1.
- `[INFERIDO]`: dedução científica sustentada por teoria linguística, evidência indireta ou padrões de pré-treino de LLMs.
- `[UNKNOWN]`: lacuna científica atual que requer investigação empírica futura.

```
+----------------------------------------------------------------------------------------------------+
|                                    CORPUS DO CAMBACICA (~180M)                                     |
+-----------------------------+-----------------------------+----------------------------------------+
| Fonte Primária              | Variedade / Registro        | Papel Científico Principal             |
+-----------------------------+-----------------------------+----------------------------------------+
| Corpus Carolina             | PT-BR (contemporâneo)       | Núcleo sintático e lexical de alta     |
|                             | Misto (social/inst/jur)     | proveniência brasileira.               |
+-----------------------------+-----------------------------+----------------------------------------+
| Wikipédia em português      | Pluricêntrico (PT-BR/PT-PT) | Âncora de conhecimento enciclopédico,  |
|                             | Expositivo formal           | densidade factual e prosa estruturada. |
+-----------------------------+-----------------------------+----------------------------------------+
| ParlamentoPT                | PT-PT (contemporâneo)       | Âncora de sintaxe europeia, oratória   |
|                             | Político / institucional    | formal e registro parlamentar falado.  |
+-----------------------------+-----------------------------+----------------------------------------+
| Literatura (Gutenberg PT)   | PT-BR / PT-PT (histórico)   | Densidade vocabular, complexidade      |
|                             | Literário / ensaístico      | sintática e expressividade estilística.|
+-----------------------------+-----------------------------+----------------------------------------+
| GigaVerbo-v2 residual       | Predom. PT-BR (web)         | Reservatório de diversidade temática,  |
|                             | Educacional / técnico / web | linguagem contemporânea e escala.      |
+-----------------------------+-----------------------------+----------------------------------------+
```

### 2.1 Corpus Carolina
- **Identificador upstream**: `carolina-c4ai/corpus-carolina` (v2.0.1, commit `55e63a519393c70a48dcfa14a558499c6bb0583b`) `[DOCUMENTADO]`.
- **Força de proveniência**: Muito alta `[DOCUMENTADO]`. Curado pelo C4AI (USP/FAPESP/IBM), com anotações padronizadas TEI XML e tipologia textual explícita.
- **Confiança de português nativo**: Muito alta `[DOCUMENTADO]`. Textos de proveniência brasileira verificada.
- **Variedade linguística conhecida**: Português Brasileiro (PT-BR) contemporâneo (1970–presente) `[DOCUMENTADO]`.
- **Contribuição de domínio e estilo**: Amplo e estruturado `[DOCUMENTADO]`. Abrange 7 taxonomias populacionais:
  - `dat` (web social, blogs, fóruns, tweets): 50,93% dos documentos;
  - `wik` (dados enciclopédicos e informativos estilo Wiki): 45,40% dos documentos;
  - `jud` (decisões e acórdãos judiciais): 1,81% dos documentos;
  - `uni` (comunicação universitária e acadêmica): 1,25% dos documentos;
  - `soc` (coletivos sociais, ONGs e política): 0,42% dos documentos;
  - `leg` (decretos legislativos e atos normativos): 0,19% dos documentos;
  - `pub` (publicidade): ~0,001% dos documentos.
- **Distribuição de comprimento**: Extremamente assimétrica `[MEDIDO]`.
  - No modo `representative` (10k docs): mediana de 811 caracteres (130 palavras); média de 4.961 caracteres (785,8 palavras).
  - No modo `diagnostic` (cotas fixas para categorias raras): o subconjunto `leg` eleva a média para 203.236 caracteres (32.171 palavras) e P99 de 2.027.714 caracteres. Documentos de leis compiladas chegam a mais de 16 milhões de caracteres em um único registro TEI.
- **Comportamento de repetição e duplicação**: 0% de duplicatas exatas na amostra `[MEDIDO]`. Registros com $\ge 20\%$ de linhas repetidas somam apenas 0,27% no modo representativo.
- **Metadados de qualidade**: Curadoria acadêmica rigorosa, sem score numérico contínuo `[DOCUMENTADO]`.
- **Incerteza de licença e redistribuição**: Moderada `[DOCUMENTADO]`. O Corpus Carolina possui licenciamento fragmentado por documento/taxonomia (alguns CC-BY, outros sob termos de uso público ou permissão de pesquisa). A redistribuição como dataset derivado exige auditoria das licenças por componente.
- **Sobreposição provável com outras fontes**: Alta com Wikipédia `[INFERIDO]`. A taxonomia `wik` (45,4% de Carolina) contém artigos de projetos Wiki lusófonos, gerando colisão direta com o dump da Wikipédia se não houver deduplicação global.
- **Valor científico único para modelo $\le 200\text{M}$**: Fornece uma base nativa inequívoca da prosódia e da gramática do português brasileiro contemporâneo em múltiplos registros (culto, coloquial, institucional).
- **Principais riscos de super-representação**: Inundação do contexto por decretos legislativos extensos (`leg`) ou por ruído de micro-textos em redes sociais (`dat`), desequilibrando a capacidade de modelagem do Cambacica.

### 2.2 Wikipédia em Português
- **Identificador upstream**: `wikimedia/wikipedia` (snapshot `20231101.pt`, commit `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`) `[DOCUMENTADO]`.
- **Força de proveniência**: Muito alta `[DOCUMENTADO]`. Dump oficial da Wikimedia Foundation processado com parser padronizado de wikitext.
- **Confiança de português nativo**: Alta `[DOCUMENTADO] / [INFERIDO]`. Majoritariamente escrito por falantes nativos, com presença de traduções manuais ou assistidas a partir de outras edições da Wikipédia.
- **Variedade linguística conhecida**: Pluricêntrica `[DOCUMENTADO] / [UNKNOWN]`. Não há etiquetagem no nível de artigo entre PT-BR, PT-PT e variedades africanas (PALOP); estimativas gerais apontam ~80% PT-BR e ~20% PT-PT/PALOP.
- **Contribuição de domínio e estilo**: Prosa expositiva factual, densa em entidades nomeadas, termos científicos, históricos, geográficos e culturais `[DOCUMENTADO]`.
- **Distribuição de comprimento**: Consistente e moderada `[MEDIDO]`.
  - Média: 7.540,9 caracteres (1.195,3 palavras).
  - Mediana: 2.663 caracteres (421 palavras); P90 de 19.445 caracteres; Max de 142.220 caracteres.
  - Documentos quase-vazios: apenas 0,91% (esboços curtos).
- **Comportamento de repetição e duplicação**: 0% de duplicatas exatas na amostra `[MEDIDO]`. Linhas repetidas $\ge 20\%$ em apenas 0,42% dos artigos (geradas por caixas de navegação ou infoboxes residuais).
- **Metadados de qualidade**: Curadoria colaborativa contínua, sem escore de qualidade nativo nos dumps brutos `[DOCUMENTADO]`.
- **Incerteza de licença e redistribuição**: Baixa `[DOCUMENTADO]`. Licença aberta CC BY-SA 3.0 / 4.0 e GFDL, plenamente compatível com redistribuição e auditoria.
- **Sobreposição provável com outras fontes**: Crítica `[DOCUMENTADO] / [INFERIDO]`. Aparece no Carolina (`wik`), em aggregators (GigaVerbo subset `wikipedia`) e em dumps gerais de web. Exige deduplicação exata e de quase-duplicatas rigorosa.
- **Valor científico único para modelo $\le 200\text{M}$**: Máxima densidade de conhecimento factual e gramática expositiva limpa por token; indispensável para modelagem de mundo em arquiteturas compactas.
- **Principais riscos de super-representação**: Monotonia de estilo expositivo, repetição de formatos de introdução enciclopédica ("X é uma cidade...", "Y foi um político...") e listas factuais desprovidas de conectivos discursivos.

### 2.3 ParlamentoPT
- **Identificador upstream**: `PORTULAN/parlamento-pt` (revisão `main`, commit `08f13e7e63ab9bfbd8c0b40955defe3bb7f68c2b`) `[DOCUMENTADO]`.
- **Força de proveniência**: Muito alta `[DOCUMENTADO]`. Transcrição oficial dos debates na Assembleia da República Portuguesa disponibilizada pela infraestrutura PORTULAN CLARIN.
- **Confiança de português nativo**: Absoluta (100%) `[DOCUMENTADO]`. Discurso oral parlamentar português transcrito e revisado oficialmente.
- **Variedade linguística conhecida**: Português Europeu (PT-PT) institucional contemporâneo `[DOCUMENTADO]`.
- **Contribuição de domínio e estilo**: Linguagem argumentativa, retórica deliberativa, intervenções faladas, vocabulário político, administrativo e econômico formal `[DOCUMENTADO]`.
- **Distribuição de comprimento**: Bimodal `[MEDIDO]`.
  - Média: 3.850,2 caracteres (589,1 palavras); mediana: 4.574 caracteres (687 palavras); P99: 6.716 caracteres.
  - Registros quase-vazios: 14,78% da amostra possuem <100 caracteres ou <20 palavras (interrupções orais, aplausos, protestos regimentais e anúncios da Mesa).
- **Comportamento de repetição e duplicação**: Muito alto `[MEDIDO]`.
  - **11,45% de duplicatas exatas** na amostra de 10.000 documentos.
  - Investigação diagnóstica comprovou que a duplicação advém de fórmulas burocráticas recorrentes: *"O Sr. Presidente: — Srs. Deputados, está aberta a sessão"*, avisos de leitura de expediente e notas editoriais padronizadas da imprensa oficial.
- **Metadados de qualidade**: Transcrição oficial padronizada, sem escores contínuos `[DOCUMENTADO]`.
- **Incerteza de licença e redistribuição**: Baixa `[DOCUMENTADO]`. Dados abertos governamentais distribuídos para pesquisa pela infraestrutura PORTULAN CLARIN.
- **Sobreposição provável com outras fontes**: Baixa `[INFERIDO]`. Quase nula sobreposição com Common Crawl ou Wikipedia, restrita a menções em portais institucionais portugueses.
- **Valor científico único para modelo $\le 200\text{M}$**: Âncora indispensável para o Português Europeu formal, garantindo o aprendizado de colocação pronominal em próclise/ênclise clássica (`falar-lhe-ei`, `disse-me`), construções verbais lusitanas e terminologia administrativa.
- **Principais riscos de super-representação**: Vício do modelo em fórmulas protocolares de plenário, distorção do estilo discursivo para debates inflamados e introdução de viés institucional excessivo se não houver deduplicação estrita de fórmulas.

### 2.4 Literatura em Português do Project Gutenberg
- **Identificador upstream**: Catálogo em português do Project Gutenberg (snapshot de catálogo com 655 eBooks elegíveis) `[DOCUMENTADO]`.
- **Força de proveniência**: Muito alta `[DOCUMENTADO]`. Edições digitalizadas e conferidas por revisores voluntários do Distributed Proofreaders.
- **Confiança de português nativo**: Absoluta (100%) `[DOCUMENTADO]`. Obras clássicas canônicas da literatura lusófona (Machado de Assis, Camões, Eça de Queirós, Raul Pompeia, Almeida Garrett, etc.).
- **Variedade linguística conhecida**: PT-BR e PT-PT históricos e literários (séculos XVI a início do século XX, anterior ao Acordo Ortográfico de 1990) `[DOCUMENTADO]`.
- **Contribuição de domínio e estilo**: Prosa de ficção, crônicas, dramaturgia, poesia, ensaios filosóficos e obras lexicográficas `[DOCUMENTADO]`.
- **Distribuição de comprimento**: Longa e pesada `[MEDIDO]`.
  - Média: 339.633,3 caracteres (51.351 palavras); mediana: 161.612 caracteres (24.945 palavras); P90: 398.138 caracteres.
  - Casos extremos: o eBook 31552 (*Novo Diccionário da Língua Portuguesa*, de Cândido de Figueiredo) apresenta 13.330.252 caracteres (~1,85 milhão de palavras) em um único texto completo.
  - Registros quase-vazios: 0,0%.
- **Comportamento de repetição e duplicação**: 0% de duplicatas exatas de livros `[MEDIDO]`. Linhas repetidas $\ge 20\%$ em 2,0% dos casos (índices, versos rimados ou cabeçalhos de capítulos). O extrator C1 já descarta os headers e footers padronizados do Gutenberg.
- **Metadados de qualidade**: Texto integral transcrito de livros físicos consagrados `[DOCUMENTADO]`.
- **Incerteza de licença e redistribuição**: Nula `[DOCUMENTADO]`. Obras em Domínio Público nas jurisdições aplicáveis e sob licença aberta do Gutenberg.
- **Sobreposição provável com outras fontes**: Baixa a moderada `[INFERIDO]`. Possível sobreposição parcial com repositórios abertos de livros (como `baixe_livros`, já bloqueado nas exclusões do GigaVerbo).
- **Valor científico único para modelo $\le 200\text{M}$**: Riqueza morfológica, variedade léxica sofisticada e estruturas de subordinação complexas ausentes da web comum; previne a degradação da prosa gerada pelo modelo em clichês contemporâneos.
- **Principais riscos de super-representação**: Ortografia arcaica pré-reforma (grafias como *sciencia*, *pharmacia*, *diccionario*, *d'elle*), que podem confundir o tokenizer e a geração ortográfica se o peso for desproporcional.

### 2.5 Residual Filtrado do GigaVerbo-v2 (`edu_high`)
- **Identificador upstream**: `Polygl0t/gigaverbo-v2` (`edu_high`, commit `7058ccf19eaeaf4505a96fc7e5305a01fc441fd8`) `[DOCUMENTADO]`.
- **Força de proveniência**: Média/Estruturada `[DOCUMENTADO]`. Agregador compilado pela comunidade Polygl0t com enriquecimento de metadados (`subset`, `quality_score`, `toxicity_score`).
- **Confiança de português nativo**: Mista / Variável `[INFERIDO] / [UNKNOWN]`. Abrange desde coleções puramente nativas (`blogset`, `quati`, `crawlPT`) até raspagens globais de Common Crawl com risco de tradução automática latente.
- **Variedade linguística conhecida**: Pluricêntrica não rotulada `[UNKNOWN]`. Predominantemente PT-BR com presença de web em língua portuguesa (.pt em `crawlPT`) e frações não medidas de PALOP.
- **Contribuição de domínio e estilo**: Notícias contemporâneas, manuais técnicos, blogs informativos, páginas institucionais e documentos PDF científicos `[DOCUMENTADO]`.
- **Distribuição de comprimento**: Ampla e balanceada `[MEDIDO]`.
  - Média: 8.316,2 caracteres (1.311,2 palavras); mediana: 4.057 caracteres (646 palavras); P90: 14.226 caracteres; P99: 68.147 caracteres.
  - Registros quase-vazios: 0,0% (graças aos filtros prévios do GigaVerbo e limiares de comprimento).
- **Comportamento de repetição e duplicação**: Quase nulo na amostra retida (1 duplicata exata em 25k docs) `[MEDIDO]`. Linhas repetidas $\ge 20\%$ em 0,61% dos documentos.
- **Metadados de qualidade**: Muito detalhados `[DOCUMENTADO] / [MEDIDO]`.
  - Escore contínuo de qualidade educacional predito por classificador FastText upstream.
  - Na partição `edu_high`: mínimo medido de 3,50; média de 3,79; mediana de 3,74; máximo de 5,03.
- **Incerteza de licença e redistribuição**: Alta `[UNKNOWN]`. Subconjuntos herdados de Common Crawl, FineWeb e mC4 não fornecem licença uniforme por URL.
- **Sobreposição provável com outras fontes**: Crítica `[DOCUMENTADO] / [INFERIDO]`. O dataset cru inclui shards de Wikipedia, Carolina, BDTD e livros. O filtro de exclusão configurado em [`configs/gigaverbo_exclusions.yaml`](../configs/gigaverbo_exclusions.yaml) remove essas colisões óbvias, mas sobreposições de páginas espelho e notícias sinfônicas ainda ocorrem.
- **Valor científico único para modelo $\le 200\text{M}$**: Fornece cobertura léxica de tópicos contemporâneos, vocabulário tecnológico e diversidade de registros do século XXI que não constam em bibliotecas clássicas.
- **Principais riscos de super-representação**: Diluição da precisão sintática por texto web medíocre, introdução de jargão de SEO e risco de absorção de "português traduzido" (*translationese*) não rotulado.

---

## 3. Análise do Residual do GigaVerbo e Estrutura Interna

O GigaVerbo-v2 é tratado estritamente como um **reservatório heterogêneo de subconjuntos**, e não como uma fonte monolítica.

### 3.1 Subconjuntos Sobreviventes às Exclusões Explícitas
Na execução do modo `candidate` sobre os 56 shards e 224 row groups distribuídos da partição `edu_high`, foram examinados 104.359 registros. Destes, 55.519 registros pertencentes ao subset sintético/traduzido `ultrachat` foram eliminados antes do reservatório determinístico, restando um pool elegível examinado de 48.840 registros.

Estatísticas medidas na amostra retida de 25.000 documentos:

| Subconjunto residual | Docs na amostra | Participação na amostra | Média de Caracteres | Mediana de Caracteres | Média de Palavras | Mediana de Palavras | Escore Médio de Qualidade |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fineweb_2_pt` | 8.176 | 32,70% | 5.895,1 | 3.826,5 | 943,5 | 606,0 | 3,787 |
| `hplt2_pt` | 7.113 | 28,45% | 6.296,5 | 3.873,0 | 1.011,2 | 619,0 | 3,793 |
| `mc4_pt` | 4.494 | 17,98% | 6.380,1 | 3.936,0 | 1.000,8 | 614,0 | 3,786 |
| `finepdfs_por_Latn` | 1.817 | 7,27% | **31.490,5** | **11.877,0** | **4.830,2** | **1.848,0** | **3,818** |
| `hplt1_pt` | 1.112 | 4,45% | 13.392,1 | 8.287,0 | 2.107,3 | 1.214,0 | 3,775 |
| `common_crawl` | 912 | 3,65% | 7.995,9 | 5.252,5 | 1.267,8 | 833,0 | 3,797 |
| `crawlPT_dedup` | 679 | 2,72% | 5.960,2 | 3.235,0 | 959,7 | 511,0 | 3,768 |
| `quati` | 267 | 1,07% | 1.169,1 | 1.202,0 | 187,4 | 188,0 | 3,723 |
| `oscar` | 197 | 0,79% | 7.909,5 | 4.974,0 | 1.273,4 | 816,0 | 3,780 |
| `blogset` | 121 | 0,48% | 2.420,7 | 2.548,0 | 372,5 | 401,0 | 3,760 |
| `culturax` | 112 | 0,45% | 5.668,4 | 3.352,0 | 906,1 | 557,0 | 3,786 |

> [!WARNING]
> As proporções acima refletem a distribuição encontrada na amostragem distribuída por row groups; elas **NÃO devem ser tomadas como estimativas da prevalência populacional completa** do GigaVerbo, mas sim como a evidência física direta das propriedades intrínsecas de cada subconjunto.

### 3.2 Especificação Provisória de Pesos Internos do GigaVerbo Residual
Para permitir a geração determinística de manifests e evitar contagem dupla ou sobreposição, todos os 11 subconjuntos sobreviventes são agrupados em **quatro baldes mutuamente exclusivos** com pesos operacionais fixados como hipóteses científicas (somando exatamente 100% do volume residual):

```
+----------------------------------------------------------------------------------------------------+
|                      ESTRUTURAÇÃO INTERNA DO RESIDUAL GIGAVERBO-V2 (100%)                          |
+--------------------------+---------------------+-------------------+-------------------------------+
| Balde / Categoria        | Subconjuntos        | Peso Provisório   | Papel e Justificativa         |
|                          | Alocados            | (Volume Palavras) | Científica                    |
+--------------------------+---------------------+-------------------+-------------------------------+
| **1. Formal / Técnico**  | `finepdfs_por_Latn` | **30,0%**         | Prosa acadêmica e técnica de  |
|                          |                     |                   | alto nível. Maior qualidade   |
|                          |                     |                   | medida (3,818). UPWEIGHT.     |
+--------------------------+---------------------+-------------------+-------------------------------+
| **2. Web Curada/Nativa** | `crawlPT_dedup`     | 10,0%             | Web em língua portuguesa (.pt).|
|                          | `quati`             | 5,0%              | Prosa didática/explicativa.   |
|                          | `blogset`           | 5,0%              | Registro coloquial autêntico. |
|                          | *Subtotal Balde 2*  | **20,0%**         | UPWEIGHT / PRESERVAR.         |
+--------------------------+---------------------+-------------------+-------------------------------+
| **3. Web Moderna Geral** | `fineweb_2_pt`      | **35,0%**         | Espinha dorsal de diversidade |
|                          |                     |                   | lexical e contemporaneidade.  |
|                          |                     |                   | PRESERVAR NÚCLEO WEB.         |
+--------------------------+---------------------+-------------------+-------------------------------+
| **4. Web Legada (Cap)**  | `mc4_pt`            | 5,0%              | Raspagens históricas de CC/IA.|
|                          | `hplt2_pt`          | 5,0%              | Alta redundância e risco      |
|                          | `hplt1_pt`          | 2,0%              | latente de tradução.          |
|                          | `common_crawl`      | 1,0%              | DOWNWEIGHT / CAP ESTRITO.     |
|                          | `oscar`             | 1,0%              |                               |
|                          | `culturax`          | 1,0%              |                               |
|                          | *Subtotal Balde 4*  | **15,0%**         |                               |
+--------------------------+---------------------+-------------------+-------------------------------+
| **TOTAL GIGAVERBO**      | **11 Subconjuntos** | **100,0%**        | Hipótese interna unificada.   |
+--------------------------+---------------------+-------------------+-------------------------------+
```

Detalhamento por balde:
- **Balde 1 (`finepdfs_por_Latn`)**:
  - `[DOCUMENTADO]`: PDFs de domínios públicos e universitários extraídos via pipeline FinePDFs.
  - `[MEDIDO]`: Média de 31.490 caracteres e 4.830 palavras por documento. Qualidade educacional média de 3,818.
  - `[INFERIDO]`: Atua como fonte funcional de prosa formal/científica/técnica sem a necessidade de um crawler de BDTD/SciELO (sem equivalência formal de proveniência a repositórios acadêmicos primários).
  - `[DECISÃO PROVISÓRIA]`: Alocação de 30% do residual (elevação deliberada frente à taxa de captura física).
- **Balde 2 (Web Curada e Nativa)**:
  - `[DOCUMENTADO]`: Textos originalmente escritos em português em domínios específicos (`crawlPT_dedup` para web em língua portuguesa (.pt), `quati` para material escolar e `blogset` para crônicas e blogs).
  - `[MEDIDO]`: Comprimentos moderados e baixíssimo risco de duplicação.
  - `[INFERIDO]`: Fornecem português natural não contaminado por tradução mecânica de portais corporativos globais.
  - `[DECISÃO PROVISÓRIA]`: Alocação de 20% do residual, garantindo equilíbrio pluricêntrico e estilístico.
- **Balde 3 (Web Moderna Geral — `fineweb_2_pt`)**:
  - `[DOCUMENTADO]`: Filtros contemporâneos do FineWeb aplicados à web lusófona.
  - `[MEDIDO]`: Escore médio de qualidade de 3,787 com mediana equilibrada (606 palavras).
  - `[INFERIDO]`: Melhor fonte geral para generalização e cobertura enciclopédica moderna não contida na Wikipédia.
  - `[DECISÃO PROVISÓRIA]`: Alocação de 35% do residual, constituindo o núcleo web dominante.
- **Balde 4 (Web Legada Genérica com Teto)**:
  - `[DOCUMENTADO]`: Derivados de Common Crawl e Internet Archive de 2019 a 2023.
  - `[MEDIDO]`: Comprimentos médios em torno de 6k–13k caracteres; grande volume bruto upstream.
  - `[INFERIDO]`: Elevada redundância mútua. Sem um teto rígido, esses conjuntos inundariam o modelo com texto de baixa densidade informativa e boilerplate de SEO.
  - `[DECISÃO PROVISÓRIA]`: Teto global de 15% do residual, divididos proporcionalmente entre as variantes.

---

## 4. Definição da Unidade de Mistura e Famílias de Composição A / B / C

### 4.1 Por que Frações de Documentos são Cientificamente Inválidas
Frações ou porcentagens baseadas em **contagem de documentos são completamente inválidas** para definir a mistura do Cambacica.

A evidência medida nas amostras persistentes comprova que o comprimento dos documentos varia por **mais de quatro ordens de magnitude**:
1. No **Corpus Carolina**, enquanto a mediana da taxonomia social `dat` é de apenas **130 palavras** (com 24,7% de documentos com <20 palavras), a média da taxonomia legislativa `leg` é de **32.171 palavras**, com registros individuais ultrapassando **2,6 milhões de palavras**.
2. Na **Literatura do Project Gutenberg**, a média por eBook é de **51.351 palavras**, com obras de referência chegando a **1,85 milhão de palavras**.
3. Se a mistura fosse estipulada em contagem de documentos:
   - 100 livros de literatura clássica representariam 1% de um lote de 10.000 documentos, mas injetariam **5,1 milhões de palavras**.
   - 5.093 microtextos de redes sociais de Carolina representariam 50,9% dos documentos, mas injetariam apenas **~660 mil palavras**.
   - O volume real de aprendizado seria completamente distorcido em favor de documentos hiper-extensos ou micro-textos, destruindo qualquer controle científico da dieta do modelo.

### 4.2 Definição Canônica da Unidade de Mistura do C1
> [!IMPORTANT]
> Todas as porcentagens e pesos de mistura apresentados neste documento e nas configurações do projeto definem expressamente:
> **"provisional share of normalized text volume; C1 proxy measured in words, not final tokenizer tokens"**
> (fatia provisória de volume de texto normalizado; proxy do Gate C1 medido em palavras, e não tokens finais de tokenizer).
> 
> Contagens de palavras **NÃO devem ser chamadas de tokens**. A conversão para o orçamento oficial em tokens ocorrerá no Gate C2, utilizando o tokenizer treinado especificamente para o Cambacica.

### 4.3 Alvos Canônicos das Três Famílias de Composição (100% Exato)
Cada família apresenta um **alvo canônico exato (somando 100,0%)** acompanhado de suas faixas de tolerância experimental:

```
+----------------------------------------------------------------------------------------------------+
|                                COMPARAÇÃO DAS FAMÍLIAS DE COMPOSIÇÃO                               |
|   (Unidade: fatia provisória de volume de texto normalizado medida em palavras [C1 proxy])         |
+--------------------------+--------------------+--------------------+-------------------------------+
| Componente / Fonte       | Família A          | Família B          | Família C                     |
|                          | (Curated-Heavy)    | (Balanced)         | (Residual-Heavy Control)      |
+--------------------------+--------------------+--------------------+-------------------------------+
| **Corpus Carolina**      | **40,0%** (35–45%) | **28,0%** (25–30%) | **12,0%** (10–15%)            |
| **Wikipédia PT**         | **25,0%** (20–25%) | **18,0%** (15–20%) | **7,0%**  (5–8%)              |
| **ParlamentoPT**         | **10,0%** (10–15%) | **10,0%** (8–12%)  | **4,0%**  (3–5%)              |
| **Literatura Gutenberg** | **8,0%**  (5–10%)  | **4,0%**  (3–5%)   | **2,0%**  (1–2%)              |
| *Total Núcleo Curado*    | *83,0%* (80–85%)   | *60,0%* (55–65%)   | *25,0%* (20–30%)              |
+--------------------------+--------------------+--------------------+-------------------------------+
| **GigaVerbo Residual**   | **17,0%** (15–20%) | **40,0%** (35–45%) | **75,0%** (70–80%)            |
+--------------------------+--------------------+--------------------+-------------------------------+
| **SOMA TOTAL EXATA**     | **100,0%**         | **100,0%**         | **100,0%**                    |
+--------------------------+--------------------+--------------------+-------------------------------+
```

### 4.4 Racional Científico de Cada Família

#### Família A: Curated-Heavy (Alta Proveniência e Densidade Informacional)
- **Alvo Canônico**: Carolina 40%, Wikipédia 25%, ParlamentoPT 10%, Gutenberg 8%, GigaVerbo residual 17%.
- **Hipótese científica**: Para uma rede causal de ~180M parâmetros, capacidade de representação é um recurso crítico. Alimentar a rede com prosa densa em conhecimento, gramaticalmente impecável e de autoria nativa inquestionável gera representações semânticas superiores com menor taxa de alucinação sintática. O residual web é limitado a uma fatia de 17% estritamente controlada para evitar lacunas de vocabulário moderno.

#### Família B: Balanced (Generalista Equilibrado)
- **Alvo Canônico**: Carolina 28%, Wikipédia 18%, ParlamentoPT 10%, Gutenberg 4%, GigaVerbo residual 40%.
- **Hipótese científica**: Hipótese central do projeto. Equilibra a disciplina sintática das fontes curadas (60% do volume) com a flexibilidade temática e contemporaneidade da web filtrada (40% do volume). Previne o enrijecimento estilístico em formatos enciclopédicos/jurídicos enquanto blinda o modelo contra a degradação gerada por web crawl em excesso.

#### Família C: Residual-Heavy Control (Controle de Escala Web)
- **Alvo Canônico**: Carolina 12%, Wikipédia 7%, ParlamentoPT 4%, Gutenberg 2%, GigaVerbo residual 75%.
- **Hipótese científica**: Linha de base de controle experimental. Reproduz o padrão usual de grandes modelos da literatura aberta (como Llama, Mistral e Tucano), onde raspagens web em massa dominam 75% da dieta de treino. Permite testar empiricamente se o volume e a variedade bruta de web crawl superam a curadoria em uma rede de escala compacta ($\le 200\text{M}$).

---

## 5. Política de Fronteiras de Documentos e Empacotamento

A política do corpus do Cambacica estabelece uma separação rigorosa de responsabilidades entre o Gate C1 (Corpus) e o Gate C4 (Pipeline de Dados e Treinamento):

1. **Preservação Integral da Identidade dos Documentos no C1**:
   - Cada texto normalizado retém sua identidade original e suas fronteiras canônicas de início e fim de documento (*End-Of-Document* / EOD).
   - **É terminantemente proibido truncar arbitrariamente documentos extensos no C1** (como livros inteiros do Gutenberg ou leis consolidadas do Carolina). Cortar um documento em um número arbitrário de caracteres destruiria a coerência sintática da obra no repositório de dados.
2. **Segmentação e Empacotamento (*Packing*) como Atribuição do Gate C4**:
   - O comprimento de contexto do modelo causal (ex: 2.048 ou 4.096 tokens) será definido nos Gates C3 e C4.
   - O pipeline de treinamento no C4 será responsável por empacotar múltiplos documentos curtos separados por delimitadores EOD (`<|endoftext|>`), bem como por **fracionar documentos extremamente longos** em segmentos de treino que caibam na janela de contexto.
   - **Proteção contra anomalias**: Um documento legislativo com 16 milhões de caracteres ou um dicionário com 1,8 milhão de palavras **nunca se tornará uma única sequência de treino descontrolada**. O empacotador do C4 fará a divisão em fatias sequenciais preservando metadados de continuidade discursiva.

---

## 6. Decisão sobre Fontes de Segunda Rodada

Avaliação científica da necessidade de inclusão ou inspeção antes do congelamento do C1:

### 6.1 Ulysses / Tesemõ
- **Informação que adicionaria**: Textos legislativos e normativos das casas federais do Brasil.
- **Já está suficientemente representado?**: **Sim.** O Corpus Carolina já supre o domínio jurídico/normativo brasileiro com `jud` (38.187 docs) e `leg` (3.982 docs), que somam centenas de milhões de palavras em registros oficiais completos. O ParlamentoPT complementa com o domínio parlamentar europeu. No Manacá, Ulysses ocupou 23,9% do treino, gerando um modelo viciado em jargão de diários oficiais. Um modelo de 180M parâmetros não pode desperdiçar parâmetros com redundância legislativa.
- **Custo de engenharia e licença**: Baixo a moderado, mas desnecessário.
- **Decisão formal**: **DEFER** (fora do pipeline principal do C1).

### 6.2 SciELO (Scientific Electronic Library Online)
- **Informação que adicionaria**: Artigos acadêmicos primários revisados por pares em diversas áreas da ciência em português.
- **Já está suficientemente representado?**: **Parcialmente.** A Wikipédia traz síntese factual; o Carolina `uni` possui 26.409 documentos; e o subconjunto `finepdfs_por_Latn` do GigaVerbo atua como fonte funcional de documentos e relatórios científicos/técnicos formais (sem equivalência formal de proveniência a SciELO ou BDTD).
- **Custo de engenharia e licença**: Muito alto. O SciELO exige raspagem de repositório OAI-PMH ou APIs JATS/XML, tratamento de licenças CC variáveis e filtragem estrita de idioma (grande volume de artigos em espanhol e inglês).
- **Decisão formal**: **DEFER para ingestão primária / NICE TO HAVE se surgir dump pré-processado auditado**. Não iniciaremos crawler próprio no C1.

---

## 7. Plano Concreto de Materialização de Dados

> [!CAUTION]
> **Importante**: O GigaVerbo-v2 não pode ser materializado baixando apenas "shards selecionados", pois os subconjuntos apresentam clustering físico em nível de *row groups* e coexistem distribuídos pelos 56 shards da partição. A aquisição do GigaVerbo deve ocorrer por streaming e filtragem direta de *row groups/subsets* no momento da extração para os buffers locais.

### 7.1 Matriz de Materialização por Fonte

```
+---------------------------------------------------------------------------------------------------------------------------------------+
|                                              MATRIZ DE MATERIALIZAÇÃO DE DADOS (GATE C1)                                              |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| Fonte                    | Revisão Pinned        | Tamanho    | Modo de   | Destino em           | Tamanho    | Overhead  | Requisito |
|                          | Upstream              | Raw Estim. | Aquisição | /mnt/data/           | Normaliz.  | Temp.     | Checksum  |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **Corpus Carolina**      | v2.0.1                | ~8,3 GB    | Totalidade| `.../raw/carolina/`  | ~12–16 GB  | ~10 GB    | Commit    |
|                          | `55e63a519393...`     | (823 shards| (todas    |                      | (Parquet   |           | SHA-256   |
|                          |                       |  xml.gz)   | taxonom.) |                      |  ZSTD)     |           | por shard |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **Wikipédia PT**         | `20231101.pt`         | ~2,5 GB    | Totalidade| `.../raw/wikipedia/` | ~7–9 GB    | ~5 GB     | Dump hash |
|                          | `b04c8d1ceb2f...`     | (comprim.) | (artigos) |                      | (Parquet)  |           | oficial   |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **ParlamentoPT**         | `main`                | ~2,7 GB    | Totalidade| `.../raw/parlamento/`| ~3–4 GB    | ~3 GB     | Commit    |
|                          | `08f13e7e63ab...`     | (`train.txt`)| (linhas)|                      | (Parquet)  |           | SHA-256   |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **Literatura Gutenberg** | Snapshot 2026-10-01   | ~0,4 GB    | Totalidade| `.../raw/gutenberg/` | ~0,5 GB    | ~0,5 GB   | EBook IDs |
|                          | (655 eBooks PT)       | (plain txt)| (catálogo)|                      | (Parquet)  |           | validados |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **GigaVerbo Residual**   | `edu_high`            | ~25–35 GB  | Streaming | `.../raw/gigaverbo/` | ~20–30 GB  | ~25 GB    | Commit    |
|                          | `7058ccf19eae...`     | (seletivo) | filtrado  |                      | (Parquet)  |           | Exclusions|
|                          |                       |            | por subset|                      |            |           | SHA-256   |
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
| **TOTAIS DO WORKSPACE**  | **Todas Pinned**      | **~39–49GB**| **Seletivo**| `/mnt/data/cambacica`| **~42–60GB**| **~43–45GB**| **Auditável|
+--------------------------+-----------------------+------------+-----------+----------------------+------------+-----------+-----------+
```

### 7.2 Reavaliação do Orçamento de Armazenamento Frente à Medição Real
- **Capacidade disponível medida em `/mnt/data`**: **64 TB livres** (1,6 TB ocupados de 66 TB totais).
- **Consumo total previsto do pipeline de materialização e deduplicação**:
  - `raw/`: ~40 a 50 GB;
  - `normalized/`: ~40 a 60 GB;
  - `deduplicated/` (após descarte de repetidos e boilerplate): ~35 a 50 GB;
  - `scratch/` (espaço temporário para LSH buckets, shuffles e índices): ~50 a 80 GB;
  - **Volume total de pico de trabalho**: **~165 a 240 GB**.
- **Conclusão**: O orçamento de trabalho consome menos de **0,38% do espaço livre medido**, confirmando total segurança de infraestrutura sem risco de saturação do storage NFS.

---

## 8. Plano de Deduplicação e Decontaminação Global

O pipeline que transformará os dados normalizados nos manifests congelados do C1 seguirá uma sequência estritamente determinística e auditável:

```
[Fontes Primárias Normalizadas]
              ↓
(1) Normalização canônica (Unicode NFC, schema PyArrow, preservação de metadados)
              ↓
(2) Filtros de Língua e Integridade (LangID FastText, razão alfabética, poda de boilerplate)
              ↓
(3) Deduplicação Exata Global (SHA-256 canônico; primárias têm precedência sobre agregadores)
              ↓
(4) Deduplicação Aproximada Global (MinHash/LSH com janelas calibradas experimentalmente)
              ↓
(5) Decontaminação de Benchmarks (13-gram chars / 8-gram words contra test splits de avaliação)
              ↓
(6) Separação de Splits (Train 99.0% / Val 0.5% / Test 0.5% estritamente pós-dedup)
              ↓
[Manifesto Final Congelado]
```

### 8.1 Detalhamento Técnico das Etapas

1. **Normalização Canônica**:
   - Normalização Unicode para forma **NFC**.
   - Preservação estrita das fronteiras de documentos (`\n\n`), impedindo a concatenação indiscriminada de parágrafos.
   - Preservação de todos os metadados de auditoria (`source`, `source_revision`, `subset`, `original_id`, `variety`, `license`).
2. **Filtros de Língua e Integridade**:
   - Descarte de documentos quase-vazios (<100 caracteres ou <20 palavras), eliminando ruído em redes sociais (`dat`) e intervenções vazias de plenário.
   - Poda de razão alfabética $< 60\%$ (`isalpha() / len(text)`).
   - Validação de idioma por FastText (`lid.176.bin`), exigindo $p(\text{pt}) \ge 0,75$.
   - Poda de documentos com $\ge 30\%$ de linhas redundantes (menus web repetidos).
3. **Deduplicação Exata Global**:
   - Hash SHA-256 sobre texto normalizado.
   - **Regra de precedência estrita**: Se um hash coincidir entre uma fonte primária curada e um agregador (ex: Wikipédia vs. subconjunto `wikipedia` do GigaVerbo; ou Carolina vs. `finepdfs`), **a fonte primária é mantida e a cópia no agregador é descartada**, marcando-se `CROSS_SOURCE_EXACT_DROP`.
   - No ParlamentoPT, fórmulas burocráticas recorrentes idênticas são colapsadas para uma única ocorrência de pretraining.
4. **Deduplicação Aproximada Global (MinHash / LSH)**:
   - Shingling por 5-gramas de palavras ($k=5$).
   - 128 permutações MinHash com 16 bandas de 8 linhas ($b=16, r=8$, limiar teórico $\approx 0,707$).
   - Limiares de Jaccard: **0,80** para pares entre fontes (*cross-source*) e **0,85** para pares internos (*within-source*). A fonte curada tem precedência sobre a fonte web residual.
5. **Decontaminação de Benchmarks**:
   - Confrontação exata de 13-gramas de caracteres e 8-gramas de palavras contra os conjuntos de avaliação congelados (BLiMP-PT, ENEM, FaQuAD, ASSIN 2, ARC-PT, MMLU-PT, HateBR).
   - Textos contaminantes são purgados de todos os splits de treino sob o rótulo `BENCHMARK_CONTAMINATED`.
6. **Divisão de Splits (Train / Validation / Test)**:
   - Executada **estritamente após** a deduplicação global e a decontaminação.
   - Particionamento determinístico por hash: **99,0% Treino**, **0,5% Validação** e **0,5% Teste**, estratificado por fonte.

---

## 9. Incertezas Científicas e Protocolo de Escolha entre A, B e C

### 9.1 Incertezas Científicas Não Resolvidas
1. **Tradução Automática Latente em Web Crawls**:
   Embora os subsets conhecidos tenham sido eliminados, web crawls como `fineweb_2_pt` e `mc4_pt` carregam traduções neurais não sinalizadas.
2. **Ausência de Rotulagem Fina de Variedades Pluricêntricas**:
   A Wikipédia e os crawls não rotulam nativamente a variante nacional em nível de documento.
3. **Calibração de Qualidade Inter-Subconjuntos**:
   O escore de qualidade do GigaVerbo reflete preferências do classificador upstream e não foi calibrado em modelos de 180M parâmetros.
4. **Fertilidade de Tokenização Dependente do Gate C2**:
   A conversão precisa de volume em palavras para orçamento de tokens vistos depende do tokenizer que será congelado no C2.

### 9.2 Protocolo Experimental para Escolha da Mistura Vencedora
A escolha final entre as Famílias A, B e C ocorrerá por comparação empírica de corridas curtas no cluster Orion:
- **Modelo e Tokenizer Fixados**: Mesma arquitetura causal de ~180M parâmetros (C3) e mesmo tokenizer (C2);
- **Alocação de Hardware**: 1 GPU NVIDIA H100 (~40 GB VRAM);
- **Orçamento de Treino**: 2,0 bilhões de tokens vistos (~11 horas por variante);
- **Critérios de Decisão**:
  1. Perplexidade de validação por domínio (Wikipédia, Carolina, ParlamentoPT, Gutenberg, FinePDFs e Web);
  2. Acurácia em sondagens sintáticas e de concordância do BLiMP-PT;
  3. Estabilidade do gradiente e ausência de divergência ou perda de representação.

---

## 10. Decisão do Gate C1 e Próximo Passo

### 10.1 Avaliação de Prontidão
O estudo de composição alcançou todas as definições conceituais, metodológicas e empíricas necessárias:
- [x] Unidade de mistura formalizada como volume de palavras normalizadas (C1 proxy);
- [x] Alvos canônicos das três famílias estabelecidos somando exatamente 100,0%;
- [x] Estrutura interna do GigaVerbo particionada em quatro baldes sem dupla contagem;
- [x] Política de fronteiras de documentos e preservação de identidade explicitada;
- [x] Capacidade e espaço livre do storage `/mnt/data` medidos e verificados;
- [x] Plano de materialização seletiva sem downloads cegos estruturado;
- [x] Fontes de segunda rodada avaliadas e mantidas em estado DEFER.

**Veredito de Prontidão**: **READY** para a criação dos arquivos canônicos de configuração:
- `configs/corpus_mix_a.yaml`
- `configs/corpus_mix_b.yaml`
- `configs/corpus_mix_c.yaml`

### 10.2 Próximo Passo Concreto
Quando solicitado:
1. Criar os três arquivos declarativos `configs/corpus_mix_*.yaml` contendo os pesos provisórios em palavras, revisões pinned e regras de baldes do GigaVerbo;
2. Iniciar a materialização seletiva das fontes primárias em `/mnt/data/cambacica-base-180m/raw/` e a extração filtrada dos row groups elegíveis do GigaVerbo-v2;
3. Gerar os manifests normalizados correspondentes e executar o pipeline de deduplicação global exata e aproximada.
