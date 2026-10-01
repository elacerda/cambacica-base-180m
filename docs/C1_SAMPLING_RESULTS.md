# Gate C1 — resultados da amostragem

## Status

Gate: **C1 — corpus**

Status: **IN PROGRESS**

Este documento registra os principais resultados científicos obtidos com a
infraestrutura de amostragem e inspeção do Cambacica após o hardening concluído
no commit `68b2633`.

As amostras descritas aqui **não constituem o corpus final de treinamento**.
Elas servem para caracterizar fontes, validar filtros, revelar anomalias e
orientar a composição final do corpus.

## 1. Estado da infraestrutura

A infraestrutura C1 de amostragem está considerada estável para a etapa atual.
Ela inclui:

- amostragem determinística e manifests de proveniência;
- distinção entre amostras representativas e diagnósticas;
- exclusões versionadas para o GigaVerbo-v2;
- detecção de duplicatas exatas;
- diagnóstico de quase-duplicatas por MinHash;
- classificação de overlaps em `WITHIN_FILE`, `SAME_SOURCE_CROSS_MODE` e
  `CROSS_SOURCE`;
- isolamento de artefatos de smoke test;
- testes unitários e testes opcionais de rede.

O conjunto persistente de inspeção possui sete arquivos:

1. `carolina/representative.parquet`;
2. `carolina/diagnostic.parquet`;
3. `wikipedia_pt/representative.parquet`;
4. `parlamento_pt/representative.parquet`;
5. `gutenberg_pt/representative.parquet`;
6. `gigaverbo_v2/audit.parquet`;
7. `gigaverbo_v2/candidate.parquet`.

## 2. Corpus Carolina

A amostra representativa usa as contagens documentadas do Corpus Carolina
v2.0.1, totalizando 2.108.999 documentos:

| Taxonomia | População | Fração aproximada | Alocação em 10.000 |
| --- | ---: | ---: | ---: |
| `dat` | 1.074.032 | 50,93% | 5.093 |
| `wik` | 957.501 | 45,40% | 4.540 |
| `jud` | 38.187 | 1,81% | 181 |
| `uni` | 26.409 | 1,25% | 125 |
| `soc` | 8.862 | 0,42% | 42 |
| `leg` | 3.982 | 0,19% | 19 |
| `pub` | 26 | ~0,001% | 0 |

A alocação usa largest remainder de forma determinística e não força presença
mínima de taxonomias raras no modo representativo.

A amostra diagnóstica mostrou que os documentos extremamente longos de `leg`
são registros upstream legítimos, e não um erro de parsing. O maior caso
inspecionado corresponde a material legislativo publicado como um único registro
TEI. Portanto, nenhuma regra de corte por comprimento foi introduzida apenas por
causa desse outlier.

## 3. GigaVerbo-v2

A revisão usada no C1 é:

`Polygl0t/gigaverbo-v2@7058ccf19eaeaf4505a96fc7e5305a01fc441fd8`

A partição `edu_high` dessa revisão contém 16.245.599 registros em 56 shards e
19 valores distintos de `subset` observados na auditoria de metadados.

Entre eles existem subsets que não devem entrar deliberadamente no corpus
principal do Cambacica, incluindo conteúdo de tradução automática, sintético,
não comercial ou sobreposto a fontes primárias já representadas. Exemplos
confirmados incluem:

- `ultrachat`;
- `bactrianx`;
- `cosmos_qa`;
- `gpt4all`;
- `xlsum`;
- `wikipedia`;
- `corpus_carolina`.

### 3.1 Frame de amostragem

Os Parquets do GigaVerbo apresentam clustering físico por subset. Ler apenas o
início de cada shard produzia uma visão enviesada da partição. O sampler atual
visita todos os 56 shards e seleciona deterministicamente row groups distribuídos
entre regiões iniciais, intermediárias e finais.

O modo `audit` preserva os registros encontrados. O modo `candidate` aplica as
exclusões antes da entrada no reservatório determinístico.

### 3.2 Candidate final de 25.000 documentos

Na execução final:

- shards visitados: 56/56;
- row groups visitados: 224;
- registros examinados: 104.359;
- registros excluídos: 55.519;
- pool elegível examinado: 48.840;
- documentos retidos: 25.000.

Os 55.519 registros excluídos nessa execução pertenciam ao subset `ultrachat`.
A composição dos 25.000 documentos retidos foi:

| Subset | Documentos |
| --- | ---: |
| `fineweb_2_pt` | 8.176 |
| `hplt2_pt` | 7.113 |
| `mc4_pt` | 4.494 |
| `finepdfs_por_Latn` | 1.817 |
| `hplt1_pt` | 1.112 |
| `common_crawl` | 912 |
| `crawlPT_dedup` | 679 |
| `quati` | 267 |
| `oscar` | 197 |
| `blogset` | 121 |
| `culturax` | 112 |

Essas proporções **não devem ser interpretadas como estimativa imparcial da
composição global do `edu_high`**. O objetivo da amostragem distribuída é expor
diversidade física e validar a política de exclusão mantendo transferência de
dados limitada.

## 4. ParlamentoPT

A amostra de 10.000 documentos apresentou 1.145 ocorrências redundantes exatas
(11,45%). A investigação mostrou que a grande maioria da repetição vem de
fórmulas editoriais e institucionais curtas, como cabeçalhos de sessão, avisos de
OCR e informações de imprensa oficial.

Não foram encontrados discursos longos inesperadamente duplicados na inspeção.
Essas repetições foram mantidas na amostra representativa porque caracterizam a
fonte; a política do corpus final será aplicada na etapa de deduplicação.

## 5. Literatura do Project Gutenberg

O maior documento da amostra foi o ebook 31552, *Novo Diccionário da Língua
Portuguesa*, de Cândido de Figueiredo, com aproximadamente 1,85 milhão de
palavras.

A investigação confirmou que se trata de uma obra legítima e completa, e não de
concatenação acidental ou erro de aquisição. Nenhum corte arbitrário de
comprimento foi introduzido.

## 6. Overlap entre fontes

Na comparação das sete amostras persistentes:

- duplicatas exatas `CROSS_SOURCE`: 0 detectadas;
- near-duplicates `CROSS_SOURCE` com MinHash/Jaccard >= 0,80: 0 detectados;
- overlaps entre modos da mesma fonte são reportados separadamente;
- repetições internas continuam sendo caracterizadas por fonte.

Esses resultados valem **somente para as amostras avaliadas**. Eles não provam
que as populações completas sejam livres de overlap. A construção do corpus
final continuará exigindo deduplicação global entre fontes.

## 7. Conclusões para C1

A etapa de engenharia da amostragem é considerada concluída para o escopo atual.
Os resultados sustentam as seguintes decisões:

- Corpus Carolina permanece um candidato forte a núcleo PT-BR de alta
  proveniência;
- Wikipédia continua útil como componente enciclopédico pluricêntrico;
- ParlamentoPT fornece PT-PT institucional, mas exige deduplicação de fórmulas
  repetitivas antes do corpus final;
- literatura em domínio público acrescenta diversidade estilística e histórica,
  com controle de licença e ortografia;
- GigaVerbo-v2 deve ser tratado como reservatório, não como corpus a ser
  concatenado integralmente;
- o residual do GigaVerbo precisa manter exclusões explícitas e deduplicação
  global contra as fontes primárias do Cambacica.

O próximo trabalho de C1 é definir e comparar composições candidatas do corpus,
não continuar ampliando a infraestrutura de sampling sem nova evidência.