# Sampling and Inspection Pipeline — Gate C1 (CORPUS)

Este documento descreve a infraestrutura de amostragem determinística,
inspeção diagnóstica e análise de duplicatas implementada para o Gate C1 do
modelo `cambacica-base-180m`.

> [!IMPORTANT]
> **Estas amostras NÃO constituem o corpus final de treinamento.**
> O propósito desta pipeline é fornecer amostras pequenas, reproduzíveis e
> não destrutivas para caracterização científica, calibração de filtros e
> medições reais na Orion antes de qualquer congelamento de manifesto ou
> treinamento de tokenizer.

---

## 1. Por que a amostragem existe

Conforme estabelecido em [`docs/CORPUS.md`](CORPUS.md), o Cambacica prioriza
qualidade sobre volume e proveniência sobre conveniência. Antes de decidir
orçamentos de tokens ou pesos entre fontes para um modelo causal de <= 200M
parâmetros, precisamos auditar o conteúdo real de cada repositório.

Amostragens preliminares permitem:

1. validar taxonomias e metadados sem baixar centenas de gigabytes;
2. medir distribuições de comprimento e taxas de caracteres alfabéticos;
3. auditar tradução automática identificável e resíduos multilíngues;
4. identificar overlaps e duplicatas entre fontes primárias e agregadores;
5. preparar dados para o Gate C2.

---

## 2. Princípio científico: representativo vs. diagnóstico

A pipeline distingue explicitamente dois usos.

### A. Amostra representativa (`representative`)

O objetivo é aproximar a distribuição natural da população definida pelo
sampler, sempre registrando a moldura amostral real no manifesto.

Quando a população completa pode ser enumerada de forma barata, a amostragem
pode operar sobre todo o catálogo. Quando isso não é viável, o manifesto deve
identificar explicitamente o limite, por exemplo `bounded_stream_prefix`.

No Corpus Carolina, a alocação representativa entre taxonomias usa as contagens
documentadas do v2.0.1 e largest remainder determinístico.

### B. Amostra diagnóstica (`diagnostic`, `audit`)

O objetivo é expor domínios minoritários, anomalias ou diversidade física do
upstream para inspeção. Ela não deve ser interpretada automaticamente como
estimativa estatística da população completa.

Amostras representativas e diagnósticas nunca devem ser confundidas ou
concatenadas como se fossem o corpus final.

---

## 3. Fontes suportadas nesta fase

| Fonte | Identificador upstream | Modos | Padrão | Moldura atual |
| --- | --- | --- | ---: | --- |
| Corpus Carolina | `carolina-c4ai/corpus-carolina` | `representative`, `diagnostic` | 10.000 / 10.000 | alocação proporcional por taxonomia + shards distribuídos; cotas explícitas no modo diagnóstico |
| Wikipédia PT | `wikimedia/wikipedia` (`20231101.pt`) | `representative` | 10.000 | `bounded_stream_prefix` sobre até 30.000 artigos |
| ParlamentoPT | `PORTULAN/parlamento-pt` | `representative` | 10.000 | `bounded_stream_prefix` sobre até 40.000 linhas; falas curtas preservadas |
| GigaVerbo-v2 | `Polygl0t/gigaverbo-v2` (`edu_high`) | `audit`, `candidate` | 25.000 / 25.000 | todos os 56 shards, com row groups deterministicamente distribuídos; exclusões aplicadas antes do reservatório no modo candidate |
| Project Gutenberg PT | catálogo em português | `representative` | ~100 obras | reservatório determinístico sobre o catálogo elegível, com remoção de boilerplate |

Fontes de segunda rodada: Ulysses/Tesemõ e SciELO. BDTD direto e FineWeb-2
direto permanecem fora do caminho principal enquanto não houver uma necessidade
científica concreta.

---

## 4. Política do GigaVerbo-v2: audit vs. candidate

O GigaVerbo-v2 é tratado como reservatório, não como corpus a ser concatenado
integralmente.

A revisão pinned usada no C1 é:

`Polygl0t/gigaverbo-v2@7058ccf19eaeaf4505a96fc7e5305a01fc441fd8`

Os Parquets apresentam clustering físico por subset. Por isso, visitar muitos
shards mas ler apenas seus primeiros registros produz uma visão enviesada.

### `audit`

- visita todos os 56 shards;
- seleciona row groups distribuídos entre regiões físicas iniciais,
  intermediárias e finais;
- não aplica exclusões;
- existe para revelar a diversidade física e validar as hipóteses sobre o
  upstream.

### `candidate`

- usa a mesma cobertura distribuída dos 56 shards;
- aplica [`configs/gigaverbo_exclusions.yaml`](../configs/gigaverbo_exclusions.yaml)
  antes da inserção no `DeterministicReservoirSampler`;
- registra contagens por subset antes da exclusão, exclusões efetivas, registros
  elegíveis e composição final do reservatório.

O manifesto mantém as seguintes invariantes:

```text
sum(records_encountered_per_subset) == records_examined
eligible_records == records_examined - total_excluded
sum(eligible_records_per_subset) == eligible_records
sum(retained_sample_per_subset) == final_sample_size
```

A auditoria da revisão pinned confirmou subsets explicitamente traduzidos,
sintéticos, não comerciais ou sobrepostos a fontes primárias, incluindo
`ultrachat`, `bactrianx`, `cosmos_qa`, `gpt4all`, `xlsum`, `wikipedia` e
`corpus_carolina`.

> [!WARNING]
> Exclusão por metadata remove apenas subsets conhecidos. Tradução automática
> latente dentro de web crawls continua sendo uma incerteza do C1.

---

## 5. Esquema unificado

Todos os extratores normalizam registros em um esquema PyArrow comum contendo:

- `text`;
- `source`;
- `source_revision`;
- `subset`;
- `original_id`;
- `original_url`;
- `license`;
- `language`;
- `language_score`;
- `variety`;
- `quality_score`;
- `publication_date`;
- `domain_category`;
- `content_sha256`.

Metadados ausentes permanecem nulos; o pipeline não fabrica valores.

---

## 6. Manifesto de proveniência

Cada execução registra, quando aplicável:

- fonte, modo, seed e tamanho solicitado/obtido;
- identificador upstream e revisão imutável;
- `population_scope` e `sampling_frame`;
- `records_examined`, `bytes_read` e `stopping_reason`;
- shards e row groups selecionados;
- hash da configuração de exclusões;
- commit Git do código local;
- checksum do Parquet gerado;
- estatísticas agregadas e contagens por subset.

No GigaVerbo candidate são registrados explicitamente:

- `records_encountered_per_subset`;
- `exclusion_counts_by_subset`;
- `eligible_records_per_subset`;
- `retained_sample_per_subset`;
- `exclusion_rules_matched`;
- `total_excluded`.

---

## 7. CLI

### Dry-run

```bash
python3 -m cambacica.corpus sample carolina --dry-run
python3 -m cambacica.corpus sample gigaverbo_v2 --mode candidate --dry-run
python3 -m cambacica.corpus sample parlamento_pt --dry-run
```

### Amostragem

```bash
python3 -m cambacica.corpus sample carolina --mode representative --size 10000 --seed 42
python3 -m cambacica.corpus sample carolina --mode diagnostic --size 10000 --seed 42
python3 -m cambacica.corpus sample wikipedia_pt --mode representative --size 10000 --seed 42
python3 -m cambacica.corpus sample parlamento_pt --mode representative --size 10000 --seed 42
python3 -m cambacica.corpus sample gigaverbo_v2 --mode audit --size 25000 --seed 42
python3 -m cambacica.corpus sample gigaverbo_v2 --mode candidate --size 25000 --seed 42
python3 -m cambacica.corpus sample gutenberg_pt --mode representative --size 100 --seed 42
```

### Inspeção e overlap

```bash
python3 -m cambacica.corpus inspect data/samples/gate_c1/
python3 -m cambacica.corpus compare data/samples/gate_c1/
python3 -m cambacica.corpus compare data/samples/gate_c1/ --minhash --threshold 0.80
```

A descoberta de arquivos persistentes ignora artefatos de smoke/test/temp; esses
artefatos devem ser gravados fora da árvore científica persistente.

---

## 8. Semântica de duplicação

O relatório de comparação separa:

- `WITHIN_FILE`: repetições dentro do mesmo sample;
- `SAME_SOURCE_CROSS_MODE`: overlap esperado entre modos da mesma fonte;
- `CROSS_SOURCE`: colisões entre fontes upstream diferentes.

A mesma classificação é aplicada aos diagnósticos de near-duplicate por
MinHash. Resultados nulos devem ser reportados explicitamente.

---

## 9. Limitações atuais

1. **Tradução latente em web:** o filtro do GigaVerbo remove subsets catalogados,
   mas não prova que scrapes gerais estejam livres de tradução.
2. **Dialetos em fontes pluricêntricas:** Wikipédia não fornece PT-BR/PT-PT/PALOP
   por artigo.
3. **Prefixos delimitados:** Wikipédia e ParlamentoPT ainda usam frames de
   prefixo limitado para a amostragem de inspeção.
4. **Amostras não substituem deduplicação global:** ausência de overlap nas
   amostras não demonstra ausência na população completa.
5. **Frequências do GigaVerbo diagnostic não são pesos de corpus:** a seleção de
   row groups é desenhada para cobertura física reproduzível, não para estimar
   imparcialmente a prevalência global de cada subset.

Resultados científicos da rodada atual são registrados em
[`docs/C1_SAMPLING_RESULTS.md`](C1_SAMPLING_RESULTS.md).