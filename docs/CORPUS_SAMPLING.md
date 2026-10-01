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
orçamentos de tokens ou pesos entre fontes para um modelo causal de $\le$ 200M
parâmetros, precisamos auditar o conteúdo real de cada repositório.

Amostragens preliminares permitem:
1. Validar taxonomias e metadados sem baixar centenas de gigabytes;
2. Medir distribuições de comprimento de documento e taxas de caracteres alfabéticos;
3. Auditar a presença de tradução automática identificável e resíduos multilíngues;
4. Identificar overlaps e duplicatas exatas entre fontes primárias e coleções agregadoras;
5. Preparar dados para o Gate C2 (treinamento e análise de tokenizer).

---

## 2. Princípio científico: Representativo vs. Diagnóstico

A pipeline faz uma distinção estrita e obrigatória entre dois modos de amostragem:

### A. Amostra Representativa (`representative`)
- **Objetivo:** Amostrar a distribuição real e proporções naturais do repositório upstream.
- **Mecanismo:** Amostragem determinística baseada no hash estável do identificador (`stable_hash64(doc_id, seed)`).
- **Semântica e Escopo da População:**
  * Quando a população total pode ser enumerada sem transferência excessiva (como o catálogo de literatura do Gutenberg com ~655–787 obras), o reservatório opera sobre **todo o catálogo**, sendo genuinamente representativo do repositório completo.
  * Quando a fonte é massiva e o streaming completo consumiria gigabytes/terabytes desnecessários (como Wikipédia com 1,18M artigos ou ParlamentoPT com 11,5M intervenções), o sampler opera sobre um **`bounded_stream_prefix`** (prefixo delimitado do fluxo). Esse limite de varredura é explicitamente registrado no manifesto (`sampling_frame`) e jamais descrito como uma amostra globalmente representativa de todo o histórico.
  * Para fontes particionadas em múltiplos shards (como GigaVerbo com 56 shards e Carolina com 823 arquivos), a amostragem adota uma moldura determinística entre partições (`deterministic_shard_subsample` ou `deterministic_shard_partition`), distribuindo a colheita entre múltiplos shards e taxonomias.

### B. Amostra Diagnóstica Estratificada (`diagnostic`)
- **Objetivo:** Forçar a exposição de minorias linguísticas, jurídicas, legislativas ou acadêmicas para inspeção qualitativa humana.
- **Mecanismo:** Amostragem estratificada (`StratifiedSampler`) com cotas mínimas fixadas por taxonomia/domínio.
- **Regra:** Amostras representativas e diagnósticas **nunca devem ser mescladas ou confundidas**.

---

## 3. Fontes suportadas nesta fase

| Fonte | Identificador Upstream | Modos Disponíveis | Padrão (Docs) | Estratégia de Acesso e Moldura |
| :--- | :--- | :--- | :--- | :--- |
| **Corpus Carolina** | `carolina-c4ai/corpus-carolina` | `representative`, `diagnostic` | 10.000 / 10.000 | Streaming XML TEI P5 (`deterministic_shard_partition` entre 7 taxonomias: `wik`, `dat`, `jud`, `leg`, `uni`, `soc`, `pub`) |
| **Wikipédia em português** | `wikimedia/wikipedia` (`20231101.pt`) | `representative` | 10.000 | Streaming via Hugging Face Hub sobre `bounded_stream_prefix` (primeiros 30.000 artigos) |
| **ParlamentoPT** | `PORTULAN/parlamento-pt` | `representative` | 10.000 | Streaming HTTP sobre `bounded_stream_prefix` (primeiras 40.000 linhas); preserva falas curtas por padrão |
| **GigaVerbo-v2** | `Polygl0t/gigaverbo-v2` (`edu_high`) | `audit`, `candidate` | 25.000 / 25.000 | `deterministic_shard_subsample` (8 de 56 shards Parquet uniformemente distribuídos) com exclusão versionada |
| **Literatura em Domínio Público** | `project_gutenberg_pt` | `representative` | ~100 obras | `full_catalog_hash_reservoir` sobre todo o catálogo de ~655 obras PT; remoção de boilerplate |

### Fontes em Segunda Rodada ou Fora de Escopo
- **Ulysses / Tesemõ:** Segunda rodada. Exige espelhamento local prévio a partir dos arquivos do Google Drive, pois o repositório git upstream contém apenas documentação.
- **SciELO:** Segunda rodada. Exige rotina de colheita estruturada via ArticleMeta API / OAI-PMH.
- **BDTD e FineWeb-2:** Fora de escopo inicial do C1. O BDTD descentralizado em ~130 universidades impõe sobrecarga de PDF/OCR excessiva para esta fase.

---

## 4. Política do GigaVerbo-v2: Audit vs. Candidate

O `Polygl0t/gigaverbo-v2` é tratado sob dois modos mutuamente exclusivos:

1. **`gigaverbo_audit` (`--mode audit`):**
   - Amostra diretamente da partição `edu_high` sem descartar nenhum subset upstream.
   - Permite inspecionar o que efetivamente existe no dataset original (inclusive resíduos traduzidos ou não comerciais).

2. **`gigaverbo_candidate` (`--mode candidate`):**
   - Aplica a política de exclusão versionada em [`configs/gigaverbo_exclusions.yaml`](../configs/gigaverbo_exclusions.yaml).
   - Descarta sistematicamente subsets deliberadamente traduzidos por máquina (`dolly-15k-libretranslate-pt`, `Bactrian-X`, `UltrachatBR`, `cosmos_qa_ptbr`, `gpt4all`), subsets com licença não comercial (`xlsum`, `Bactrian-X`) e duplicações de fontes já presentes no Cambacica (`corpus-carolina`, `wikipedia`, `bdtd`, `baixelivros`).
   - Visita múltiplos shards Parquet distribuídos ao longo da partição `edu_high`.

> [!WARNING]
> A exclusão por metadados filtra apenas subsets catalogados. A presença de tradução automática latente dentro de scrapes gerais da web (Common Crawl) permanece como uma incógnita mensurável.

---

## 5. Esquema unificado (Parquet)

Todos os extratores normalizam seus registros no esquema PyArrow canônico de 14 campos:

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `text` | string (não-nulo) | Texto normalizado em Unicode NFC com espaços aparados |
| `source` | string (não-nulo) | Identificador canônico da fonte (`carolina`, `wikipedia_pt`, etc.) |
| `source_revision` | string (nulo) | Versão upstream (ex: `v2.0.1`, `20231101.pt`, data de snapshot) |
| `subset` | string (nulo) | Sub-partição upstream (ex: taxonomia `jud`, subset `fineweb_2_pt`) |
| `original_id` | string (nulo) | Identificador original no dataset upstream |
| `original_url` | string (nulo) | URL canônica do documento |
| `license` | string (nulo) | Licença aplicável (estritamente preservada conforme upstream; `None` se não rotulado linha a linha) |
| `language` | string (nulo) | Rótulo de idioma documentado (ex: `pt`, `pt-BR`, `pt-PT`) |
| `language_score` | float32 (nulo) | Confiança do detector de idioma upstream |
| `variety` | string (nulo) | Variedade linguisticamente documentada (`pt-BR`, `pt-PT`) |
| `quality_score` | float32 (nulo) | Escore educacional ou de qualidade upstream |
| `publication_date` | string (nulo) | Data ou ano de publicação original |
| `domain_category` | string (nulo) | Classificação temática ou tipológica |
| `content_sha256` | string (não-nulo) | Hash SHA-256 do texto normalizado calculado localmente |

Metadados ausentes são representados estritamente como nulos (`None`), evitando a fabricação de dados. Em particular:
- Registros do GigaVerbo possuem `license = None` (já que o repositório upstream não fornece licença por linha individual).
- Registros do Gutenberg são anotados como `"Project Gutenberg License / US Public Domain (jurisdiction-dependent)"`, evitando a generalização indevida para domínio público mundial irrestrito.
- Registros do Carolina preservam a licença declarada no cabeçalho TEI ou `"Unspecified / Source-specific (see TEI header)"`.

---

## 6. Manifesto de proveniência

Cada execução grava um arquivo `manifest_<mode>.json` ao lado do `.parquet`, registrando:
- Fonte, modo, tamanho solicitado e contagem real de documentos;
- Seed determinística;
- Identificador upstream, revisão legível e commit SHA exato e imutável (quando disponível);
- `population_scope`: estimativa ou contagem total da população upstream;
- `sampling_frame`: descrição explícita da moldura amostral (ex: prefixo delimitado vs. reservatório do catálogo integral);
- `records_examined`: total de registros avaliados no upstream durante a varredura;
- `bytes_read`: bytes transferidos pela rede ou lidos em disco;
- `stopping_reason`: motivo de finalização da colheita;
- Hash SHA-256 do arquivo de configuração de exclusões (quando aplicável);
- Commit Git do código executor local;
- Checksum SHA-256 e tamanho em bytes do arquivo Parquet gerado;
- Estatísticas agregadas de caracteres, palavras e listas de IDs (ex: `selected_ebook_ids`).

---

## 7. Como executar

### Planejamento e Dry-Run (Visibilidade sem Download)

É possível planejar a amostragem antes de baixar qualquer dado utilizando o argumento `--dry-run`:

```bash
# Verificar plano, revisão resolvida, moldura amostral e limites de segurança
python3 -m cambacica.corpus sample carolina --dry-run
python3 -m cambacica.corpus sample gigaverbo_v2 --mode candidate --dry-run
python3 -m cambacica.corpus sample parlamento_pt --dry-run
```

### Amostragem Efetiva

A amostragem pode ser invocada via módulo Python ou pelo utilitário em `scripts/`:

```bash
# Corpus Carolina — Amostra representativa (10.000 documentos)
python3 -m cambacica.corpus sample carolina --mode representative --size 10000 --seed 42

# Corpus Carolina — Amostra diagnóstica estratificada (10.000 documentos)
python3 -m cambacica.corpus sample carolina --mode diagnostic --size 10000 --seed 42

# Wikipédia em português — Amostra representativa (10.000 artigos)
python3 -m cambacica.corpus sample wikipedia_pt --mode representative --size 10000 --seed 42

# ParlamentoPT — Amostra representativa de debates em PT-PT (10.000 documentos)
# Por padrão, preserva todas as falas válidas (min_length=0). Filtro de comprimento opcional via --min-length
python3 -m cambacica.corpus sample parlamento_pt --mode representative --size 10000 --seed 42

# GigaVerbo-v2 — Amostra candidata filtrada (25.000 documentos)
python3 -m cambacica.corpus sample gigaverbo_v2 --mode candidate --size 25000 --seed 42

# GigaVerbo-v2 — Amostra de auditoria sem filtros (25.000 documentos)
python3 -m cambacica.corpus sample gigaverbo_v2 --mode audit --size 25000 --seed 42

# Literatura em Domínio Público — Gutenberg PT (~100 obras completas)
python3 -m cambacica.corpus sample gutenberg_pt --mode representative --size 100 --seed 42
```

### Inspeção diagnóstica

Para analisar métricas de comprimento, taxa de caracteres alfabéticos, duplicatas exatas internas e distribuição de metadados:

```bash
# Inspecionar uma amostra específica
python3 -m cambacica.corpus inspect data/samples/gate_c1/carolina/representative.parquet

# Inspecionar todas as amostras existentes no diretório
python3 -m cambacica.corpus inspect data/samples/gate_c1/

# Obter relatório em formato JSON estruturado
python3 -m cambacica.corpus inspect data/samples/gate_c1/carolina/representative.parquet --json
```

### Comparação inter-fontes (Detecção de Overlap)

Para verificar colisão de documentos idênticos entre fontes diferentes (por exemplo, documentos do Carolina presentes dentro do GigaVerbo):

```bash
# Diagnóstico de duplicatas exatas entre amostras
python3 -m cambacica.corpus compare data/samples/gate_c1/

# Diagnóstico de quase-duplicatas via MinHash LSH (Jaccard >= 0.80)
python3 -m cambacica.corpus compare data/samples/gate_c1/ --minhash --threshold 0.80
```

---

## 8. Limitações atuais

1. **Tradução latente em dados web:** O filtro de exclusão do GigaVerbo remove datasets catalogados de tradução automática, mas não detecta eventuais traduções não sinalizadas presentes no Common Crawl.
2. **Distribuição dialetal em fontes pluricêntricas:** A Wikipédia em português não rotula artigos por variedade (PT-BR vs. PT-PT vs. PALOP); a identificação dialetal precisa ser avaliada experimentalmente.
3. **Prefixos delimitados de fluxo:** As amostragens representativas sobre Wikipédia e ParlamentoPT limitam a varredura a um prefixo inicial do fluxo de streaming para evitar transferência desproporcional. Essas amostras representam a distribuição desse prefixo sequencial, não uma seleção pseudo-aleatória sobre a totalidade histórica absoluta das fontes.
