# Gates do projeto Cambacica

Os gates existem para separar decisões científicas de otimizações prematuras e
para tornar o treinamento reproduzível.

## C0 — Escopo científico e infraestrutura

**Objetivo:** congelar o problema que estamos tentando resolver.

Critérios de saída:

- nome e identidade do modelo definidos;
- português nativo como requisito;
- faixa de 150–200M parâmetros definida;
- alvo nominal de ~180M;
- aproximadamente 40 GB de H100 como infraestrutura-alvo;
- single-GPU como configuração principal;
- modelo base como escopo;
- ausência de compromisso com modelos maiores explicitada.

Status: **PASS**

## C1 — Corpus

**Objetivo:** definir quais dados representam o português do Cambacica.

Deve incluir:

- inventário de fontes;
- licenças e restrições;
- detecção de língua;
- filtragem de qualidade;
- deduplicação;
- metadados de proveniência;
- análise de variantes do português;
- estimativa de tokens;
- conjunto de validação separado do treino.

Progresso atual:

- inventário de fontes concluído;
- pipeline de amostragem determinística e inspeção diagnóstica implementado (`docs/CORPUS_SAMPLING.md`);
- resultados da primeira rodada de caracterização registrados (`docs/C1_SAMPLING_RESULTS.md`);
- infraestrutura de sampling considerada estável para o escopo atual;
- estudo de composição do corpus iniciado (`docs/C1_CORPUS_COMPOSITION_STUDY.md`);
- análise científica de composição, caracterização das 5 famílias e propostas A/B/C concluídas (`docs/C1_COMPOSITION_ANALYSIS.md`);
- especificações de misturas candidatas A/B/C versionadas como hipóteses experimentais provisórias (`configs/corpus_mix_*.yaml`);
- plano de materialização seletiva em storage `/mnt/data` documentado (`configs/corpus_materialization.yaml`);
- materialização bruta das fontes concluída e verificada:
  - Gutenberg raw materialization: COMPLETE
  - ParlamentoPT raw materialization: COMPLETE
  - Wikipedia PT raw materialization: COMPLETE
  - Carolina raw materialization: COMPLETE
  - GigaVerbo residual raw materialization: COMPLETE
    - measured persisted size: 41,543,747,437 bytes
    - 15,756,679 eligible/persisted records
    - 2,260 Parquet payload files
    - production manifest SHA-256: `358003af2241583ae353647f04601be1f00b096007476742a715e485b2a1ca14`
- normalização de produção e caracterização do C1 concluídas; etapa atual: implementação e piloto da deduplicação exata.

### Normalização e caracterização de produção do C1

- contrato congelado: [`C1_NORMALIZATION_SPEC.md`](C1_NORMALIZATION_SPEC.md), versão 1.0.0;
- normalização de produção **COMPLETE** e verificada para as cinco fontes no commit `ccf365279573bc5a443ac944785ece8e4f63fad2`;
- caracterização **COMPLETE**; volume bruto total de **21.510.183.558 palavras normalizadas**;
- capacidades máximas brutas, antes de deduplicação: A **263.186.287** palavras (Gutenberg), B **526.372.575** (Gutenberg), C **423.812.666** (GigaVerbo `blogset`);
- contrato exato versionado em [`C1_EXACT_DEDUP_SPEC.md`](C1_EXACT_DEDUP_SPEC.md), versão 1.0.0;
- implementação da deduplicação exata e piloto local concluídos e verificados; dados normalizados permanecem imutáveis;
- deduplicação exata de produção: **NOT RUN / NEXT**;
- near dedup: **PENDING**; limiares aguardam piloto representativo;
- decontaminação de benchmarks: **PENDING**; inventário e regra de matching ainda não definidos;
- o Gate C1 continua **IN PROGRESS**; C2 permanece **PENDING**.

Próximos critérios de C1:

- selecionar fontes finais e revisões pinned;
- congelar políticas de idioma, qualidade e licença;
- executar deduplicação exata de produção e verificar os outputs;
- calibrar e executar near dedup após piloto representativo;
- definir inventário e executar decontaminação de benchmarks;
- comparar composições candidatas do corpus;
- definir split train/validation/test reproduzível;
- produzir manifests e estatísticas finais para a interface com C2.

Status: **IN PROGRESS**

## C2 — Tokenizer

**Objetivo:** escolher e congelar um tokenizer treinado para o corpus
português.

Deve comparar configurações candidatas usando métricas objetivas de
fragmentação e cobertura.

Status: **PENDING**

## C3 — Arquitetura e benchmark na Orion

**Objetivo:** escolher a arquitetura final dentro da faixa de 150–200M.

Deve medir variantes candidatas na fração disponível da H100.

Status: **PENDING**

## C4 — Pipeline de dados e treinamento

**Objetivo:** validar packing, batching, checkpointing, retomada, logs e
reprodutibilidade.

Status: **PENDING**

## C5 — Smoke training

**Objetivo:** provar que a receita completa aprende de forma estável em uma
execução curta e descartável.

Status: **PENDING**

## C6 — Treinamento principal

**Objetivo:** executar a configuração científica congelada.

A quantidade final de tokens de pretraining ainda não está definida e deverá
ser decidida a partir dos resultados dos gates anteriores.

Status: **PENDING**

## C7 — Avaliação

**Objetivo:** caracterizar o modelo final em português e documentar suas
limitações.

Status: **PENDING**
