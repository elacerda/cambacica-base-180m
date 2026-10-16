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
- normalização e caracterização concluídas;
- D1: **COMPLETE / PASS**; D2–D2d: **COMPLETE**;
- near-deduplication da primeira execução: **APPROVED / CLOSED**, zero remoções;
- C1-BD1: **APPROVED**; C1-BD2–BD2.6: **COMPLETE**;
- C1-BD3: **COMPLETE / VERIFIED**; C1-BD4: **COMPLETE / SCIENTIST APPROVED**;
- exclusões por contaminação de benchmarks: **ZERO**; workstream de decontaminação **CLOSED FOR FIRST TRAINING**.
- C1-S1: **COMPLETE / VERIFIED**; o split determinístico completo foi aprovado, executado e verificado independentemente para 21.603.689 documentos. Manifest SHA-256 `8b937f90d40487c643fe48baa6b5c2950713fe2e0d2d534f3e2a3c08e74dbe16`; assignments SHA-256 `ef52e3cf3033934df13e3d16c207e1783c509dff6fb91954fb8c25ab719ece1e`. Evidências e identidades publicadas em [`C1_SPLIT_AND_MIXTURE_READINESS.md`](C1_SPLIT_AND_MIXTURE_READINESS.md);
- C1-S2: **COMPLETE / VERIFIED**; seleção **PASS**, materialização **MATERIALIZED** e verificação independente **PASS**. A versão v1 tem 46.583 documentos e 19.999.380 palavras normalizadas (shortfall de 620); todas as tolerâncias de cotas por fonte e subset passaram. Identidades, checksums e ressalvas científicas estão em [`C1_TOKENIZER_SAMPLE.md`](C1_TOKENIZER_SAMPLE.md). O sample preserva a referência provisória B e não escolhe a mistura final A/B/C;
- CALAME: convenção já aprovada de **2.076 linhas de snapshot / 2.075 avaliáveis**, excluindo somente o alvo whitespace da linha 718 do denominador futuro; implementação C7 permanece futura;
- A/B/C continuam hipóteses: A curated-heavy, B balanced (referência provisória de preparação), C residual-heavy control. O vencedor será escolhido depois de experimentos comparáveis com tokenizer C2 e arquitetura C3 fixos.

### Normalização, caracterização e deduplicação exata do C1

- contrato de normalização congelado: [`C1_NORMALIZATION_SPEC.md`](C1_NORMALIZATION_SPEC.md), versão 1.0.0;
- normalização de produção **COMPLETE** e verificada para as cinco fontes no commit `ccf365279573bc5a443ac944785ece8e4f63fad2`;
- caracterização **COMPLETE**; volume bruto total de **21.510.183.558 palavras normalizadas**;
- contrato exato versionado em [`C1_EXACT_DEDUP_SPEC.md`](C1_EXACT_DEDUP_SPEC.md), versão 1.0.1;
- deduplicação exata de produção: **COMPLETE** (versão de implementação 1.0.1);
- verificação da deduplicação exata: **PASS**;
- contagens de registros: 21.649.425 de entrada, 21.603.689 retidos, 45.736 descartados elegíveis (33.748 grupos de duplicatas exatas);
- palavras normalizadas: 21.510.183.558 antes, 21.470.091.017 depois (40.092.541 removidas);
- frações de perda exata: perda documental de 0,2113% (0,00211257); perda de palavras de 0,1864% (0,00186389);
- capacidades máximas pós-exato (propostas A/B/C): A **263.186.287** palavras (Gutenberg), B **526.372.575** (Gutenberg), C **423.812.666** (GigaVerbo `blogset`), inalteradas em relação ao pré-exato;
- piloto de near dedup (D2): **COMPLETE**, resultados em [`C1_NEAR_DEDUP_PILOT.md`](C1_NEAR_DEDUP_PILOT.md) e artefatos em `/mnt/data/cambacica-base-180m/dedup-pilots/near-v1/`;
- calibração científica targeted (D2b): **COMPLETE**, resultados em [`C1_NEAR_DEDUP_CALIBRATION.md`](C1_NEAR_DEDUP_CALIBRATION.md) e artefatos em `/tmp/cambacica-base-180m/dedup-pilots/near-calibration-v1/`;
- censo near de corpus completo (D2c): **COMPLETE / PASS**, verificado para o manifesto exato pinned; relatório em [`C1_NEAR_DUP_CENSUS_D2C.md`](C1_NEAR_DUP_CENSUS_D2C.md);
- revisão científica e avaliação de política (D2d): **COMPLETE**, com packet em `/mnt/data/cambacica-base-180m/dedup-review/d2d-v1/`;
- decisão científica de near dedup para a primeira execução: **APPROVED / CLOSED**; preservar todos os registros pós-D1, com zero remoções por near dedup. A recomendação B de D2d permanece somente como direção futura. Veja [`C1_NEAR_DEDUP_FINAL_DECISION.md`](C1_NEAR_DEDUP_FINAL_DECISION.md);
- produção de near dedup: **NOT RUN BY DESIGN**; não é um blocker para a decontaminação de benchmarks;
- decontaminação de benchmarks: **BD1 APPROVED; BD2–BD2.6 COMPLETE; BD3 COMPLETE / VERIFIED; BD4 COMPLETE / SCIENTIST APPROVED**. Decisão para o primeiro treinamento: reter os 18 documentos afetados e excluir zero; corpus pós-D1 preservado. O workstream está **CLOSED FOR FIRST TRAINING**. Não é necessária outra experiência de decontaminação para essa decisão. Veja o registro BD4 [`C1_BENCHMARK_DECONTAMINATION_BD4.md`](C1_BENCHMARK_DECONTAMINATION_BD4.md), a aprovação BD3 [`C1_BD3_SCIENTIFIC_APPROVAL.md`](C1_BD3_SCIENTIFIC_APPROVAL.md), a aprovação inicial [`C1_BENCHMARK_DECONTAMINATION_APPROVAL.md`](C1_BENCHMARK_DECONTAMINATION_APPROVAL.md), os resultados BD2 [`C1_BENCHMARK_DECONTAMINATION_BD2.md`](C1_BENCHMARK_DECONTAMINATION_BD2.md), o preflight [`C1_BENCHMARK_DECONTAMINATION_PREFLIGHT.md`](C1_BENCHMARK_DECONTAMINATION_PREFLIGHT.md), a recuperação [`C1_BENCHMARK_DECONTAMINATION_RECOVERY.md`](C1_BENCHMARK_DECONTAMINATION_RECOVERY.md) e o plano [`C1_BENCHMARK_DECONTAMINATION_PLAN.md`](C1_BENCHMARK_DECONTAMINATION_PLAN.md);
- o Gate C1 continua **IN PROGRESS** enquanto direitos e políticas de uso não
  estiverem resolvidos; C2 iniciou somente após a validação limitada da
  implementação descrita em [`C2_TOKENIZER_EXPERIMENTS.md`](C2_TOKENIZER_EXPERIMENTS.md).

Próximos critérios de C1:

- confirmar a base de uso científico local quando a autorização existente não cobrir os caveats de direitos; separar revisão de direitos para publicação/redistribuição. Não há novo estudo de decontaminação pendente;
- manter os contratos e identidades publicados de C1-S1 e C1-S2; qualquer mudança de política, taxonomia ou quotas exige nova versão e aprovação científica. Palavras normalizadas não são tokens, e C1-S2 não seleciona a mistura final A/B/C;
- registrar políticas aplicáveis de fonte/idioma/qualidade/licença e capacidades reais pós-holdout, tolerâncias de cotas e qualquer reutilização explícita. A seleção do vencedor A/B/C fica para os experimentos posteriores;
- fixar a implementação futura de avaliação no C7, usando a convenção CALAME já aprovada; isso não autoriza implementação C7 neste estágio.

Status: **IN PROGRESS**

## C2 — Tokenizer

**Objetivo:** escolher e congelar um tokenizer treinado para o corpus
português.

Deve comparar configurações candidatas usando métricas objetivas de
fragmentação e cobertura.

Matriz dos seis candidatos SentencePiece executada na validação fixa C1-S1;
artefatos, amostra e tabelas de comparação passaram pelas verificações de
identidade e checksum. O cientista aprovou SentencePiece Unigram 24.576
(`unigram-v24576`) como tokenizer definitivo para o primeiro treinamento.
Resultados completos e hashes estão em
[`C2_TOKENIZER_SELECTION_V1.md`](C2_TOKENIZER_SELECTION_V1.md). O comando de
comparação permanece neutro e não declara vencedor. C2 está **COMPLETE / SCIENTIST
APPROVED / VERIFIED**.

## C3 — Arquitetura e benchmark na Orion

**Objetivo:** escolher a arquitetura final dentro da faixa de 150–200M.

As três propostas A/B/C têm implementação PyTorch e contagem analítica de
parâmetros. A validação local limitada passou para contabilidade,
configuração, política de GPU mockada, métricas e artefatos; testes que
dependem de PyTorch foram ignorados porque o pacote opcional não está
instalado neste ambiente. Resultado local verificado em 2026-10-16: `pytest`
345 passed / 9 skipped; `ruff check .` reportou apenas 2 E402 preexistentes
em `tests/model/test_transformer.py`, fora do escopo desta etapa.

A matriz C3.4 de seis benchmarks sintéticos foi executada na Orion
(`cenpesor`, GPU física 0) em 2026-10-16, sob autorização do cientista:
todos os seis pares arquitetura/contexto (A/B/C × 2.048/4.096) terminaram
completos, sem violação de política de memória, no commit
`f73d885a5dad7f6d7294c48e3027d85d8dfd26b1` (árvore limpa). Os artefatos
permanecem em `/mnt/data/cambacica-base-180m/c3-benchmarks/` com checksums
verificados. Nenhum corpus de produção ou piloto de qualidade foi executado.

O manual fixa GPU física 0 de `cenpesor`, UUID
`GPU-7562e34d-b0a2-a725-2ad5-ed1327be3cb0`, orçamento operacional máximo de
40.960 MiB e alvo inicial de pico de 36.864 MiB. GPU 1 e o serviço Qwen ficam
fora do escopo. Limitações do allocator PyTorch e monitoramento NVML estão
registradas no método e runbook.

**Decisão C3.5 — APPROVED (2026-10-16):** o cientista aprovou a
**Arquitetura C (wide/shallow)** — 9 camadas, hidden 1.024, 16 cabeças
(head dim 64), SwiGLU 4.096, vocabulário 24.576, RMSNorm, RoPE e atenção
causal SDPA; 176.180.224 parâmetros com tying (201.346.048 sem tying). A
seleção é baseada em eficiência computacional medida na matriz C3.4 (melhor
throughput e menor pico de memória nos dois contextos comparados), **não**
em qualidade de linguagem demonstrada: as perdas do benchmark são calculadas
sobre IDs sintéticos e não podem ser interpretadas como qualidade
linguística. As Arquiteturas A e B permanecem preservadas como
configurações experimentais de comparação. Registro completo em
[`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md).

Continuam indecididos: contexto final, tying, optimizer e schedule, batch
efetivo, orçamento de tokens, mistura final A/B/C e
autorização/tamanho de qualquer piloto de qualidade (C3.7). A próxima etapa
é a C3.6 — configuração de treinamento e validação de escalabilidade —,
com plano no registro de seleção.

Consulte [`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md),
[`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md),
[`C3_ARCHITECTURE_IMPLEMENTATION.md`](C3_ARCHITECTURE_IMPLEMENTATION.md),
[`C3_BENCHMARK_METHODOLOGY.md`](C3_BENCHMARK_METHODOLOGY.md) e
[`C3_BENCHMARK_RUNBOOK.md`](C3_BENCHMARK_RUNBOOK.md).

Status: **IN PROGRESS — Arquitetura C APPROVED (matriz C3.4 completa); próxima etapa C3.6: configuração de treinamento e validação de escalabilidade**

C3 só poderá ser marcado **COMPLETE** depois que o cientista revisar e
aprovar a configuração de treinamento e os resultados da etapa C3.6. A
decisão de arquitetura não congela a receita de treinamento.

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
