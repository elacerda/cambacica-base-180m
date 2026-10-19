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
parâmetros. A suíte de validação incluiu configuração, parâmetros, GPU
mockada, métricas, artefatos, acumulação e checkpoint/restart. Validação
anterior à execução C3.6: **387 passed / 5 skipped** com PyTorch
2.3.1+cu121; os cinco skips correspondem a testes opcionais de rede.
Esse resultado é de CPU/testes e não substitui as medições reais da GPU.

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

Em C3.6, o cientista aprovou contexto 4.096, tying, 176.180.224 parâmetros,
BF16 como precisão computacional de referência e Orion GPU física 0.
**C3.6-M: COMPLETE / PASS** — as nove células de microbatch 1/2/4 ×
acumulação 1/2/4 foram executadas em 2026-10-17 na GPU 0, todas concluídas
com resultados e checksums válidos, máximo observado **110.674 tokens/s**,
pico NVML de **19.740 MiB** e 83,48 segundos para a matriz.
**C3.6-R: COMPLETE / PASS** — checkpoint/restart em três processos
independentes, com losses idênticos bit a bit e **333/333 tensores**
idênticos, delta máximo zero. A publicação `finalize-suite` retornou
`status=published` no NFS em
`/mnt/data/cambacica-base-180m/c3-scalability/c3.6/c36-matrix-20261016-gpu0-v1/`.
A conclusão científica do C3.6 foi aprovada em 2026-10-19.
Resultados detalhados em [`C3_6_SCALABILITY_RESULTS_V1.md`](C3_6_SCALABILITY_RESULTS_V1.md);
o baseline original permanece em
[`C3_TRAINING_BASELINE_V1.md`](C3_TRAINING_BASELINE_V1.md).

Continuam indecididos: otimizador e schedule definitivos, batch efetivo,
orçamento de tokens, política de activation checkpointing, mistura final
A/B/C, armazenamento dos estados e checkpoints de treinamento.
O microbatch 4 / acumulação 1 (16.384 tokens/update) é candidato
**computacional provisório**, não uma escolha científica congelada.
O piloto de qualidade C3.7 segue **BLOCKED** por C1 (direitos/política
de uso e mistura/view final) e requer autorização científica específica.
Nenhum teste C3.6 autoriza treinamento ou processamento do corpus real.

Consulte [`C3_ARCHITECTURE_SELECTION_V1.md`](C3_ARCHITECTURE_SELECTION_V1.md),
[`C3_ARCHITECTURE_PLANNING.md`](C3_ARCHITECTURE_PLANNING.md),
[`C3_ARCHITECTURE_IMPLEMENTATION.md`](C3_ARCHITECTURE_IMPLEMENTATION.md),
[`C3_BENCHMARK_METHODOLOGY.md`](C3_BENCHMARK_METHODOLOGY.md) e
[`C3_BENCHMARK_RUNBOOK.md`](C3_BENCHMARK_RUNBOOK.md).

Status: **COMPLETE / PASS — Arquitetura C selecionada; C3.6-M e C3.6-R executados e aprovados**

Os critérios de saída de C3 foram atingidos mediante revisão científica da
matriz e do restart em 2026-10-19. A conclusão deste gate é restrita à
arquitetura e à validação de engenharia sintética; **não** congela a
receita de treinamento nem libera o C3.7, C4, C5 ou C6. C1 permanece
IN PROGRESS, e C4 continua PENDING até seu próprio planejamento e aprovação.

## C4 — Pipeline de dados e treinamento

**Objetivo:** validar packing, batching, checkpointing, retomada, logs e
reprodutibilidade.

**Planejamento C4-P autorizado em 2026-10-19.** Contrato preliminar
versionado em [`C4_PIPELINE_DESIGN_PROPOSAL_V1.md`](C4_PIPELINE_DESIGN_PROPOSAL_V1.md).
**P1 (packing contínuo delimitado por EOS) APPROVED / SCIENTIST APPROVED**
em 2026-10-19, com atenção causal entre documentos do mesmo split
explicitamente permitida. Registro científico, geometria de janelas,
rastreabilidade e contrato dos testes de referência em CPU em
[`C4_P1_PACKING_DECISION_V1.md`](C4_P1_PACKING_DECISION_V1.md).
**C4-P1.1 APPROVED / SCIENTIST APPROVED (2026-10-19):**
EOS id 2 após cada documento, EOS previsto como target válido, sem
inserção automática de BOS por documento, atenção causal contínua entre
documentos do mesmo split, janelas de 4.096 input/label com backing de
4.097 e stride 4.096, padding de inputs com token de vocabulário válido e
labels inválidos `-100` excluídos da loss. A decisão está registrada em
[`C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md`](C4_P1_1_SEQUENCE_LOSS_CONTRACT_V1.md).
Permanecem abertas políticas BOS para reinício/épocas, inserção de specials
por adaptadores, amostragem/mistura A/B/C, batch, optimizer, schedule e
orçamento de tokens. **Integração C4-P1.2 verificada em CPU com tokens artificiais (COMPLETE / PASS).**

Autorização concluída: referência P1 em Python puro e integração C4-P1.2
com a classe PyTorch do Transformer **somente em CPU e com tokens
artificiais**, sem acessar CUDA/NVML, sem ler, tokenizar ou materializar
corpus C1 de produção, e sem executar treinamento. A implementação de
streaming/amostragem, mesmo sintética, requer aprovação própria.
Processamento de corpus real, execução GPU, piloto C3.7, smoke training C5
e pretraining C6 seguem sem autorização. Direitos e mistura C1 continuam
pendentes.

**C4-P1.3 — BASELINE DE DESENHO APPROVED / SCIENTIST APPROVED
(2026-10-19):** documentação em
[`C4_P1_3_STREAMING_SAMPLING_DESIGN_V1.md`](C4_P1_3_STREAMING_SAMPLING_DESIGN_V1.md).
O cientista selecionou **O1 — ordenação determinística por hash com
índice externo em disco, sobre view e split imutáveis**, **produtor
lógico único com emissão ordenada e buffers limitados**, e **cursor
na última janela contígua confirmada/comprometida pelo consumidor**.
Os testes sintéticos de interrupção e replay permanecem critérios
futuros de verificação, não testes já realizados.

**Não foram decididos** seed concreto, identidade/serialização final
das chaves de ordenação, composição A/B/C, quotas, substituição/reuso,
épocas, orçamento de memória/disk, formato/cadência de snapshot ou
acoplamento atômico ao estado do otimizador. A decisão O1 não aprova
uma política de amostragem completa.

**C4-P1.3a autorizado separadamente e publicado em 2026-10-19:**
implementação sintética em CPU com ordenação O1 via índice SQLite,
produtor lógico único, P1 EOS/overlap, cursor de janela confirmada e
testes de retomada em
[`C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md`](C4_P1_3A_SYNTHETIC_CPU_IMPLEMENTATION_V1.md).
Commit de implementação confirmado:
`0c30fffcfa17c76767f9516283410c7fbd514683` (somente três arquivos
novos). **C4-P1.3a COMPLETE / PASS em 2026-10-19**:
testes focados na Orion **49 PASS (0,89 s)**, `ruff check` PASS;
depois do push e sincronização, `ruff format --check` reportou
**2 arquivos já formatados**, e a suíte completa CPU com CUDA oculto
reportou **450 PASS, 5 SKIP (20,09 s)**. Os motivos exatos dos cinco
skips não foram fornecidos no resultado desta execução; isso não é
evidência de validação de módulos opcionais ignorados. Nenhum novo
código de treinamento/corpus foi introduzido neste fechamento.

Autorização executada em C4-P1.3a: **somente protótipo e testes
sintéticos em CPU**. Nenhum corpus real, CUDA/GPU, C3.7, C5, C6 ou
treinamento foi autorizado.

**C4-P1.3b — COMPLETE / PASS (2026-10-19), benchmark sintético CPU:**
implementação autorizada e publicada em
`22db87a9c35cf532202c051dfe0ce43ea0530fb0`, com somente
`p1_scalability_synthetic.py`, seus testes e documentação adicionados.
Na Orion, os dois novos arquivos passaram no `ruff check` e
`ruff format --check`; a suíte completa com CUDA oculto apresentou
**470 PASS / 5 SKIP (20,62 s)**. Três execuções independentes
com tokens artificiais e persistência em `/tmp` no ext4 local
`/dev/nvme0n1p2` mediram 128, 1.024 e 8.192 documentos de
128 tokens: pico RSS de 21.136, 22.012 e 23.960 KiB, índice total
de 135.736, 897.594 e 6.943.291 bytes, duração total de
0,086197, 0,506973 e 6,924444 s, respectivamente. O maior
caso teve 1.056.768 tokens e 258 janelas. O tempo de ack + fsync
foi 2,415402 s de 5,546933 s de leitura + ack.
Os resultados são **medidas de uma execução por tamanho**,
sem generalização para o C1 real ou prova de recuperação de uma falha
de energia. Registro em
[`C4_P1_3B_SYNTHETIC_CPU_SCALABILITY_V1.md`](C4_P1_3B_SYNTHETIC_CPU_SCALABILITY_V1.md).

**O escopo não muda:** sem acesso a corpus C1, tokenização real,
GPU/CUDA/NVML, treinamento, seleção da mistura, ou autorização
de escalabilidade de produção. Novas otimizações do cursor ou
batching dependem de avaliação/autorização separada.

**C4-P2 — SCIENTIFIC DESIGN PROPOSED / REVIEW PENDING
(2026-10-19):** o cientista autorizou **somente planejamento
e documentação** do pipeline futuro. Proposta versionada em
[`C4_P2_PRODUCTION_PIPELINE_SCIENTIFIC_DESIGN_V1.md`](C4_P2_PRODUCTION_PIPELINE_SCIENTIFIC_DESIGN_V1.md).
O documento distingue cotas em palavras normalizadas vs tokens C2,
admissão de documentos inteiros, ordenação O1 vs interleaving de fontes,
limites de épocas, reuso, accounting de EOS/targets, interface de
tokenização, batch, persistência e checkpoint atomicamente associado
ao estado de treinamento. **Nenhuma nova política foi aprovada.**
Mistura A/B/C e permissões de uso continuam bloqueadas pelo C1;
semente, scheduler, épocas, token budget, batch, otimização do cursor,
recursos e critérios de execução permanecem OPEN.

**Autorização C4-P2: documentação exclusivamente.** Não houve
implementação, processamento/tokenização do corpus C1,
execução de testes, uso de GPU/CUDA/NVML, treinamento, C3.7,
C5 ou C6 nesta etapa.

**C4-P2.1 — REVIEW DOCUMENTED / SCIENTIST DECISIONS PENDING
(2026-10-19):** autorização explícita apenas para revisar cientificamente,
em papel, as dependências até o primeiro treinamento. Documento
[`C4_P2_1_TRAINING_READINESS_DECISION_REVIEW_V1.md`](C4_P2_1_TRAINING_READINESS_DECISION_REVIEW_V1.md)
separa **C5 (primeiro smoke real e limitado)** de **C6 (pré-treino
principal)**, com os bloqueios e a matriz de aprovação. A proposta
recomenda priorizar um pequeno C5 em *view* explicitamente liberada
para treino e holdout, enquanto a composição e orçamento do C6 são
decididos em paralelo; **não declara nenhuma fonte liberada** e
**não seleciona** M1/M2, cotas Q-T/Q-W, merge S1/S2, épocas,
reuso, treinamento ou orçamento de tokens. C1 documentou
tetos no-repeat aproximados em palavras normalizadas pós-holdout
A ≤259.918.425; B ≤519.836.850; C 415.336.413 **apenas
ilustrativo** (98% assumidos). Estes números não são tokens C2
nem garantias de capacidade para um experimento real.

**Decisão científica posterior (2026-10-19): prioridade M1 APPROVED.**
Após a revisão C4-P2.1, o cientista concordou em priorizar
**um C5 real, curto e limitado, sobre view explicitamente autorizada**
antes do C6 principal. Esta é **aprovação de prioridade de trabalho**,
não uma autorização de uso de fontes, definição de dados, implementação
C4-D/C4-T, receita, GPU ou execução C5. O próximo bloqueio decisório
é o **C1: base de uso/direitos da(s) fonte(s) escolhida(s), obrigações e
view de treino/holdout preservando C1-S1**. Mistura A/B/C final,
orçamento de tokens, amostragem, épocas e checkpoint permanecem abertos.

**Autorização C4-P2.1 permanece somente documental.** Pipeline real,
tokenização C1, acesso a corpus, qualquer GPU/treinamento e C5/C6
continuam sem autorização; o C4 não foi concluído.

Status: **IN PROGRESS — P1.2/P1.3a/P1.3b SYNTHETIC CPU PASS; P1.3 DESIGN APPROVED; P2 DESIGN PROPOSED; P2.1 M1 PRIORITY APPROVED / EXECUTION AND POLICIES PENDING**

A implementação da integração de loss **C4-P1.2** foi autorizada e
adicionada (referência PyTorch apenas CPU, modelo C3 com dimensões reduzidas,
validação explícita de `ignore_index=-100` e cálculo de CE mascarado).
**C4-P1.2 COMPLETE / PASS (2026-10-19):** 51 testes focados aprovados
após formatação (2,51 s); suíte completa antes da formatação: 425
aprovados e cinco skips (19,35 s), com CUDA oculto. `ruff check` e
`ruff format --check` passaram nos dois arquivos corrigidos, e o
`git diff --check` passou. Commit de formatação publicado:
`29e966a283b2fc7f1343fefacceeb3c47ad9287f`, que altera somente
os dois arquivos Python da integração. Esta validação é sintética/CPU,
sem execução real de pretraining.
Relatório e evidências em
[`C4_P1_2_CPU_LOSS_INTEGRATION_V1.md`](C4_P1_2_CPU_LOSS_INTEGRATION_V1.md).
Nenhuma execução GPU, treino, processamento do corpus ou alteração da
arquitetura C ocorreu nesta tarefa.

O protótipo de referência, limitado a tokens e metadados artificiais,
está em [`src/cambacica/packing/p1_reference.py`](../src/cambacica/packing/p1_reference.py),
com testes em [`tests/packing/test_p1_reference.py`](../tests/packing/test_p1_reference.py).
Validação: 24/24 testes focados; suíte CPU disponível com 379 aprovados e
10 skips registrados em [`C4_P1_PACKING_DECISION_V1.md`](C4_P1_PACKING_DECISION_V1.md).
Ruff e formatação passaram nos arquivos Python alterados; verificações de
repositório completo ainda apontam problemas preexistentes em arquivos
inalterados. A implementação não conclui C4 nem autoriza pipeline real,
CUDA, treinamento, C3.7, C5 ou C6. BOS por documento, scoring de EOS,
fronteiras e mascaramento de labels foram aprovados em C4-P1.1;
a integração PyTorch em CPU foi verificada em C4-P1.2.
O streaming **sintético** O1 e cursor de confirmação foram validados
em C4-P1.3a. **Permanecem abertos** a amostragem/streaming de produção,
retomada conjunta com estado do otimizador, mistura, batch final,
receita de treinamento e as decisões bloqueadas por C1.

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
