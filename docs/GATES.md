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
- Inventário de fontes concluído;
- Pipeline de amostragem determinística e inspeção diagnóstica implementado (`docs/CORPUS_SAMPLING.md`).

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
