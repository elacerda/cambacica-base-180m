# Cambacica Base 180M

**Cambacica** é um pequeno modelo de linguagem causal, treinado do zero e
nativamente em português.

Este repositório acompanha o desenvolvimento científico e o treinamento do
primeiro modelo da família:

> `cambacica-base-180m`

## Objetivo

Construir um modelo base de aproximadamente **150–200 milhões de parâmetros**,
com alvo nominal de **~180M**, treinável rapidamente usando aproximadamente
**40 GB de uma NVIDIA H100** na Orion.

O objetivo desta etapa não é escalar para modelos maiores. O Cambacica será
tratado como um projeto completo em sua própria escala, com ênfase em:

- português como língua nativa de pretraining;
- treinamento do zero;
- tokenizer treinado para português;
- corpus selecionado e documentado;
- experimentos controlados;
- reprodutibilidade;
- avaliação em português;
- execução em uma única GPU.

## Estado

O tokenizer para o primeiro treinamento está congelado: SentencePiece Unigram
24.576, candidato `unigram-v24576`. A decisão foi aprovada pelo cientista e os
artefatos de produção e comparação foram verificados. Consulte o [registro de
seleção C2](docs/C2_TOKENIZER_SELECTION_V1.md).

C1 permanece **IN PROGRESS** enquanto direitos e políticas de uso estiverem
pendentes. O split determinístico C1-S1 está COMPLETE / VERIFIED; C1-S2 está
COMPLETE / VERIFIED. A mistura final A/B/C continua undecidida. A mistura B foi
somente a referência provisória para amostragem do tokenizer. Consulte os
[gates do projeto](docs/GATES.md) e o [estado e identidades do split
C1-S1](docs/C1_SPLIT_AND_MIXTURE_READINESS.md).

C3 está em planejamento. A arquitetura, o context length, o optimizer, o
orçamento final de tokens e a receita de treinamento continuam sem decisão.
As propostas para revisão científica estão em
[`docs/C3_ARCHITECTURE_PLANNING.md`](docs/C3_ARCHITECTURE_PLANNING.md).

## Princípios

1. **Português nativo**  
   O corpus de pretraining deve ser composto por textos genuinamente escritos
   em português. Conteúdo em outras línguas pode ocorrer naturalmente dentro
   desses textos, mas não será incluído deliberadamente como componente de
   treinamento.

2. **Treinamento do zero**  
   O modelo não será obtido por adaptação ou continuação de pretraining de um
   modelo estrangeiro.

3. **Modelo pequeno por projeto**  
   O tamanho será escolhido dentro da faixa de aproximadamente 150–200M
   parâmetros a partir de medições reais de eficiência na Orion.

4. **Single-GPU**  
   A receita principal deve funcionar em aproximadamente 40 GB de uma H100,
   sem exigir paralelismo distribuído.

5. **Base primeiro**  
   Esta fase cobre um modelo causal base. Instruction tuning, RLHF, DPO ou
   variantes conversacionais não fazem parte do escopo inicial.

6. **Sem roadmap de escala**  
   Este repositório não pressupõe uma futura versão de 1B ou maior.

## Gates

O trabalho será organizado em gates científicos e de engenharia:

- **C0 — escopo científico e infraestrutura**
- **C1 — corpus**
- **C2 — tokenizer**
- **C3 — arquitetura e benchmark na Orion**
- **C4 — pipeline de dados e treinamento**
- **C5 — smoke training**
- **C6 — treinamento principal**
- **C7 — avaliação**

Veja [`docs/GATES.md`](docs/GATES.md).

## Estrutura inicial

```text
cambacica-base-180m/
├── configs/      # configurações versionadas de experimentos
├── docs/         # decisões, gates e relatórios
├── scripts/      # utilitários operacionais e científicos
├── src/          # código do projeto
├── tests/        # testes
└── README.md
```

Artefatos grandes, datasets, checkpoints e logs de treinamento não devem ser
versionados diretamente no Git.
