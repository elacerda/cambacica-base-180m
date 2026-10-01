# Escopo científico

## Modelo

Nome do projeto: **Cambacica**

Identificador inicial do modelo:

`cambacica-base-180m`

O sufixo `180m` representa o alvo nominal de escala. O número final de
parâmetros será determinado após o benchmark de arquitetura e deverá permanecer
aproximadamente na faixa de **150–200M**.

## Língua

O Cambacica será um modelo **nativamente em português**.

Isso significa que o pretraining será projetado para português desde o início:

- seleção do corpus;
- treinamento do tokenizer;
- análise de fragmentação;
- avaliação;
- documentação científica.

O objetivo não é exigir que todo documento contenha exclusivamente palavras
portuguesas. Nomes próprios, termos técnicos, citações e fragmentos de outras
línguas podem ocorrer naturalmente em textos escritos em português.

Conteúdo predominantemente em outras línguas não deve ser incluído
deliberadamente como componente do corpus de pretraining.

## Tipo de modelo

O primeiro Cambacica será:

- autoregressivo;
- causal;
- decoder-only;
- treinado do zero;
- modelo base, não instruct/chat.

A arquitetura exata ainda não está definida.

## Infraestrutura-alvo

O treinamento deve ser viável usando aproximadamente **40 GB de uma NVIDIA
H100** na Orion.

O projeto deve privilegiar uma receita single-GPU simples e reproduzível.

A escolha definitiva entre aproximadamente 150M, 180M e 200M parâmetros deverá
ser feita por benchmark, considerando pelo menos:

- tokens por segundo;
- uso de VRAM;
- tamanho de batch efetivo;
- tempo por step;
- estabilidade;
- utilização efetiva da GPU;
- throughput do pipeline de dados.

## Não objetivos

Nesta etapa, não fazem parte do escopo:

- escalar para 1B ou mais parâmetros;
- treinamento distribuído como requisito;
- instruction tuning;
- RLHF;
- DPO;
- agentes;
- multimodalidade;
- otimização para inglês;
- compatibilidade artificial com tokenizers de modelos estrangeiros.

## Política de decisão

Decisões científicas relevantes devem ser:

1. explicitadas;
2. medidas quando possível;
3. versionadas;
4. congeladas antes do treinamento principal;
5. alteradas depois apenas com justificativa documentada.
