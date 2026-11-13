# Data card: RAGTruth

## 1. Source
- Dataset: `wandb/RAGTruth-processed` on Hugging Face (a processed copy of RAGTruth).
- What it is: answers written by several LLMs, such as GPT-4, Llama-2 and Mistral from a given context,
  with hallucinated spans marked by people.
- Why this copy: this is the standard copy for training and benchmarking, and its labels are already summarised per answer.

## 2. Size & splits
| split | rows |
|---|---|
| train | 15090 |
| test  | 2700 |

- No validation split, so we'll have to create my own from train split.

## 3. Label
- `label = 1` (hallucinated) if value of a key is 1 in `hallucination_labels_processed`, else `0`.
- Why that column and not `hallucination_labels`: because it's a str 
- Class balance (train): 8369 supported / 6721 hallucinated (55% / 45%).
- Majority-class baseline accuracy: 55%

## 4. Findings
| task | rows | hallucinated |
|---|---|---|
| Data2txt | 5298 | 69% |
| QA       | 5034 | 31% |
| Summary  | 4758 | 31% |

- What this means: A model could guess “made up” for every Data2txt answer and look good without checking anything, so we should report scores per task type. 
- Context length: median 317 words, max 1750. What this means: each model differ some can only read 500 tokens and many context are longer so we might need to cut long documents into pieces
- Answer length: supported answers average 119 words, hallucinated 146. What this means: if the answers are quite long it considers them as made up

## 5. Risks & open questions
- A model could guess from the task type alone instead of reading the answer.
- Hallucinated answers tend to be longer, so a model could guess “long means hallucinated” instead of checking the facts.
- The shortest answer was 1 so we need to check it because it's odd.
- We still need to check `quality` and `model` column