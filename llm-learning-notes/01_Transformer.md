# CS324 Introduction — Study Notes

## One-sentence summary

This lecture introduces large language models as probabilistic models over token sequences, showing how next-token prediction, self-supervised learning, scale, and in-context learning turn language modeling from a narrow statistical task into a general-purpose interface with significant capabilities and risks.

---

## Core concepts map

A language model assigns probabilities to token sequences, because modeling language can be framed as estimating how likely a sequence of tokens is.

A token sequence enables probabilistic modeling by giving the model a discrete sequence $x_1, x_2, \dots, x_L$ over which it can define a joint probability.

The joint probability of a sequence leads to autoregressive factorization, because the chain rule of probability lets us decompose $p(x_1,\dots,x_L)$ into a product of conditional next-token probabilities.

Autoregressive factorization enables next-token prediction by turning the difficult problem of modeling an entire sequence into repeated predictions of $p(x_i \mid x_1,\dots,x_{i-1})$.

Next-token prediction enables self-supervised learning, because raw text can automatically provide input-label pairs: the prefix is the input and the next token is the label.

Self-supervised learning enables large-scale pretraining, because models can learn from massive unlabeled text corpora without requiring expensive human annotations.

Cross-entropy loss trains the model by penalizing it when it assigns low probability to the true next token.

Logits enable probability prediction by giving the model raw scores over the vocabulary before normalization.

Softmax converts logits into a probability distribution, because it maps arbitrary scores into non-negative values that sum to one.

Temperature modifies the output distribution by making it sharper or flatter, which controls how deterministic or random generation becomes.

Decoding turns a probability distribution into an actual generated token by using strategies such as greedy decoding or sampling.

Prompting enables task specification by placing instructions, examples, or context before generation.

Completion follows from prompting because an autoregressive model continues the given context one token at a time.

In-context learning enables temporary task adaptation by allowing the model to infer a task pattern from examples in the prompt without updating its parameters.

N-gram models are early language models, but they suffer from short context length and data sparsity because they only condition on the previous $n-1$ tokens.

Neural language models improve generalization by using embeddings and neural networks to learn reusable semantic and syntactic representations.

Scale enables stronger capabilities by increasing model size, training data, and compute, allowing models to learn broader patterns, knowledge, and task formats.

Scale also increases risk, because more capable models can produce more convincing hallucinations, amplify bias, generate disinformation, and require resources accessible only to a few institutions.

---

## Key formulas

### Sequence probability

$$
p(x_1,\dots,x_L)
$$

This denotes the probability assigned to the entire token sequence. It is the basic mathematical object behind language modeling.

---

### Autoregressive factorization

$$
p(x_1,\dots,x_L)
=
\prod_{i=1}^{L} p(x_i \mid x_1,\dots,x_{i-1})
$$

This decomposes the probability of a full sequence into a product of next-token conditional probabilities. Here, $x_i$ is the $i$-th token, $x_1,\dots,x_{i-1}$ is the previous context, and $\prod$ is the product operator.

A shorter notation is:

$$
p(x_1,\dots,x_L)
=
\prod_{i=1}^{L} p(x_i \mid x_{<i})
$$

where $x_{<i}$ means all tokens before position $i$.

---

### N-gram approximation

$$
p(x_i \mid x_1,\dots,x_{i-1})
\approx
p(x_i \mid x_{i-n+1},\dots,x_{i-1})
$$

An n-gram model approximates next-token prediction by only looking at the previous $n-1$ tokens. This makes it computationally simple but weak at long-range dependency modeling and generalization.

---

### Softmax

$$
\mathrm{softmax}(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax converts logits into a probability distribution over the vocabulary. Higher logits receive higher probabilities, and all probabilities sum to one.

---

### Temperature-adjusted softmax

$$
\mathrm{softmax}\left(\frac{z_i}{T}\right)
=
\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

Temperature controls the sharpness of the probability distribution during generation. Lower $T$ makes the distribution sharper and more deterministic; higher $T$ makes it flatter and more random.

---

### Cross-entropy loss

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

For one-hot labels in next-token prediction, this becomes:

$$
\mathrm{loss} = -\log q(x_{\text{true}})
$$

Cross entropy penalizes the model when it assigns low probability to the true next token.

---

## What I understood well

I understood that a language model is fundamentally a probability model over token sequences, and that autoregressive language models use the chain rule to reduce sequence modeling into repeated next-token prediction.

I understood the generation pipeline: the model outputs logits, softmax converts them into a probability distribution, temperature adjusts the distribution, and decoding selects the actual next token.

I understood that cross-entropy loss trains the model by penalizing low probability assigned to the true next token, rather than simply comparing predicted and actual values directly.

I understood self-supervised learning: raw text provides its own labels because each prefix can be paired with the next token.

I understood the main weakness of n-gram models: they are computationally efficient but statistically inefficient, with short context length, data sparsity, and weak generalization.

I understood that in-context learning does not involve backpropagation or parameter updates; it works by conditioning the model on examples or instructions in the prompt.

---

## What needs review

I should review the distinction between language models in general and autoregressive language models specifically. A language model assigns probabilities to sequences, while an autoregressive language model does so through next-token conditional probabilities.

I should review the distinction between zero-shot, one-shot, and few-shot prompting. Zero-shot gives no examples, one-shot gives one example, and few-shot gives multiple examples.

I should be careful not to describe in-context learning as fine-tuning. Fine-tuning updates model parameters, while in-context learning only changes the context and therefore the output distribution.

I should review the information-theoretic interpretation of entropy and cross entropy, especially the connection between language modeling, uncertainty, perplexity, and compression.

I should review why low perplexity usually indicates better language modeling, but does not necessarily imply better factuality, reasoning, safety, or alignment.

I should review the role of scale: larger models are not merely smaller models with more parameters; scale can enable qualitatively stronger capabilities while also increasing risks.




# CS324 Capabilities — Study Notes

## One-sentence summary of the entire lecture

This lecture explains how a general-purpose language model trained for **next-token prediction** can be adapted to many **downstream tasks** through **prompting**, **in-context learning**, **conditional generation**, and **sequence scoring**, while also showing that strong language modeling does not necessarily guarantee factuality, reasoning, or reliability.

------

## Core concepts map

A **language model** → assigns probabilities to token sequences, because language modeling can be framed as estimating how likely a sequence of tokens is.

A **token sequence** → enables probabilistic modeling, because the model can decompose a long text into ordered units $(x_1, x_2, \dots, x_L)$.

**Next-token prediction** → enables full sequence modeling, because the probability of a whole sequence can be factorized into conditional probabilities of each token given previous tokens.

**Conditional next-token probabilities** → enable **sequence probability**, because multiplying $(p(x_i \mid x_{1:i-1}))$ across all positions gives the probability of the entire sequence.

**Sequence probability** → enables **perplexity**, because perplexity is computed from the average negative log-likelihood assigned to the true next tokens.

**Perplexity** → measures language modeling quality, because it reflects how uncertain the model is when predicting the true next token.

**Perplexity** does not → guarantee reasoning ability, because predicting likely text is different from performing verified reasoning or exact computation.

**Adaptation** → turns a general-purpose language model into a task-specific model, because tasks can be expressed through instructions, examples, or additional training.

**Prompting-based adaptation** → enables task behavior without parameter updates, because the task is specified through natural language instructions and in-context examples.

**Fine-tuning** → enables task adaptation by updating model parameters, because the model is further trained on task-specific data.

**In-context learning** → enables few-shot task learning, because examples in the prompt demonstrate the input-output pattern the model should follow.

**Question answering** → can be formulated as conditional generation, because the model generates an answer given a question.

**Translation** → can be formulated as conditional generation, because the model generates a target-language sentence given a source-language sentence.

**Multiple-choice tasks** → can be reformulated as sequence scoring problems, because the model can score each candidate completion and choose the one with the highest conditional probability.

**Closed-book QA** → relies on parametric knowledge, because the model answers factual questions without retrieving external documents.

**Open-book QA / RAG** → improves factual grounding, because it uses retrieved external evidence instead of relying only on model parameters.

**Open-ended generation** → requires human evaluation, because there is often no single correct answer and automatic metrics may fail to capture factuality, coherence, or human-likeness.

**Prompt engineering** → elicits desired model behavior, because prompt wording, formatting, and examples can strongly affect model outputs.

**Prompt sensitivity** → limits prompting reliability, because small changes in wording or format can lead to different completions.

------

## Key formulas

### 1. Sequence probability

$$
p(x_{1:L}) = \prod_{i=1}^{L} p(x_i \mid x_{1:i-1})
$$

This formula says that the probability of a full token sequence can be factorized into a product of conditional next-token probabilities.

Where:

- $x_{1:L}$: the full token sequence
- $x_i$: the $i$-th token
- $x_{1:i-1}$: all previous tokens before $x_i$
- $p(x_i \mid x_{1:i-1})$: the probability of the next token given the previous context
- $\prod$: product over all token positions

Key English expression:

> The probability of the whole sequence can be factorized into a product of conditional next-token probabilities.

------

### 2. Perplexity

$$
\text{PPL}
=
\exp\left(
-\frac{1}{L}
\sum_{i=1}^{L}
\log p(x_i \mid x_{1:i-1})
\right)
$$

Perplexity measures how uncertain the model is when predicting the true next token. It is the exponentiated average negative log-likelihood per token.

Important interpretation:
$$
\text{lower perplexity} = \text{better language modeling}
$$
But:
$$
\text{lower perplexity} \neq \text{stronger reasoning ability}
$$
Key English expression:

> Perplexity is the exponentiated average negative log-likelihood per token.

------

### 3. Sequence scoring

$$
\hat{a} = \arg\max_{a_k} p(a_k \mid c)
$$

This formula says that, given a context $c$, the model selects the candidate answer $a_k$ with the highest conditional probability.

Where:

- $c$: context
- $a_k$: the $k$-th candidate completion
- $\arg\max$: returns the candidate with the highest score

Key English expression:

> Multiple-choice tasks can be reformulated as sequence scoring problems.

------

### 4. Conditional generation

$$
\hat{y} = \arg\max_y p(y \mid x)
$$

This formula says that the model generates or selects the most likely output $y$ given input $x$.

Examples:
$$
p(\text{answer} \mid \text{question})
$$
Key English expression:

> Conditional generation produces an output conditioned on an input.

------

## What I understood well

You understood **adaptation** well: a general-purpose language model can be turned into a task-specific model through either prompting-based adaptation or training-based adaptation.

You understood the core difference between **prompting** and **fine-tuning**: prompting does not update model parameters, while fine-tuning updates the model on task-specific data.

You understood **sequence probability**: a language model gives a whole sequence a probability by multiplying conditional next-token probabilities.

You understood the main intuition behind **perplexity**: lower perplexity means the model is less uncertain when predicting the true next token, but perplexity does not directly measure reasoning ability.

You understood **sequence scoring** clearly: the model can score predefined candidate completions and choose the one with the highest conditional probability.

You understood **conditional generation** for both question answering and translation: QA generates an answer given a question, while translation generates a target-language sentence given a source-language sentence.

You understood the distinction between **surface fluency** and **factual reliability**: text can sound human-like while still being factually wrong.

You understood the basic idea of **prompt engineering**: prompts are designed to elicit desired behavior from a language model, but the process is often heuristic and sensitive to wording.

------

## What needs review

You should review the distinction between **source language** and **source sentence**, and between **target language** and **target sentence**. In translation formulas, $(x)$ and $(y)$ usually refer to sequences or sentences, not the languages themselves.

You should review **closed-book QA vs open-book QA**. Closed-book QA does not mean “offline”; it means the model answers without retrieving external documents. Open-book QA does not necessarily mean “searching the internet”; it means using retrieved external evidence.

You should review the relationship between **loss** and **perplexity**. Perplexity is derived from the average negative log-likelihood, but it is not exactly the raw loss itself.

You should review **length bias** in sequence scoring. Longer sequences tend to get lower total probability because they involve multiplying more probabilities less than one, which is why length normalization is often needed.

You should review the distinction between **pattern matching** and **algorithmic reasoning**. Language models can learn arithmetic-like patterns from text, but this does not necessarily mean they can perform reliable symbolic computation.

You should continue improving technical English collocations, especially:

> formulate A as B
> reformulate X as Y
> factorized into
> conditioned on
> task-specific model
> in-context learning
> externally retrieved evidence
> digit-level manipulation
> hallucination-free output