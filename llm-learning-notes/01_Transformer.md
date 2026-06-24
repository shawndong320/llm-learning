## CS324 Introduction Key Takeaway

**What a language model is**
 A language model is fundamentally a probability distribution over token sequences. An autoregressive LM models the sequence by repeatedly predicting the next token.

**Core formula**
$$
p(x_1,\dots,x_L)=\prod_{i=1}^{L}p(x_i \mid x_{1:i-1})
$$
This means the probability of a full sequence is decomposed into the product of next-token probabilities conditioned on previous tokens.

**Training objective**
 LLMs are not directly trained to “know answers.” They use self-supervised learning: raw text is automatically converted into
 `context input → next-token label`
 pairs, and cross entropy penalizes the model when it assigns low probability to the true next token.

**Generation pipeline**
 The model outputs logits, softmax converts logits into a probability distribution, temperature adjusts the sharpness or randomness of that distribution, and decoding chooses the actual next token.

**Role of temperature**
 Lower temperature makes the distribution sharper and outputs more deterministic. Higher temperature makes the distribution flatter and outputs more diverse, but also more error-prone.

**N-gram vs neural LM**
 N-gram models are also language models, but they only condition on the previous $n-1$ tokens. They suffer from data sparsity and weak generalization. Neural LMs use embeddings and neural networks to learn more generalizable representations, but require much more computation.

**In-context learning**
 In-context learning means the model infers a task pattern from the prompt without updating its parameters. It is not fine-tuning and does not involve backpropagation.

**Why scale matters**
 Larger models are not merely smaller models with more parameters. Scaling enables stronger prompt-based task solving, few-shot learning, code generation, and question answering, while also increasing risks such as hallucination, bias, cost, and access inequality.

**Main idea of CS324 Introduction**
 The lecture is not mainly about Transformer details. Its goal is to build the big picture: probability modeling → next-token prediction → self-supervised pretraining → scale → in-context learning → capabilities and risks.



## 