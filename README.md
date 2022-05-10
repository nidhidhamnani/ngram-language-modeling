# Probablistic Language Models

Implemented unigram and trigram probablistic language models and provided a detailed comparative analysis of the same.

For more details, please look at [`report.pdf`](https://github.com/nidhidhamnani/ngram-language-modeling/blob/main/report.pdf)

## Getting Started

To install tabulate:

```pip install tabulate```

To run the unigram model:

```python unigram_model.py```

To run the trigram model with linear interpolation smoothing:

```python trigram_model.py```

# Files

- unigram_lm.py: Describes the higher level interface for a language model, and contains an implementation of a simple back-off based unigram model

- generator.py: Contains a simple word and sentence sampler for any language model

- unigram_model.py: Contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models, and generate sample sentences from all the models

- trigram_lm.py: Describes the higher level interface for a language model, and contains an implementation of trigram model with linear interpolation

- trigram_model.py: Contains methods to read the appropriate data files from the archive, train and evaluate all the trigram language models, and generate sample sentences from all the models

**Note:** This project was implemented as a part of CSE 256: Statistical NLP (Spring 2022) at UCSD
