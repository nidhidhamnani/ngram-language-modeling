#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Trigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.unigram = dict()
        self.bigram = dict()
        self.trigram = dict()
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    # Computes unigram frequencies
    def unigram_word(self, w):
        if w in self.unigram:
            self.unigram[w] += 1.0
        else:
            self.unigram[w] = 1.0

    # Computes bigram frequencies
    def bigram_word(self, w1, w2):
        if (w1, w2) in self.bigram:
            self.bigram[(w1, w2)] += 1.0
        else:
            self.bigram[(w1, w2)] = 1.0

    # Computes trigram frequencies
    def trigram_word(self, w1, w2, w3):
        if (w1, w2, w3) in self.trigram:
            self.trigram[(w1, w2, w3)] += 1.0
        else:
            self.trigram[(w1, w2, w3)] = 1.0

    # Learns data based on each sentence
    def fit_sentence(self, sentence):
        for i in range(len(sentence)):
            self.unigram_word(sentence[i]) #updating unigram counts

            # Updating bigram counts
            if i==0:
                self.bigram_word('START', sentence[i]) # inserting START token in the beginning of the sentence
            else:
                self.bigram_word(sentence[i-1], sentence[i])

            # Updating trigram counts
            if i==0:
                self.trigram_word('START', 'START', sentence[i]) # inserting START, START tokens in the beginning of the sentence
            elif i==1:
                self.trigram_word('START', sentence[i-1], sentence[i])
            else:
                self.trigram_word(sentence[i-2], sentence[i-1], sentence[i])
        
        # Inserting START and END_OF_SENTENCE tokens for all the three models
        self.unigram_word('START')
        self.unigram_word('END_OF_SENTENCE')
        self.bigram_word('START', 'START')
        self.bigram_word('START', 'END_OF_SENTENCE')
        self.trigram_word('START', 'START', 'END_OF_SENTENCE')
        if(len(sentence)>1):
            self.trigram_word(sentence[len(sentence)-2], sentence[len(sentence)-1], 'END_OF_SENTENCE')
        if(len(sentence)>0):
            self.bigram_word(sentence[len(sentence)-1], 'END_OF_SENTENCE')
        if(len(sentence)==1):
            self.trigram_word('START', sentence[len(sentence)-1], 'END_OF_SENTENCE')


    def norm(self):
        """Normalize trigram counts and convert to log2-probs."""
        uni_tot = 0
        for word in self.unigram:
            uni_tot += self.unigram[word]
        self.uni_tot = uni_tot

        # Setting values for lambdas
        self.lambda1 = 0.34
        self.lambda2 = 0.33
        self.lambda3 = 0.33
        for (w1, w2, w3) in self.trigram:
            temp = ((self.lambda1*self.trigram[(w1, w2, w3)])/self.bigram[(w1, w2)]) + \
                    ((self.lambda2*self.bigram[(w2, w3)])/self.unigram[w2]) + \
                    ((self.lambda3*self.unigram[w3])/uni_tot)

            self.model[(w1, w2, w3)] = log(temp, 2)


    def cond_logprob(self, word, previous):
        # Calculating probabilities for 'word' using the 'previous' sentence

        w1 = ''
        w2 = ''

        if len(previous) == 0:
            w1 = 'START'
            w2 = 'START'
        elif len(previous) == 1:
            w1 ='START'
            w2 = previous[0]
        else:
            w1 = previous[len(previous)-2]
            w2 = previous[len(previous)-1]

        # If the word exists in trigram use the normalized count
        if (w1, w2, word) in self.model:
            return self.model[(w1, w2, word)] 
        else:
            # If (w1, w2, word) do not exist in trigram and (w2, word) do no exist in bigram
            # then use the weighted unigram frequency based on linear interpolation smoothing
            if (word in self.unigram) and ((w2, word) not in self.bigram): 
                temp = ((self.lambda3*self.unigram[word])/self.uni_tot)
                return log(temp, 2)

            # If (w1, w2, word) do not exist in trigram and (w2, word) exist in bigram
            # then use the weighted unigram and bigram frequencies based on linear interpolation smoothing
            elif (word in self.unigram) and (w2 in self.unigram) and ((w2, word) in self.bigram):
                temp = ((self.lambda3*self.unigram[word])/self.uni_tot) + ((self.lambda2*self.bigram[(w2, word)])/self.unigram[w2])
                return log(temp, 2)
            
            # If (w1, w2, word) do not exist in trigram, (w2, word) do not exist in bigram, and word 
            # do not exist in unigram then use the back-off probability
            else:
                return self.lbackoff

    def vocab(self):
        return self.unigram.keys()
    
    def trigram_vocab(self):
        return self.model.keys()

