#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sys

import matplotlib.pyplot as plt


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)




def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    for member in tar.getmembers():
        if dname in member.name and ('train.txt') in member.name:
            print('\ttrain: %s'%(member.name))
            train_txt = unicode(tar.extractfile(member).read(), errors='replace')
        elif dname in member.name and ('test.txt') in member.name:
            print('\ttest: %s'%(member.name))
            test_txt = unicode(tar.extractfile(member).read(), errors='replace')
        elif dname in member.name and ('dev.txt') in member.name:
            print('\tdev: %s'%(member.name))
            dev_txt = unicode(tar.extractfile(member).read(), errors='replace')

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data

def learn_unigram(data, percentage, verbose=True):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from unigram_lm import Unigram
    unigram = Unigram()
    data_size = len(data.train)
    data.train = data.train[:int(percentage*data_size)]
    unigram.fit_corpus(data.train)
    if verbose:
        print("vocab:", len(unigram.vocab()))
        # evaluate on train, test, and dev
        print("train:", unigram.perplexity(data.train))
        print("dev  :", unigram.perplexity(data.dev))
        print("test :", unigram.perplexity(data.test))
        # from generator import Sampler
        # sampler = Sampler(unigram)
        # print("sample 1: ", " ".join(str(x) for x in sampler.sample_sentence([])))
        # print("sample 2: ", " ".join(str(x) for x in sampler.sample_sentence([])))
    return unigram

if __name__ == "__main__":

    graphs = {
        'train': [],
        'dev': [],
        'test': []
    }

    current_corpus = "reuters"

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for k in percentages:
        data = read_texts("data/corpora.tar.gz", current_corpus)
        model = learn_unigram(data, k)
        v = len(model.vocab())
        print(v)
        graphs['train'].append(model.perplexity(data.train))
        graphs['test'].append(model.perplexity(data.test))
        graphs['dev'].append(model.perplexity(data.dev))


    plt.xlabel('Training data size (in percentage)')
    plt.ylabel('Perplexity')
    title = current_corpus+': training data size vs perplexity'
    plt.title(title)
    plt.plot(percentages, graphs['train'], 'r', label='Training Perplexity')
    plt.plot(percentages, graphs['dev'], 'b', label='Dev Perplexity')
    plt.plot(percentages, graphs['test'], 'g', label='Test Perplexity')
    plt.legend()
    plt.show()
