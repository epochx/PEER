# Courtney Napoles
# <napoles@cs.jhu.edu>
# 21 June 2015
# ##
# gleu.py
# 
# This script calculates the GLEU score of a sentence, as described in
# our ACL 2015 paper, Ground Truth for Grammatical Error Correction Metrics
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
# 
# For instructions on how to get the GLEU score, call "compute_gleu -h"
#
# Updated 2 May 2016: This is an updated version of GLEU that has been
# modified to handle multiple references more fairly.
#
# Updated 6 9 2017: Fixed inverse brevity penalty
# 
# This script was adapted from bleu.py by Adam Lopez.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

import math
from collections import Counter

import sys
import os
import scipy.stats
import numpy as np
import random
import tempfile


class GLEU :

    def __init__(self,n=4) :
        self.order = 4

    def load_hypothesis_sentence(self,hypothesis) :
        self.hlen = len(hypothesis)
        self.this_h_ngrams = [ self.get_ngram_counts(hypothesis,n)
                               for n in range(1,self.order+1) ]

    def load_sources(self,spath) :
        self.all_s_ngrams = [ [ self.get_ngram_counts(line.split(),n)
                                for n in range(1,self.order+1) ]
                              for line in open(spath) ]

    def load_references(self,rpaths) :
        self.refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for rpath in rpaths :
            for i,line in enumerate(open(rpath)) :
                self.refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        # count number of references each n-gram appear sin
        self.all_rngrams_freq = [ Counter() for i in range(self.order) ]

        self.all_r_ngrams = [ ]
        for refset in self.refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)

                for k in ngrams.keys() :
                    self.all_rngrams_freq[n-1][k]+=1

                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]

    def get_ngram_counts(self,sentence,n) :
        return Counter([tuple(sentence[i:i+n])
                        for i in range(len(sentence)+1-n)])

    # returns ngrams in a but not in b
    def get_ngram_diff(self,a,b) :
        diff = Counter(a)
        for k in (set(a) & set(b)) :
            del diff[k]
        return diff

    def normalization(self,ngram,n) :
        return 1.0*self.all_rngrams_freq[n-1][ngram]/len(self.rlens[0])

    # Collect BLEU-relevant statistics for a single hypothesis/reference pair.
    # Return value is a generator yielding:
    # (c, r, numerator1, denominator1, ... numerator4, denominator4)
    # Summing the columns across calls to this function on an entire corpus
    # will produce a vector of statistics that can be used to compute GLEU
    def gleu_stats(self,i,r_ind=None):

      hlen = self.hlen
      rlen = self.rlens[i][r_ind]
      
      yield hlen
      yield rlen

      for n in range(1,self.order+1):
        h_ngrams = self.this_h_ngrams[n-1]
        s_ngrams = self.all_s_ngrams[i][n-1]
        r_ngrams = self.get_ngram_counts(self.refs[i][r_ind],n)

        s_ngram_diff = self.get_ngram_diff(s_ngrams,r_ngrams)

        yield max([ sum( (h_ngrams & r_ngrams).values() ) - \
                    sum( (h_ngrams & s_ngram_diff).values() ), 0 ])

        yield max([hlen+1-n, 0])

    # Compute GLEU from collected statistics obtained by call(s) to gleu_stats
    def gleu(self,stats,smooth=False):
        # smooth 0 counts for sentence-level scores
        if smooth :
            stats = [ s if s != 0 else 1 for s in stats ]
        if len(list(filter(lambda x: x==0, stats))) > 0:
            return 0
        (c, r) = stats[:2]
        log_gleu_prec = sum([math.log(float(x)/y)
                             for x,y in zip(stats[2::2],stats[3::2])]) / 4
        return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)

def get_gleu_stats(scores) :
    mean = np.mean(scores)
    std = np.std(scores)
    #ci = scipy.stats.norm.interval(0.95,loc=mean,scale=std)
    return [mean, std]
    

def write_file(file_path, elements):
    with open(file_path, 'w') as f:
        for element in elements:
            f.write(f"{element}\n")


def gleu(reference_dict, source_dict, hypothesis_dict, n=4, debug=False):
    """
    - reference
        Target language reference sentences.
        Multiple files for multiple references
    
    - source
        Source language source sentences
    
    - hypothesis
        Target language hypothesis sentences to evaluate
        (can be more than one file--the GLEU score of each
        file will be output separately). Use '-o -' to read
        hypotheses from stdin.",
    
    - n (int)
        Maximum order of ngrams (default: 4)
    
    - debug (bool)
        Debug; print sentence-level scores (default: False)
        
    - iter (int)
        Number of iterations to run (default: 500)
    """
    num_iterations = 1

    
    reference_list =  []
    source_list = []
    hypothesis_list = []

    for index, reference in reference_dict.items():
        if index in source_dict and index in hypothesis_dict:
            reference_list.append(reference_dict[index])
            source_list.append(source_dict[index])
            hypothesis_list.append(hypothesis_dict[index])

    with tempfile.TemporaryDirectory() as tmpdirname:
        reference = os.path.join(tmpdirname, 'reference.txt')
        source = os.path.join(tmpdirname, 'source.txt')
        hypothesis =  os.path.join(tmpdirname, 'hypothesis.txt')

        write_file(reference, reference_list)
        write_file(source, source_list)
        write_file(hypothesis, hypothesis_list)

        reference = [reference]

        # if there is only one reference, just do one iteration
            
        gleu_calculator = GLEU(n)

        gleu_calculator.load_sources(source)
        gleu_calculator.load_references(reference)

        instream = open(hypothesis)
        hyp = [line.split() for line in instream]

        # first generate a random list of indices, using a different seed
        # for each iteration
        indices = []
        for j in range(num_iterations) :
            random.seed(j*101)
            indices.append([random.randint(0,len(reference)-1)
                            for i in range(len(hyp))])

        if debug :
            print()
            print('===== Sentence-level scores =====')
            print('SID Mean Stdev 95%CI GLEU')

        iter_stats = [ [0 for i in range(2*n+2)]
                    for j in range(num_iterations) ]

        for i,h in enumerate(hyp) :

            gleu_calculator.load_hypothesis_sentence(h)
            # we are going to store the score of this sentence for each ref
            # so we don't have to recalculate them 500 times

            stats_by_ref = [ None for r in range(len(reference)) ]

            for j in range(num_iterations) :
                ref = indices[j][i]
                this_stats = stats_by_ref[ref]

                if this_stats is None :
                    this_stats = [ s for s in gleu_calculator.gleu_stats(
                        i,r_ind=ref) ]
                    stats_by_ref[ref] = this_stats

                iter_stats[j] = [ sum(scores)
                                for scores in zip(iter_stats[j], this_stats)]

            if debug:
                # sentence-level GLEU is the mean GLEU of the hypothesis
                # compared to each reference
                for r in range(len(reference)) :
                    if stats_by_ref[r] is None :
                        stats_by_ref[r] = [s for s in gleu_calculator.gleu_stats(
                            i,r_ind=r) ]

                print(i),
                print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats,smooth=True)
                                            for stats in stats_by_ref])))

        if debug:
            print('\n==== Overall score =====')
            print('Mean Stdev 95%CI GLEU')
            print(' '.join(get_gleu_stats([gleu_calculator.gleu(stats)
                                        for stats in iter_stats ])))
    
        return get_gleu_stats([gleu_calculator.gleu(stats)
                                for stats in iter_stats ])[0]