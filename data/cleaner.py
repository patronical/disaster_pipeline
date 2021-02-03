# cleaner.py
# routines that uses custom frequency dicitionary 
# and string to string lookup dictionary to clean
# text messages used in both classification model
# and text messages used for model prediction

import os
import regex as re
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from collections import defaultdict
from spellhelper import Spellhelper


# preclean routines

def CleanNumeric(token):
    '''
    input numeric token
    clean the token
    return simplified token list
    '''
    n_map = {ord(c): " " for c in "0123456789"}
    alphas = token.translate(n_map)
    toks = ['number'] + nltk.word_tokenize(alphas)
    return toks


def CleanSymbol(token):
    '''
    input symbolic token
    clean the token
    return simplified string
    '''
    #simplify paths
    if (token.count('/')>2)|(token[:2]=='//')|(token[:4]=='www.'):
        clean_tok = ['path']
    else:
        # remove special characters
        sc_map = {ord(c): " " for c in \
                  "!@#$£¥¢©®™§†±≠%^&*()[]{};:,‚./<>?\|¨´'`~-=_+¬∞µ•√∏∫°áàâæëéîñöōüû"}
        clear_sym = token.translate(sc_map)
        ctoks = nltk.word_tokenize(clear_sym)
        clean_tok = []
        for tok in ctoks:
            if not bool(re.search(r'\d', tok)):
                clean_tok.append(tok)
            # process numericals
            else:
                clean_tok += CleanNumeric(tok)
    return clean_tok


def PreClean(token):
    '''
    import token
    preclean by type
    return pre-cleaned token list
    '''
    if bool(re.match("^[a-zA-Z0-9]*$", token)):
        if bool(re.search(r'\d', token)):
            clean = CleanNumeric(token)
        else:
            clean = [token]
    else:
        clean = CleanSymbol(token)
    return clean


# spell check routine

def SpellSeg(token):
    '''
    import token
    attempt simple spell correction
    then attempt simple segmentation
    then attempt complicate segmentation
    return best effort correction of the three
    '''
    # simple spell correction
    sc = speller.spellcheck(token)
    if sc != token:
        return sc

    # simple segmentation after simple spell check didn't work
    else:
        segs = speller.segcheck(token)
        score = len(''.join(segs))/len(segs)
        if score > 1.4:
            best = [t for t in segs if len(t)>1]
            bstring = ' '.join(best)
            return bstring

        # complicated segmentation indicated by low score
        else:
            # assume shorts are noise
            if len(token) < 8:
               return 'noise'
            # attempt to find conjoined typo and fix
            else:
                # name assumption as base case
                maxscore = 0
                bstring = 'name'
                # slice token and attempt fix
                for i in range(1, min(30, len(token)-3)):
                    splitcorr = [speller.spellcheck(token[:i]), 
                                 speller.spellcheck(token[i:])]
                    segA = speller.segcheck(splitcorr[0])
                    segB = speller.segcheck(splitcorr[1])
                    segs = segA + segB
                    toks = [t for t in segs if len(t)>1 and t in speller.counter]
                    # found something
                    if len(toks) > 0:
                        score = len(''.join(toks))/len(toks)
                    # nothing worthwhile found
                    else:
                        score = 0
                    # update if something good was found
                    if score > maxscore:
                        maxscore = score
                        bstring = ' '.join(toks)

                return bstring


def PreCleanMess(text):
    '''
    import text
    tokenize text
    preclean tokens
    return cleaned token list
    '''
    dirty_tokens = nltk.word_tokenize(text.lower())
    preclean_tok = []
    for token in dirty_tokens:
        preclean_tok += PreClean(token)
    return preclean_tok


def CleanText(text):
    '''
    import text
    tokenize
    preclean text into tokens
    check tokens in lookup or frequency dict
    return combined checked tokens put into string
    '''
    # preclean
    tokens = PreCleanMess(text)

    # spell check
    cleantokens = []
    for token in tokens:

        # attempt simple string to string lookup
        if token in corr_dict:
            cleantokens.append(corr_dict[token])

        # token not in corpus check via frequency dictionary
        else:
            cleantokens.append(SpellSeg(token))

    return ' '.join(cleantokens)
