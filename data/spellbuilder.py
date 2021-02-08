# spellbuilder.py
# builds frequency dictionary and lookup dictionary for customized spell corrections

import os
import sys
import regex as re
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from collections import defaultdict
import pickle
from spellhelper import Spellhelper


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
    sort token
    preclean by type
    return cleaned token list
    '''
    if bool(re.match("^[a-zA-Z0-9]*$", token)):
        if bool(re.search(r'\d', token)):
            clean = CleanNumeric(token)
        else:
            clean = [token]
    else:
        clean = CleanSymbol(token)
    return clean


def LoadCorpus(df):
    '''
    Load paragraphs into dataframe
    Convert paragraphs to tokens
    Return counter and tokens
    '''
    # Load Corpus of Paragraphs
    df_mess = df.copy()

    # put all messages into a single string
    TEXT=""
    for i,nrows in df_mess.iterrows():
        TEXT += (nrows['message'])
        
    # tokenize the text string
    raw_tokens = nltk.word_tokenize(TEXT.lower())
    
    # pre-clean the tokens
    pc_tokens = []
    for token in raw_tokens:
        pc_tokens += PreClean(token)
    
    #build counter of corpus tokens
    C = nltk.FreqDist(pc_tokens)
    
    return C, pc_tokens


def FindTypos(speller, C_corpus):
    '''
    import speller function and Corpus counter
    build a spell correction dictionary
    return spell correction dictionary and typos list
    '''
    scdict = defaultdict(str)
    sp_typos = []
    for token in sorted(C_corpus.keys()):
        # spelling okay
        if token in speller.counter:
            scdict[token] = token
        # typos
        else:
            sp_typos.append(token)
                
    return scdict, sp_typos


def ApplyCorr(string_dict, tokens):
    '''
    import string dictionary
    import list of tokens to correct
    compiled corrected version
    render from token list to string and back
    return list of corrected tokens
    '''
    # input string corrections into list
    corr_strings = []
    for token in tokens:
        if token in string_dict:
            corr_strings.append(string_dict[token])
        else:
            corr_strings.append(token)
            
    # convert list to string then into tokens    
    big_string = ' '.join(corr_strings)
    corr_tokens = nltk.word_tokenize(big_string)
    
    return corr_tokens


def FindTypos(speller, C_corpus):
    '''
    import speller function and Corpus counter
    build an string lookup dictionary of correctly spelled tokens
    return string lookup dictionary and typos list
    '''
    scdict = defaultdict(str)
    sp_typos = []
    for token in sorted(C_corpus.keys()):
        # spelling okay
        if token in speller.counter:
            scdict[token] = token
        # typos
        else:
            sp_typos.append(token)
                
    return sp_typos, scdict


def SpellSeg(speller, token):
    '''
    import spelling function
    import token
    attempt spelling correction and score
    attempt segmentation correction and score
    compare scores and select winning score as correction
    return corrected string and score
    '''
    # spell check
    spell_cor = speller.spellcheck(token)
    if spell_cor != token:
        sc_score = speller.counter[spell_cor]
    else:
        sc_score = 0

    # seg check
    segs = speller.segcheck(token)
    valid = len(''.join(segs))/len(segs)
    if valid > 1.4:
        best = [t for t in segs if len(t)>1]
        seg_cor = ' '.join(best)
        scores = [speller.counter[t] for t in best]
        sg_score = np.mean(scores)
    else:
        sg_score = 0
        
    # flunked out of being corrected
    if (sc_score==0) & (sg_score==0):
        bstring = token
        score = 0
        
    # correction found
    else:
        if sc_score > sg_score:
            bstring = spell_cor
            score = sc_score
        else:
            bstring = seg_cor
            score = sg_score
    
    return bstring, score


def FirstPass(speller, string_dict, typos_list):
    '''
    import speller function 
    import string lookup dictionary
    import typo token list
    attempt spelling correction and score
    attempt segmentation correction and score
    compare scores and select winning score as correction
    add best effort correction to string dictionary
    return dictionary and flunked token list
    '''
    flunk_list = []
    for token in typos_list:
        # attempt correction
        bstring, score = SpellSeg(speller, token)
        # flunkies
        if score == 0:
            flunk_list.append(token)
        else:
            # update correction
            string_dict[token] = bstring
            for seg in bstring.split():
                if seg not in string_dict:
                    string_dict[seg] = seg
            print(token, ' -> ', bstring)

    return string_dict, flunk_list


def SecondPass(speller, string_dict, flunk_list):
    '''
    import speller function
    import string dictionary
    import flunked token list
    iterate through flunked texts
    split off probable noise
    permutate splitting within text 
    limit splits out to 30 characters
    attempt corrections on split sides
    recombine correction results
    adopt optimal split result for text
    return dictionary and noise
    '''
    # combine split with segmenting
    noise = []
    for text in flunk_list:
        # assume shorts are noise
        if len(text) < 8:
            noise.append(text)
        # attempt to find best correction pair
        else:
            # noise assumption as base case
            maxscore = 0
            maxres = text
            # slice text and attempt fix
            split_range = min(30, len(text))
            split_min = int(split_range*0.4)
            split_max = int(split_range*0.6)+1
            for i in range(split_min,split_max):
                # split
                text_left = text[:i]
                text_right = text[i:]
                # attempt corrections
                tl_string, tl_score = SpellSeg(speller, text_left)
                tr_string, tr_score = SpellSeg(speller, text_right)
                split_result = ''
                # recombine
                if tl_score > 0:
                    split_result += tl_string + ' '
                if tr_score > 0:
                    split_result += tr_string
                score = tl_score + tr_score
                # looking for the optimal scoring split pair
                if score > maxscore:
                    maxscore = score
                    maxres = split_result.strip()
            # process optimal split pair result
            if maxres == text:
                noise.append(text)
            else:
                print(text, ' -> ', maxres)
                for t in maxres.split():
                    if t not in string_dict:
                        string_dict[t] = t
                string_dict[text] = maxres
       
    return string_dict, noise


def BuildFiles(df):
    '''
    import corpus
    pipeline spell check, segment, complex segment process
    generate frequency and lookup dictionaries
    print results
    save dictionaries
    return filenames
    '''
    # Instantiate Spelling tool
    speller = Spellhelper()

    # big text corpus
    C_corpus, corpus_tokens = LoadCorpus(df)

    # estimate number of unique spelling errors
    sp_errors = len([t for t in C_corpus if t not in speller.counter])

    # important words in corpus that may have higher frequency
    # in corpus then in standard english usage
    # could be names or acronyms that may not be commonly known
    # this is a sample, a domain expert is needed to double check these
    new_words = ['ayuda', 'bbc', 'cbs', 'center', 'centered', 
                 'cyber', 'debris', 'donde', 'euro', 'fecal', 
                 'feces', 'fema', 'foxnews', 'franken', 'fyi', 
                 'giardia', 'gmo', 'google', 'gps', 'gui', 
                 'haiti', 'http', 'https', 'hungry', 'mbc', 
                 'meds', 'msnbc', 'nbc', 'nyc', 'omg', 'ppe', 
                 'redcross', 'reiki', 'rescue', 'sandy', 'scary', 
                 'skyfm', 'skynews', 'sulfate', 'sulfide', 'sulfur', 
                 'tele', 'terre', 'tumblr', 'tweeting', 'tweets', 
                 'twitter', 'ucla', 'unicef', 'vegan', 'volcano', 
                 'wikipedia','wtf']

    # find typos
    sp_typos, scdict = FindTypos(speller, C_corpus)

    # attempt first pass corrections
    sc_dict, flunk_list = FirstPass(speller, scdict, sp_typos)

    # apply corrections
    sc_tokens = ApplyCorr(scdict, corpus_tokens)

    #build counter of corpus tokens
    C_spell = nltk.FreqDist(sc_tokens)

    # update speller frequency dict with corpus words
    speller.updatefreq(C_spell)

    # Process Second Pass Segmentations and Spelling
    sc_dict, noise = SecondPass(speller, sc_dict, flunk_list)

    # apply corrections to corpus
    splitseg_tokens = ApplyCorr(sc_dict, sc_tokens)

    #build counter of corpus tokens
    C_splits = nltk.FreqDist(splitseg_tokens)

    # update frequencies of corpus words
    speller.updatefreq(C_splits)

    # update  noise entries
    for n in noise:
        scdict[n] = 'noise'

    # print results
    print('Spelling Error Set Size: ', sp_errors)
    print('1st Pass Corrections: ', sp_errors - len(flunk_list))
    print('2nd Pass Split Corrections: ', len(flunk_list)-len(noise))
    print('Best Guess Possible Noise: ', len(noise))
    print('Initial Vocab Count', len(C_corpus))
    print('Final Vocab Count:', len(C_splits))

    # filename specification
    fd_file = 'data/freq_dict.txt'
    lu_file = 'data/lookup_dict.pkl'

    # save frequency dictionary
    speller.savefreqdict(fd_file)

    # pickle lookuup dictionary  
    with open(lu_file, 'wb') as handle:
        pickle.dump(scdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return fd_file, lu_file


