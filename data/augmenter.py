# augmenter.py
# augment data for disaster corpus
# clean text using cleaner
# create BOW's from POS tags for each class
# create simulated messages for class from its BOW's
# sample entries out of class
# augment sampled message with simulated message
# augment class flag to complement augmented simulation


# import libraries
import sys
import pandas as pd
import random
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from itertools import chain
import cleaner

# text processing routine

def cleanmess(text):
  '''
  import message
  clean the message
  return the cleaned message
  '''
  sents = sent_tokenize(text)
  all_sents = []
  for sent in sents:
      clean = cleaner.CleanText(sent) + '.'
      all_sents.append(clean)

  return ' '.join(all_sents)

def tokenize(text):
    '''
    input clean message
    tokenize by sentences
    stem and remove stop words
    return cleaned tokens of text
    '''
    stop_words = stopwords.words("english") + ['.']
    lemmatizer = WordNetLemmatizer()

    sents = sent_tokenize(text)
    all_tokens = []
    for sent in sents:
        # convert previously cleaned sentence to tokens
        raw_tokens = nltk.word_tokenize(sent)
        pos_tags = pos_tag(raw_tokens)
        #process tokens
        stp_tokens = [(t,p) for t,p in pos_tags if t not in stop_words]
        tokens = [(lemmatizer.lemmatize(t),p) for t,p in stp_tokens]
        all_tokens += tokens

    return all_tokens


# utilities for message simulation

def BuildFreqDict(messages):
    '''
    Import series of messages
    Tokenize messages
    Build Token Count dictionaryt,p
    Return Token Count dictionary
    '''
    texts = messages.apply(tokenize).tolist()
    dwf = Counter(chain.from_iterable(texts))
    return dwf


def BuildBOWs(category, df_corpus):
    '''
    import category and corpus
    build frequency dictionary
    sort by part of speech
    build bag of words
    shuffle bag of words
    consolidate bags
    return list of bags
    '''
    # nltk POS notes:
    #
    # Here we see that 'and' is CC, a coordinating conjunction; 
    # 'now' and 'completely' are RB, or adverbs;
    # 'for' is IN, a preposition; 
    # 'something' is NN, a noun; 
    # and 'different' is JJ, an adjective.
    #
    # https://www.nltk.org/book/ch05.html
    #
    # build frequency dictionary
    df = df_corpus.copy()
    dwf = BuildFreqDict(df[df[category] == 1]['message'])
    # build item lists
    nouns =[item for item in dwf if item[1] in ['NN','NNS','VBG','VBD']]
    verbs = [item for item in dwf if item[1] in ['VB','VBN','VBG','VBD']]
    adjectives = [item for item in dwf if item[1] in ['JJ']]
    adverbs = [item for item in dwf if item[1] in ['RB']]
    pos_items = [nouns,verbs,adjectives,adverbs]
    # build bags of words
    bow_list = []
    for item_list in pos_items:
        bow = []
        for item in item_list:
            bow += [item[0]]* dwf[item]
        random.shuffle(bow)
        bow_list.append(bow.copy())
        
    return bow_list


def BuildSimulatedMessage(bow_list):
    '''
    import bag of word lists
    random sample sentences
    random sample words
    return message
    '''
    # Basic sentence structures
    # Subject–Verb
    # Subject–Verb–ObjectS=
    # Subject–Verb–Adverb
    # Subject–Verb–Noun
    # https://www.wordy.com/writers-workshop/basic-english-sentence-structure/
    
    # sorted lists
    nouns,verbs,adjectives,adverbs = bow_list

    def RandomSent():
        '''
        generate a random sentence
        '''
        last_part = ['', random.choice(nouns), random.choice(adjectives),
                     random.choice(adverbs), random.choice(nouns)]
        sentence = '{} {} {}'.format(random.choice(nouns), 
                                     random.choice(verbs),
                                     random.choice(last_part))
        return sentence.rstrip()

    min_length = random.randint(74,200)
    message = ''
    while len(message) < min_length:
        message += random.choice([ RandomSent() + '. ',
                                   RandomSent() + ' and ' + RandomSent() + '. ',
                                   RandomSent() + ' or ' + RandomSent() + '. '])
    return message


def CreateChildAloneMessage():
    '''
    Simulate a random child alone message.
    Return random message string.
    '''
    params =[['I','he','she','somebody','a stranger','authorities','helper','group'],
             ['found','lost','met','saw','lost track of', 'discovered','got'],
             ['got lost', 'abducted', 'was abandoned', 'got in an accident','is hurt and alone','is panicking'],
             ['child','boy','girl','son','daughter','kid','teenager'],
             ['Emma', 'Liam','Olivia','Noah','Ava','Elijah', 'Isabella', 'Logan'],
             ['four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen'],
             ['black','brown','red','blond','bald','pink','blue'],
             ['swimsuit','white shirt','red shoes','green jacket','jeans','baseball hat'],
             ['wearing glasses','skinny','fat','loud','shy','tall','short'],
             ['coughing','sneezing','hyperactive','cold','sweating','hurt','healthy'],
             ['around noon','this morning','in the afternoon','after lunch', '3pm','10 a.m.','last night'],
             ['yesterday', 'day before yesterday', 'Friday', 'sat', 'today', 'now', 'this week'],
             ['grocery store','park','pool','school','beach','mall','street','sidewalk']]

    who_rept, obs_verb, event_verb, event_noun,\
    kid_name, kid_age, kid_hair, kid_outfit,\
    kid_feat, kid_health, event_time, event_day,\
    event_place = [random.choice(l) for l in params]

    sents = [['A {} {}. '.format(event_noun, event_verb),
              '{} {} a {}. '.format(who_rept,obs_verb,event_noun),
              'A {} {} named {}. '.format(event_noun, event_verb, kid_name)],
             ['{} and {} hair. '.format(kid_age, kid_hair),
              '{} hair and {}. '.format(kid_hair, kid_health),
              '{} wearing {} and {} years. '.format(kid_health, kid_outfit, kid_age)],
             ['Here {} at {}.'.format(event_time, event_place),
              'Was at the {}, {}. '. format(event_place, event_time),
              'Need help {}, please come to {}, {}. '.format(event_day, event_place, event_time)]]
    
    message = ('').join([random.choice(s) for s in sents])
    
    return message


def BuildCAs(dft):
    '''
    import train dataframe
    slice by category not true
    sample entries as copies
    update categories
    add simulated messages
    return piggy backed df
    '''
    # Simulate 'child_alone' message piggy-backed on other samples
    id_start = max(dft['id'].values) + 1
    df_ca = dft[(dft['related']==1) & (dft['offer']!=1)].sample(1000).copy()
    df_ca['id'] = [i for i in range(id_start, id_start+1000)]
    df_ca['aid_related'] = 1
    df_ca['child_alone'] = 1
    df_ca['message'] = df_ca['message'].map(lambda x: CreateChildAloneMessage() + x)
    df_ca = df_ca.reset_index(drop=True)
    
    return df_ca


def BuildSims(df_aug, dft):
    '''
    import augment and training df's
    itterate by low count categories
    slice by category not true
    sample entries as copies
    update categories
    add simulated messages
    return piggy backed df
    '''
    # work on copy
    dfa = df_aug.copy()
    # build dictionary of partner categories for low count categories
    partner = {'offer': ['aid_related'],
               'search_and_rescue':['aid_related'],
               'security':['aid_related','direct_report'],
               'military':['aid_related','direct_report'],
               'clothing':['aid_related'],
               'money':['request'],
               'missing_people':['aid_related'],
               'refugees':['aid_related'],
               'death':['aid_related'],
               'electricity':['request','aid_related','direct_report'],
               'tools':['aid_related'],
               'hospitals':['aid_related','infrastructure_related'],
               'shops':['aid_related','food'],
               'aid_centers':['aid_related','infrastructure_related'],
               'other_infrastructure':['aid_related','infrastructure_related','request'],
               'fire':['weather_related'],
               'cold':['weather_related']}
    
    # Simulate messages and augment them as piggy-backed on other samples
    for cat in list(partner.keys()):
        id_start = max(dfa['id'].values) + 1
        dfc = dft[(dft['related']==1) & (dft[cat]!=1)].sample(1000).copy()
        dfc['id'] = [i for i in range(id_start, id_start+1000)]
        dfc[cat] = 1
        for prdnr in partner[cat]:
            dfc[prdnr] = 1
        # tack on simulated messages
        bow = BuildBOWs(cat, dft)
        dfc['message'] = dfc['message'].map(lambda x: BuildSimulatedMessage(bow) + x)
        dfc = dfc.reset_index(drop=True)
        dfa = pd.concat([dfa, dfc], axis = 0, ignore_index=True)
    dfa.reset_index(drop=True, inplace=True)
    
    return dfa


def simulate(df_dirty):
    '''
    import formatted df
    clean messages
    split off validation set
    simulate data for empty categories
    return clean, validation, training, and simulated df's
    '''
    dfc = df_dirty.copy()
    # clean
    dfc['message'] = dfc.message.apply(cleanmess)

    # category column names
    category_colnames = dfc.columns[2:].tolist()

    # start with some related zero's
    dfv = dfc[dfc['related']==0].sample(36).copy()

    # add category entries to validation set
    idx_list = dfv.index.tolist()
    # add in some one's for each target category
    for cat in category_colnames:
        # manage child alone has no entries
        if cat != 'child_alone':
            df_slice = dfc[dfc[cat]==1].copy()
            samp = df_slice.sample()
            while samp.index[0] in dfv.index:
                samp = df_slice.sample() 
            dfv = pd.concat([dfv, samp], axis=0)
        else:
            pass
    
    # independant train and validation data 
    dft = dfc[~dfc.index.isin(dfv.index)].copy()

    # Simulate child alone Messages
    df_aug = BuildCAs(dft)

    # sample a 'child alone' entry for validation
    samp = df_aug.sample()
    dfv = pd.concat([dfv, samp], axis=0)

    # Simulate low count class messages
    dfa = BuildSims(df_aug, dft)

    return dfc, dfv, dft, dfa