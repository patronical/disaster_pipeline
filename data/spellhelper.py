# spellhelper.py
# spelling helper class for text cleaning
# this is a wrapper class for frequency dictionary
# that complements peter norvig spell check routines
'''
Implementation Notes
For default case Download the frequency dictionary and put adjacent to this file
https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt
License: Apache Software License (MIT)
'''
import regex as re
import numpy as np
from collections import Counter

class Spellhelper:
    '''
    A class for frequency dictionary based spell correction and segmentation.
    '''
    import peter_norvig_spelling as pns

    def __init__(self, filename = 'frequency_dictionary_en_82_765.txt'):
        '''
        initialize counter by loading frequency dictionary file
        '''
        def LoadFD(file):
            '''
            extract frequency dictionary file
            transform data and load into counter
            return counter
            '''
            # extract
            with open(file, 'r') as myfile:
                DICT=myfile.read().replace('\n', '')

            # transform
            DICT2 = re.sub(r'(\d+)(\w+)', r'\1 \2', DICT.lower())
            d_tup = re.findall(r'(\w+)\s(\d+)', DICT2)

            # load
            C = Counter()
            for pair in d_tup:
                C[pair[0]] = int(pair[1])
        
            return C

        self.filename = filename
        self.counter = LoadFD(self.filename)


    def spellcheck(self, token):
        '''
        input token string
        check spelling
        return best effort spell check
        '''
        self.pns.COUNTS = self.counter
        spell_check = self.pns.correct(token)

        return spell_check


    def segcheck(self, token):
        '''
        input token string
        check segmentation
        return list of words as best effort segmentation check
        '''
        self.pns.COUNTS = self.counter
        segs = self.pns.segment(token)

        return segs


    def updatefreq(self, C_corpus):
        '''
        import corpus frequency counters
        build interpolation intervals
        convert corpus frequencies
        update english frequencies
        set counter to updated values
        '''
        C_english = self.counter
    
        # range variables
        eng_max = C_english.most_common()[0][1]+1
        eng_min = C_english.most_common()[-1][1]
        cor_max = C_corpus.most_common()[0][1]+1
        cor_min = C_corpus.most_common()[-1][1]
    
        # interpolation intervals
        eng_int = np.linspace(eng_min,eng_max,1000)
        cor_int = np.linspace(cor_min,cor_max,1000)
    
        # interpolator
        def IntFreq(token):
            '''
            import token
            lookup token frequency in corpus
            interpolate frequency in english
            round float frequency into integer
            assign updated counter
            '''
            cf = C_corpus[token]
            idx = np.max(np.where(cor_int<=cf))
            c_base = cor_int[idx]
            e_base = eng_int[idx]
            cspan = cor_int[idx+1]-c_base
            espan = eng_int[idx+1]-e_base
            f_intp = (cf - c_base)*(espan/cspan) + e_base
            return int(round(f_intp,0))
    
        # sort entries
        tok_updates = [token for token in C_corpus if token in C_english]
    
        # update frequencies
        C_eng = C_english.copy()
        for token in tok_updates:
            tf = IntFreq(token)
            C_eng[token] = tf
        
        self.counter = C_eng

        print('Frequency dictionary updated.')


    def addwords(self, C_corpus, new_word_list):
        '''
        import english and corpus frequency counters
        import words not in English Dict but in Corpus
        build interpolation intervals
        convert corpus frequencies
        add new words to english dict
        set counter to updated values
        '''
        C_english = self.counter
    
        # range variables
        eng_max = C_english.most_common()[0][1]+1
        eng_min = C_english.most_common()[-1][1]
        cor_max = C_corpus.most_common()[0][1]+1
        cor_min = C_corpus.most_common()[-1][1]
    
        # interpolation intervals
        eng_int = np.linspace(eng_min,eng_max,1000)
        cor_int = np.linspace(cor_min,cor_max,1000)
    
        # interpolator
        def IntFreq(token):
            '''
            import token
            lookup token frequency in corpus
            interpolate frequency in english
            round float frequency into integer
            return english frequency
            '''
            cf = C_corpus[token]
            idx = np.max(np.where(cor_int<=cf))
            c_base = cor_int[idx]
            e_base = eng_int[idx]
            cspan = cor_int[idx+1]-c_base
            espan = eng_int[idx+1]-e_base
            f_intp = (cf - c_base)*(espan/cspan) + e_base
            return int(round(f_intp,0))
    
        # input new words
        C_eng = C_english.copy()
        news = [word for word in new_word_list if word in C_corpus]
        for word in news:
            tf = IntFreq(word)
            C_eng[word] = tf
        
        self.counter = C_eng

        print('New words added to frequency dictionary.')


    def savefreqdict(self, filename):
        '''
        import filename
        format counter as frequency dictionary
        save the Counter as frequency dictionary
        '''
        C = self.counter

        #Format Counter for Output
        Cout = sorted(C.items(), key=lambda pair: pair[1], reverse=True)

        # Save for later re-run from start in place of generic dictionary
        with open(filename, encoding='utf-8', mode='w') as f: 
            for tag, count in Cout:  
                f.write('{} {}\n'.format(tag, count))

        print('Spell Check Counter saved to ' + filename)

