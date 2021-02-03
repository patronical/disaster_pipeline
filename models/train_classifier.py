# train_classifier.py
# loads data from database
# splits dataset into training and test sets
# builds a text processing and machine learning pipeline
# trains and tunes a model using grid search
# outputs results on the test set
# exports the final model as a pickle file
import sys
import os
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import dill as pickle
import buildvisuals
import joblib

try:
    import builtins
except ImportError:
    import __builtin__ as builtins


def LoadData(filename):
    '''
    load data from database into dataframe
    identify target columns
    test that there's 36 columns
    return target columns and dataframe
    '''
    # load
    sys.path.insert(1, '../data')
    conn = sqlite3.connect(filename)
    df = pd.read_sql('SELECT * FROM MessCatRaw', con = conn)
    # targets
    t_cols = df.columns[2:-2].tolist()
    # test
    assert (len(t_cols) == 36)

    return t_cols, df


def BareTokenize(text):
    '''
    input text string
    clean and lower case characters of string
    tokenize text
    lematize and remove stop words
    add bigrams
    return cleaned tokens and bigrams
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize
    s = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    tokens = word_tokenize(s)
    # stemming
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # add bi-grams
    bigrams = [a + ' ' + b for a,b in list(nltk.bigrams(tokens))]
    # combine
    all_tokens = tokens + bigrams

    return all_tokens


def BuildFreqFilter(df):
    '''
    import datframe
    tokenize messages
    build counter of word frequencies
    build filter dictionary
    save filter dictionary
    test that the vocab is over 30,000 tokens
    return filter dictionary
    '''
    messages = df['message']
    texts = messages.apply(BareTokenize).tolist()
    dwf = Counter(chain.from_iterable(texts))
    dtf = {x: count for x, count in dwf.items() if count >= 3}
    assert(len(dtf) > 30000)
    return dtf


def FiltTokenize(text):
    '''
    input text string and filter dictionary
    tokenize
    apply filter
    return list of filtered tokens
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize
    s = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    tokens = word_tokenize(s)
    # stemming
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # add bi-grams
    bigrams = [a + ' ' + b for a,b in list(nltk.bigrams(tokens))]
    # combine
    all_tokens = tokens + bigrams
    # filter
    filt_tokens = [t for t in tokens if t in dtf]
    
    return filt_tokens


def SplitData(df, t_cols):
    '''
    input dataframe and targets
    split off validation set
    split off train and test data
    assign targets and features for validation
    assign targets and features for train and test
    split train and test data
    return datasets
    '''
    # split off validation set
    dfv = df[df['val']==1].copy()
    # independant train and test data 
    dft = df[df['val']==0].copy()

    # assign features and targets for validation
    Xval = dfv['message'].copy()
    Yval = dfv[t_cols].copy()

    # assign features and targets for train and test
    X = dft['message'].copy()
    Y = dft[t_cols].copy()

    # test train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=42)

    return dft,Xval,Yval,X_train,Y_train,X_test,Y_test


def BuildModel(X_train,Y_train, dtf):
    '''
    import train dataset and filter 
    builds a text processing and machine learning pipeline
    trains and tunes a model using grid search
    returns optimized model
    '''
    # basic pipeline
    over_samp = SMOTE()
    under_samp = RandomUnderSampler()
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf3 = make_pipeline(over_samp, under_samp, logreg)
    pipe3 = Pipeline([('tfidf', TfidfVectorizer(tokenizer=FiltTokenize, 
                                                use_idf=True, 
                                                sublinear_tf=True, 
                                                ngram_range = (1,1))),
                      ('skb',  SelectKBest(chi2, k=10000)),
                      ('clf3', MultiOutputClassifier(clf3, n_jobs=-1))])
    
    # grid search definition
    parameters = {'skb__k': [10000,'all']}
    model = GridSearchCV(pipe3, param_grid=parameters)

    return model


def PrintClassReports(Y_predictions, Y_target):
    '''
    input classification predictions and targets
    compute figures
    print summary and estimate
    '''
    # classification reports
    cols = Y_target.columns.tolist()
    Y_targ = Y_target.values
    print('------------------------------------------------------')
    for i in range(36):
        print(cols[i])
        print(classification_report(Y_targ.T[i],Y_predictions.T[i],
                                    zero_division=0))
        print('------------------------------------------------------')


def SaveModel(model, filename):
    '''
    import model, filter dictionary, and filename
    save for later
    '''
    # save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    # save dictionary
    d_filename = filename[:-4] +'_dict.pkl'
    with open(d_filename, 'wb') as fd:
        pickle.dump(dtf, fd)

    print('Model saved to : '+ filename)


def main():
    '''
    Load data, preprocess, split, build, train, evaluate, save, visualize.
    '''
    if len(sys.argv) != 3:
        print('Please provide the filepaths of the database and model '\
              '\n\nExample: DisasterResponse.db Logregmod.pkl')
        sys.exit()

    # pipeline
    database_filepath = sys.argv[1]
    model_filepath = sys.argv[2]

    # load
    print('loading...')
    t_cols, df = LoadData(database_filepath)

    # preprocess
    print('preprocessing...')
    global dtf
    dtf = BuildFreqFilter(df)

    # split
    print('splitting data...')
    dft,Xval,Yval,X_train,Y_train,X_test,Y_test = SplitData(df, t_cols)

    # model
    print('building model...')
    model = BuildModel(X_train,Y_train,dtf)
    
    # train model
    print('training model...')
    model.fit(X_train, Y_train)
    
    # predict and evaluate
    print('evaluating model...')
    Y_pred = model.predict(X_test)
    PrintClassReports(Y_pred, Y_test)

    # save
    print('saving model...')
    # save model
    filename = model_filepath
    SaveModel(model, filename)

    # build visuals
    print('building train data visuals...')
    dft['prep'] = dft['message'].apply(FiltTokenize)
    buildvisuals.dft = dft
    buildvisuals.BuildFig(Y_train, 'models/train_disaster.png')
    
    print('processing complete.')


if __name__ == '__main__':
    main()

