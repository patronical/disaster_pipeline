import sys
sys.path.insert(1, '../data')
sys.path.insert(1, '../models')
import os
os.path.join('../models', 'train_classifier.py')
import json
import plotly
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import dill as pickle
import joblib
from spellhelper import Spellhelper
import cleaner
from flask import Flask
from flask import render_template, request, jsonify
from PIL import Image
import plotly.graph_objects as gobj

app = Flask(__name__, template_folder='template')

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


def FiltTokenize(text):
    '''
    input text string
    clean and lower case characters of string
    tokenize text
    lematize and remove stop words
    add bigrams
    return cleaned tokens and bigrams
    '''
    stop_words = stopwords.words("english") + ['.']
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


# load model
global dtf
with open(os.path.join('../models','classifier.pkl'),'rb') as f:
    model = pickle.load(f)
# load spellcheck utilities    
with open(os.path.join('../models','classifier_dict.pkl'),'rb') as fd:
    dtf = pickle.load(fd)
with open(os.path.join('../data','lookup_dict.pkl'), 'rb') as handle:
        cleaner.corr_dict = pickle.load(handle)
cleaner.speller = Spellhelper(os.path.join('../data','freq_dict.txt'))

# import visuals
img = Image.open(os.path.join('../models','train_disaster.png')) 

#displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
                
    # create visuals
    # https://plotly.com/python/images/
    
    # Create figure
    fig = gobj.Figure()
    
    # Constants
    img_width = 1334
    img_height = 908
    scale_factor = 0.5
    
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        gobj.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    
    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )
    
    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img,
        )
    )
    
    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    
    # encode plotly graphs in JSON
    ids = ["fig"]
    graphJSON = json.dumps([fig], cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # clean the input string
    clean = cleanmess(query)
    # use model to predict classification for cleaned message
    classification_labels = model.predict([clean])[0]
    t_cols = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 
              'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 
              'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 
              'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 
              'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    classification_results = dict(zip(t_cols, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()