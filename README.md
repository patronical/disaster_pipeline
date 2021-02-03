# disaster_pipeline
Udacity Data Science Nanodegree Project
# Disaster Response Pipeline 

## Table of Contents
1. [Description](#description)
2. [Environment](#environment)
	1. [Get Code](#get_code)
	3. [Run Code](#run_code)
	4. [Notebooks](#notebooks)
	5. [Files](#files)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Results](#results)

<a name="description"></a>
## Description

This Project utilizes a collection of messages to build a classifier that can categorize messages.
The collection of messages was provided by Figure Eight and is an example of a supervised learning.
This is deemed supervised learning in that the collection of messages has been previously classified.
In other words, the classifier here is not looking for clusters and finding it's own categories.
The messages can be classified as relevant in a mixture of thirty six different categories. 

We train and test a classifier using these pre-classified categorizations in the hope that in the end,
the classifier will likewise classify the messages in a similar way.  Many of these messages seem to 
come from real disaster events.  The ability to classify quickly a glut of messages during a disaster 
might enable more timely routing of relevant resources to the needs at hand.

This project consists of three basic sections:

1. Processing data

In a nutshell, this section extracts data from the Figure 8 source files, transform the data in
ways comensurate with the goal at hand, and loads it into a database for the classifier section.

The extraction portion begins with loading two csv (comma separated value) files. One is for messages,
and the other is for categories.  These are joined to form approximately 26,386 rows with 36 categories
of interest for the classifier.

The transform portion contains numerous steps.  One is to convert categorical classifications into 
numbers easy for the computer to understand, one for yes and zero for no.  We drop duplicates and
records with no data (Nan's).  We also drop Related class category 'twos' as potentially nonsense
which does not seem useful to the goal here.  Once formatted, there's some regex module scrubbing
that clears numerics and symbols and urls, etc.  One of the most significant cleaning steps relates 
to spell check correcting the messages.  Since disasters contain their special spin on the English
language, we need to produce a custom dictionary that can make sense of the messages.  Two python
objects are made for this, a Frequency dictionary with correctly spelled terms keyed to occurance
count values, and a Lookup dictionary with corpus strings keyed to correctly spelled or segmented
string values.  The string to string lookups are significantly faster in finding corrections than
the frequency dictionary spell correction algorithms.  Finally, the long messages are truncated
to a reasonable amount of tokens.

Unfortunately the dataset is heavily imbalanced, meaning some of the classifications are not well
represented to an extent that sufficient training can occur.  So an additional transform step 
relates to augmenting the classification data.  Here, a bag of words approach is used to form
randomly generated sentences that are tacked onto randomly sampled entries to bump classifications
data for the low-runner categories.  Furthermore, a class that contained no data, "child_alone",
was simulated using a ad-hoc random sentence approach based upon lost child best practices.
The execution of this project seems unusual in this approach, however classification performance
was found to improve by this striving.  It is an example of natural language oversampling.  
The number of rows grows from 26,000 to 44,000.

Lastly, the extracted and transformed data is loaded into an SQL database for the classifier section.


2. Building Classifier

Here, the ETL data from the previous section is loaded into a machine learning pipeline
that works to optimize data feature extraction for improved classification performance.

Initially the corpus data is piped through a conventional tokenizer that removes stop words, 
lematizes, and generates bigrams from the messages.  These tokens are then frequency counted
for the generation of a token frequency filter.  The filter removes tokens that exhibit low
frequency in the corpus, say less than 5 counts.  The filter is a dictionary where tokens
are checked as a key or elsed dropped from participating.  The resulting tokenize function
then adds this filtering step to the conventional tokenizer to become a custom tokenizer.

Validation tags added during the augmentation step are used to split off never before seen
data for assessment of the classifier as a finishing step.  There's a conventional test 
train split for the remaining rows for piping into the model for training.

Imbalance learn modules are used as components of the model pipeline: SMOTE oversampling,
random undersampling, TfidfVecorizer (utilizing the custom tokenizer), a Chi-square feature
selector, and MultiOutputClassifier based upon Logistic Regression are plumbed into a
Grid Search that seeks the best Chi-square feature counts, generally more are found to
perform better.

The model performance is revealed after training using classification reports.  
Here, the recall of the low runner categories seems mediocre at best, seems the consensus on 
this data set.  For target one classifications, the optimized recall figure (accessible in 
the ML Pipeline Preparation Jupyter Notebook), is ~0.75, with balanced accuracy around ~0.83.  
Here Validation results agreed, indicating that the model was not over trained.

With the training data accessible, three follow on visualizations were built to convey to the 
webpage.  One visualization is a histogram of the train data message counts, and two other are 
token count histograms for "related" and "unrelated" categories.  These visualizations are 
saved to an image file for the web app to display.

3. Web App 

In this section, components of the previous sections are utilized for the purpose of
classifying messages into disaster categories.  

The three visualizations imaged together are ported into a plotly figure upon the first page.
There's a text query that enables input of a message.  Behind the scenes, this message string
goes through similar steps as used in the ETL.  There's a cleaner step and spell check corrector
before heading through the custom tokenizer with frequency filter - as utilized by the model
pipeline for classification purposes.  the classification results are indicated on the web app.


<a name="environment"></a>
## Environment

For running this on a local computer:

This is a Python 3.6+ based project built with Anaconda.
It is recommended that anaconda is loaded to run the code here.
Anaconda is a data science package manager that is set up via
a downloader program from their website, www.anaconda.com.

Once anaconda is set up, there's some command line steps used
to install the program modules for this project, 
warning - it can take hours.

command line environment installation step examples:
conda update -n base -c defaults conda  
conda install -c anaconda numpy  
conda install -c anaconda pandas  
conda install -c conda-forge matplotlib  
conda install -c conda-forge imbalanced-learn  
conda install -c anaconda sqlite  
conda install -c anaconda sqlalchemy  
conda install -c anaconda dill  
conda install -c anaconda scipy  
conda install -c anaconda seaborn  
conda install -c anaconda scikit-learn  
conda install -c anaconda pillow  
conda install -c anaconda pip  
conda install -c anaconda flask  
conda install -c plotly plotly  
conda install -c anaconda nltk  
conda install -c conda-forge nltk_data  
conda install -c anaconda notebook  
conda update --all  
pip install -U imbalanced-learn  


For running this in the Udacity online Workspace (Feb'2021):

Most of the packages are pre-loaded but may be down revision.
To run the code there, I did these steps from terminal command line:

easy_install numpy
pip install --upgrade matplotlib
pip install -U imbalanced-learn
pip install -U seaborn
python
'>>>import nltk
'>>>nltk.download('stopwords')
'>>>nltk.download('averaged_perceptron_tagger')


<a name="get_code"></a>
### Get Code
There's a Git application that assists in copying code from Github.  You install this via command 
line depending upon your operating system.  Once installed, create a new folder, right click git-bash 
for a command line from inside the folder, and enter a git-clone line below to copy the code over.
```
git clone https://github.com/patronical/disaster_pipeline.git
```
<a name="run_code"></a>
### Run Code:
1. The Process Data section:
    - Open a terminal and navigate to the code folder
	  Run this command:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
		

2. The Train Classifier section:
     - Open a terminal and navigate to the code folder
	   Run this command:
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`


3. The Web App section:
     - Open a terminal and navigate to the app directory.
       Launch via the following command:
       python run.py

       If this is a local build,
       Open a web browser and copy http://0.0.0.0:3001/ into the address bar.

       Alternately, if this is being launced from the Udacity workspace,
       retrieve WORKSPACEID and WORKSPACEDOMAIN from a terminal beforehand:
       env | grep WORK
       python run.py
   
       Open a web browser and copy/paste: https://WORKSPACEID-3001.WORKSPACEDOMAIN
	   

<a name="notebooks"></a>
### Notebooks

Within **data** and **models** folders are jupyter notebooks used for development.  
1. **ETL Pipeline Preparation**: early development of the process data section.  
2. **augment_message**: simulation of messages for oversampling purposes.  
3. **peter_norvig_spelling**: development of spell check interfacing.  
4. **spell_check**: development of frequency and lookup dictionarys.  
5. **test_etl**: block by block executions of the process data script.  
6. **ML Pipeline Preparation Notebook**: early development of the train classifier section.  
7. **test_mlp**: block by block execution of the train classifier section.  
8. **test_frontend**: takes a trained model and runs message samples of various types.  

<a name="files"></a>
### Files

**app/templates/***: templates/html files for web app.  
**data/process_data.py**: process data code.  
**models/train_classifier.py**: train classifier code.  
**run.py**: web app code.  
**peter_norvig_spelling.py**: Peter Norvig's spelling utilities.  
**spellhelper.py**: class that manages frequency dictionary intefaced to Peter Norvig's spelling code.  
**spellbuilder.py**: custom frequency dictionary and string to string lookup dictionary build code.  
**cleaner.py**: code for message cleaning including spelling corrections.  
**augmenter.py**: code for oversampling messages via augmentation or simulation.  
**build_visuals.py**: code for building visuals from train data.  
**lookup_dict.pkl**: string to string spell correction dictionary.  
**freq_dict.txt**: custom disaster tuned frequency dictionary for spell corrections.  
**frequency_dictionary_en_82_765.txt**: standard English frequency dictionary.  
**messages.csv**: raw data.  
**categories.csv**: raw data.  
**train_disaster.png**: visualizations image file.  
**DisasterResponse.db**: cleaned database.  

<a name="authors"></a>
## Authors

* [Patrick Parker](https://github.com/patronical)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) Data Science Nanodegree Program.  
* [Figure Eight](https://www.figure-eight.com/) Disaster Response Data Set.  
* [Peter Norvig](https://www.peternorvig.com/) Spell Correction Code.  
* [SymSpell](https://github.com/wolfgarbe/SymSpell/) English Frequency Dictionary.  

<a name="results"></a>
## Results

1. Classification performance results are found in summary file, (classification_results.pdf).

2. Sample screenshots of the website is found in a summary file, (screenshots.pdf).
