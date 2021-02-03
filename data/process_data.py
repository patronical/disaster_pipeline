# etl_pipeline.py
# loads message and categories sets
# merges the two datasets
# splits, drops duplicates
# fixes categorical problems
# cleans messages
# stores the cleaned messages
# based upon ETL_Pipeline_Preparation_r8.ipynb

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import cleaner
import augmenter
import spellbuilder
from spellhelper import Spellhelper
import pickle

def load_data(messages_filepath, categories_filepath):
    '''
    Input message and category data file paths
    return merged dataframe of the two files
    '''
    # load
    df_mess = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)

    # merge
    df = df_mess[['id','message']].merge(df_cat, left_on='id', right_on='id')

    return df


def Truncate(text, length=501):
    '''
    input string and trim length
    strip and rejoin
    trim to nearest word if too long
    return truncated cleaned string
    '''
    strip = text.rstrip()
    if len(strip) < length:
        clean = ' '.join(strip.split())
                         
    else:                
        tokens = strip[:length + 1].split()
        clean = ' '.join(tokens[0:-1])
                         
    return clean


def format_data(df):
    '''
    input merged dataframe
    determine categorys
    format category columns
    remove duplicates
    drop extraneous class
    return formatted df
    '''
    # category column names
    row = df.categories[0]
    category_colnames = [s[:-2] for s in row.split(';')]

    # format category columns
    df_form = pd.DataFrame()
    for i, nrow in df.iterrows():
        df_form[nrow['id']] = [int(s[-1]) for s in df.categories[i].split(';')]
    df_form = df_form.transpose().reset_index().rename(columns={'index':'id'})
    df_form.columns = ['id'] + category_colnames
    df = df.merge(df_form, left_on='id', right_on='id')
    df = df.drop(['categories'], axis=1)

    # remove duplicates
    df = df.drop_duplicates()

    # drop rows with NaN's for emergent cases
    df.dropna(inplace=True)

    # drop related category 2's
    df.drop(df[df['related']==2].index, inplace = True)

    return df


def simulate(df):
    '''
    import formatted df
    split off validation set
    simulate data for empty categories
    return validation, training, and simulated df's
    '''
    # category column names
    category_colnames = df.columns[2:].tolist()

    # start with some related zero's
    dfv = df[df['related']==0].sample(36).copy()

    # add category entries to validation set
    idx_list = dfv.index.tolist()
    # add in some one's for each target category
    for cat in category_colnames:
        # manage child alone has no entries
        if cat != 'child_alone':
            df_slice = df[df[cat]==1].copy()
            samp = df_slice.sample()
            while samp.index[0] in dfv.index:
                samp = df_slice.sample() 
            dfv = pd.concat([dfv, samp], axis=0)
        else:
            pass
    
    # independant train and validation data 
    dft = df[~df.index.isin(dfv.index)].copy()

    # Simulate child alone Messages
    df_aug = augmenter.BuildCAs(dft)

    # sample a 'child alone' entry for validation
    samp = df_aug.sample()
    dfv = pd.concat([dfv, samp], axis=0)

    # Simulate low count class messages
    dfa = augmenter.BuildSims(df_aug, dft)

    return dfv, dft, dfa


def save_data(df, database_filename):
    '''
    input cleaned dataframe and database name
    write cleaned dataframe to database table named MessCatRaw
    return none
    '''
    db_path = 'sqlite:///' + database_filename
    engine = create_engine(db_path)
    df.to_sql('MessCatRaw', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) != 4:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        sys.exit()

    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    # load
    print('loading...')
    df = load_data(messages_filepath, categories_filepath)

    # format
    print('formatting...')
    df = format_data(df)

    # spell utility build
    print('spell_corrections...')
    fd_file, lu_file = spellbuilder.BuildFiles(df)

    # clean and simulate
    print('cleaning and simulating...')
    augmenter.cleaner.speller = Spellhelper(fd_file)
    with open(lu_file, 'rb') as handle:
        augmenter.cleaner.corr_dict = pickle.load(handle)
    dfc, dfv, dft, dfa = augmenter.simulate(df)

    print('finishing...')
    # join
    df_all = pd.concat([dfc, dfa], axis = 0)
    # add validation and simulation flags
    df_all['val'] = df_all.index.isin(dfv.index)
    df_all['sim'] = df_all.index.isin(dfa.index)
    # trim
    df_all['message'] = df_all['message'].apply(Truncate)
    df_all.drop(df_all[df_all['message'].str.len()<=27].index, inplace = True)
    # save
    save_data(df_all, database_filepath)  
    print('cleaned data saved to database!')
    
if __name__ == '__main__':
    main()