#!/usr/bin/env python3

import os
import sys
import argparse

import pandas as pd
import numpy as np

import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler


def clean_text (df : pd.DataFrame, new_col : str) -> None:
    '''Tokenize; remove stop words and puntuation; make lower case'''

    regex = re.compile('[%s]' % re.escape(string.punctuation))

    stop_words = stopwords.words('english')
    stop_words.append('the')

    #Make lower case
    df[new_col] = df['excerpt'].apply(lambda x: ''.join([str.lower(y) for y in x]))

     #Remove stop words
    df[new_col] =\
        df[new_col].apply(lambda x: ' '.join([w for w in x.split() if not w in stop_words]))

    #Tokenize words
    df[new_col] = df[new_col].apply(word_tokenize)

    #Remove punctuation
    rm_pnct =\
        lambda x : ' '.join([y.translate(str.maketrans('','',string.punctuation + ' ')) for y in x])
    df[new_col] = df[new_col].apply(rm_pnct)
    return


def stemmatize_text(df:pd.DataFrame, col : str) -> None:
    '''Stematize text in df[col]'''

    snowball = SnowballStemmer('english')
    #porter = PorterStemmer()
    df[col] = df[col].apply(lambda x: ' '.join([snowball.stem(w) for w in x.split()]))
    return


def tf_idf_counter(df : pd.DataFrame, col2count : str) -> list:
    '''Returns a descending-sorted list with the tf_idf values for
       all words of col2count'''

    count_vectorizer = CountVectorizer(min_df=1)

    # tfidf
    term_freq_matrix_all = count_vectorizer.fit_transform(df[col2count])
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(term_freq_matrix_all)

    # print idf_all values
    df_idf_all =\
    pd.DataFrame(tfidf.idf_, index=count_vectorizer.get_feature_names(),columns=["idf_weights"])

    # sort ascending
    return(list(df_idf_all.sort_values(by=['idf_weights']).index))


def freq(term : str, document : str) -> int :
    '''Frequency of a term in a document'''
    return document.split().count(term)


def add_common_words(words : list, pdf : pd.DataFrame, col : str) -> None :
    '''Adds a new column for each `word` in `words` counting the number of
       occurrencies of the word in each text'''
    for word in words :
        pdf.loc[:, word] = \
            pdf[col].apply(lambda doc : freq(word, doc))
    return


if __name__ == '__main__':

    # Read inputs from terminal
    parser = argparse.ArgumentParser(description='Process text')

    # File argument
    parser.add_argument('-f',
                        '--file',
                     type=str,
                     help='(path to) .csv to process')

    # Column to clean argument
    parser.add_argument('-c',
                        '--column',
                     type=str,
                     help='column to process')

    # Column where to store the processed text
    parser.add_argument('-nc',
                        '--new_column',
                        type=str,
                        help='column to store the processed text')

    # Name of the new file
    parser.add_argument('-nf',
                        '--new_file',
                        type=str,
                        help='name of the new file')


    args = parser.parse_args()

    # Check that the -f argument is indeed a file
    if not os.path.isfile(args.file):
        print('The file provided does not exist')
        sys.exit()

    ## Text processing
    file_to_process   = args.file
    column_to_process = args.column
    to_store          = args.new_column
    new_file          = args.new_file

    # Read the data
    df = pd.read_csv(file_to_process, encoding = 'latin-1')

    # Process the text
    clean_text(df, to_store)


    # Stemmatize the text
    stemmatize_text(df, to_store)

    # Create a descending-sorted list with the tf_idf score of each word
    all_words = tf_idf_counter(df, to_store)

    # We set the word variables to be the words from the 100th to the 600th
    # since we've seen that it is the optimal range in which they are not
    # too common nor too uncommon
    word_variables = all_words[100: 600]

    # Add common words
    add_common_words(word_variables, df, to_store)

    # Normalize the occurrencies' values
    scaler = MinMaxScaler()
    df[word_variables] = scaler.fit_transform(df[word_variables])

    df.to_csv('../data/outputs/' + new_file ,index=False)
