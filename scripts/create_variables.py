import argparse
import os
import sys

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

import textstat as txst

from text_processing import *
from readability_measures import *

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

    # Readability measures

    df['punctuation_count'] = df[column_to_process].apply(punct)
    df['punctuation_score'] = df[column_to_process].apply(punct_score)
    df['lexicon_count'] = df[column_to_process].map(lambda t: txst.lexicon_count\
                                                  (t, removepunct = True))
    df['lexicon_score'] = df[column_to_process].apply(lexicon_score)
    df['sentence_count'] = df[column_to_process].apply(txst.sentence_count)
    df['sentence_score'] = df[column_to_process].apply(sentence_score)
    df['rd_automatedindex'] = df[column_to_process].apply(\
                                            txst.automated_readability_index)
    df['rd_fogscale'] = df[column_to_process].apply(txst.textstat.gunning_fog)
    df['rd_colemanliau'] = df[column_to_process].apply(txst.coleman_liau_index)
    df['rd_flesch_ease'] = df[column_to_process].apply(txst.flesch_reading_ease)
    df['rd_linearwrite'] = df[column_to_process].apply(\
                                                    txst.linsear_write_formula)
    df['rd_fleschkincaid_grade'] = df[column_to_process].apply(\
                                                    txst.flesch_kincaid_grade)
    df['rd_dalechall'] = df[column_to_process].apply(\
                                            txst.dale_chall_readability_score)
    df['rd_consensus'] = df[column_to_process].map(lambda t: txst.text_standard(\
                                                    t, float_output = True))

    df.to_csv('../data/outputs/' + new_file ,index=False)

    # parser = argparse.ArgumentParser()
    # parser._action_groups.pop()
    # req = parser.add_argument_group('required named arguments')
    # req.add_argument("--input", "-i", help="set input file", required=True)
    # req.add_argument("--output", "-o", help="set output file", required=True)
    # req.add_argument("--feature", "-f", help="set feature name", required=True)
    # args = parser.parse_args()
    # input_nm = args.input
    # var = args.feature
    # output_nm = args.output
    # print("[+] Reading file...")
    # input_df = pd.read_csv(input_nm, encoding = 'latin-1')

    # print("[+] Computing readability measures...")

    # df.to_csv(new_file + '.csv',index=False)
    print("[+] Done!")
