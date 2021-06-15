#!/usr/bin/env python3

import argparse
import textstat as txst
import pandas as pd
import string

def punct (text):
    return len([x for x in text if x in string.punctuation])

def punct_score (text):
    return round(100*punct(text)/(len(text)-text.count(" ")),4)

def lexicon_score(text):
    return round(100*txst.lexicon_count(text, removepunct = True)/\
                 (len(text)-text.count(" ")),4)

def sentence_score(text):
    return round(100*txst.sentence_count(text)/(len(text)-text.count(" ")),4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    req = parser.add_argument_group('required named arguments')
    req.add_argument("--input", "-i", help="set input file", required=True)
    req.add_argument("--output", "-o", help="set output file", required=True)
    req.add_argument("--feature", "-f", help="set feature name", required=True)
    args = parser.parse_args()
    input_nm = args.input
    var = args.feature
    output_nm = args.output
    print("[+] Reading file...")
    input_df = pd.read_csv(input_nm, encoding = 'latin-1')
    
    print("[+] Computing readability measures...")
    input_df['punctuation_count'] = input_df[var].apply(punct)
    input_df['punctuation_score'] = input_df[var].apply(punct_score)
    input_df['lexicon_count'] = input_df[var].map(lambda t: txst.lexicon_count\
                                                  (t, removepunct = True))
    input_df['lexicon_score'] = input_df[var].apply(lexicon_score)
    input_df['sentence_count'] = input_df[var].apply(txst.sentence_count)
    input_df['sentence_score'] = input_df[var].apply(sentence_score)
    input_df['rd_automatedindex'] = input_df[var].apply(\
                                            txst.automated_readability_index)
    input_df['rd_fogscale'] = input_df[var].apply(txst.textstat.gunning_fog)
    input_df['rd_colemanliau'] = input_df[var].apply(txst.coleman_liau_index)
    input_df['rd_flesch_ease'] = input_df[var].apply(txst.flesch_reading_ease)
    input_df['rd_linearwrite'] = input_df[var].apply(\
                                                    txst.linsear_write_formula)
    input_df['rd_fleschkincaid_grade'] = input_df[var].apply(\
                                                    txst.flesch_kincaid_grade)
    input_df['rd_dalechall'] = input_df[var].apply(\
                                            txst.dale_chall_readability_score)
    input_df['rd_consensus'] = input_df[var].map(lambda t: txst.text_standard(\
                                                    t, float_output = True))
    input_df.to_csv(output_nm + '.csv',index=False)
    print("[+] Done!")