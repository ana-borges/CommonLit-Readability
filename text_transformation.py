#Analysis
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


train_df = pd.read_csv('train_df.csv', encoding = 'latin-1')
test_df = pd.read_csv('test_df.csv', encoding = 'latin-1')

count_vectorizer = CountVectorizer(min_df=1)

# All tfidf
term_freq_matrix_all = count_vectorizer.fit_transform(test_df.cleaned_text)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix_all)

# print idf_all values
df_idf_all =\
    pd.DataFrame(tfidf.idf_, index=count_vectorizer.get_feature_names(),columns=["idf_weights"])

# sort ascending
all_words = \
    list(df_idf_all.sort_values(by=['idf_weights']).index)

def freq(term : str, document : str) -> int :
    return document.split().count(term)

def add_common_words(words : list, pdf : pd.DataFrame) -> None :
    for word in words :
        pdf.loc[:, word] = \
            pdf.cleaned_text.apply(lambda doc : freq(word, doc))
    return

word_variables = all_words[40:]

add_common_words(word_variables, train_df)
add_common_words(word_variables, test_df)


train_df[word_variables] = scaler.fit_transform(train_df[word_variables])
test_df[word_variables] = scaler.fit_transform(test_df[word_variables])

train_df.to_csv('train_def.csv',index=False)
test_df.to_csv('test_def.csv',index=False)
