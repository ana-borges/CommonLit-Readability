import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

train_df = pd.read_csv('data/train.csv', encoding = 'latin-1')
test_df = pd.read_csv('data/test.csv', encoding = 'latin-1')

## EXPLORATORY ANALYSIS

# Let's see the distribution of the lengths of the texts

train_df['length'] = train_df['excerpt'].apply(len)

def create_wordcloud(df):
    ''' Function to generate a wordcloud from a data frame'''
    words=''.join(list(df['excerpt']))
    spam_wc=WordCloud(width=512,height=512).generate(words)
    plt.figure(figsize=(8,6),facecolor='k')
    plt.imshow(spam_wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    return

sns.jointplot(x=train_df["length"], y=train_df["target"], kind='scatter', s=200, color='grey', edgecolor="skyblue", linewidth=2)
sns.set(style="dark", color_codes=True)
sns.jointplot(x=train_df["length"], y=train_df["target"], kind='kde', color="skyblue")
sns.jointplot(x=train_df["length"], y=train_df["target"], kind='hex', marginal_kws=dict(bins=30, fill=True))



def sea_histogram (df, column: str):
    fig, ax = plt.subplots()
    #plt.figure(facecolor='k')
    sns.histplot(df[column], kde = True, ax=ax)
    #sns.histplot(df[column2], kde = True, ax=ax)
    plt.show()
    return


# def stemmer(data_tokenized : list) -> list :
#     final_doc = []
#     for i, doc in enumerate(data_tokenized):
#         final_doc.append([])
#         for word in doc:
#             final_doc[i].append(''.join(snowball.stem(word)))
#             final_doc[i] = list(map(lambda x: ''.join(x), final_doc[i]))
#     return final_doc
