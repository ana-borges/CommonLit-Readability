import pandas as pd

import keras
from transformers import BertTokenizer, BertModel, TFAutoModelForSequenceClassification

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Data import

train_df = pd.read_csv('data/train.csv', encoding = 'latin-1')
test_df = pd.read_csv('data/test.csv', encoding = 'latin-1')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# For every sentence...
encoded_dict=list(range(len(train_df.excerpt)))
for (text, i) in zip(train_df.excerpt, range(len(train_df.excerpt))):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict[i] = tokenizer.encode_plus(
                        train_df['excerpt'][i],                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        padding='longest',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'tf',     # Return pytorch tensors.
                   )

# Load pre-trained model (weights)
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)




##BAD! THIS IS FOR PYTORCH
# model = BertModel.from_pretrained('bert-base-uncased' ,
#                                   # Whether the model returns all hidden-states.
#                                   output_hidden_states = True,)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

