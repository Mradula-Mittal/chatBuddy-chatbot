import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#nltk is for pre-data processing

import json
#loads json files directly into python
import pickle
#pickle loads pickle files

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


#Initializing chatbot Training
words = []
classes = []
documents = []
ignore_words = ['?', '!', '#', '*']

#importing my intents file
data_file = open('health_chat_intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intent']:
    for pattern in intent['patterns']:

        #tokenization - each word and tokenizing it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #adding documents
        documents.append((w, intent['tag']))

        #prevent repeats
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmetization




