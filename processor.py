#import chatbot
import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbuddy_model.h5')

import json
import random
intents = json.loads(open('health_chat_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#clean up the input sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#creating the bag of words
def bow(sentence, words, show_details = True):
    #tokenize and lemmatize aka clean_up_sentence
    sentence_words = clean_up_sentence(sentence)
    #initializing bag of words(matrix of n words)
    bag =  [0]*len(words)
    for s in sentence_words:
        for index, word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[index] = 1
                if show_details:
                    print("Found in bag: %s" % word)
                    #used % in place of ,

    return(np.array(bag))



