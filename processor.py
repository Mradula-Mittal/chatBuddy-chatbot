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

def predict_class(sentence, model):
    #filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25  #to avoid too much overfitting
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    #sort by strength of probability
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    print(tag)
    list_of_intents = intents_json['intents']
    # print(list_of_intents[0])
    # print(i.get(tag))
    for i in list_of_intents:
        if (i.get("tag") == tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "Sorry, didn't understand your question. Please ask the right question."    
    else:
        result = "Sorry, didn't understand your question. Please ask the right question."    
    return result

def chatbot_response(message):
    ints = predict_class(message, model)
    res = getResponse(ints, intents)
    return res