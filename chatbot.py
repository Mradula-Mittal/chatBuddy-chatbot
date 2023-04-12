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


words = []
classes = []
documents = []
ignore_words = ['?', '!', '#', '*']

#importing my intents file
data_file = open('health_chat_intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
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
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmetized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Building the Deep Learning Model 
#initializing the training data
training = []
output_empty = [0]*len(classes)

for doc in documents:
    #initializing bag of words
    bag = []
    #list of tokenized words for the pattern , then lemmatize each word
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        #if word match found in current pattern, create the bag_of_words array with 1
    
    #create a key (output_row) for the list
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

#dataset shuffle
random.shuffle(training)
training = np.array(training)

#train-test split -> X = patterns and Y = intents
train_x = list(training[:,0])
train_y = list(training[:,1])

print('Training data generated - A step completed!')

#Model Creation
#3 layers -> first layer (128 neurons), second layer (64 neurons) and third output layer contains number of neurons equal to number of intents to predict output intent with "softmax"
#Sequential model in keras is used here
model = Sequential()
model.add(Dense())