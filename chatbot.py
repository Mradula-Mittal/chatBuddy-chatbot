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

#creating the pickel files
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
model.add(Dense(128, input_shape = (len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Stochastic gradient descent with Nesterov accelerated gradient gives good results for this one.
#Stochastic is more efficient than normal gradient descent

sgd = SGD(lr= 0.01, decay=1e-6, momentum=0.9 , nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fitting and saving the model (model saved in chatbuddy_model.h5 file)
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbuddy_model.h5', hist)

print('Model Created! Stay happy and move forward :>')