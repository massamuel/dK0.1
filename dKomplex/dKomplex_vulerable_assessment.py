import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow.keras as keras
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


twitter_dat = pd.read_csv("IRAhandle_tweets_2.csv")

twitter_dat = twitter_dat[(twitter_dat.region == "United States") | (twitter_dat.language == "English")]
twitter_dat = twitter_dat[twitter_dat.retweet == 1]
twitter_dat = twitter_dat[twitter_dat.followers > 1000]

twitter_dat = twitter_dat[['content','account_category']]

twitter_dat = twitter_dat[(twitter_dat.account_category == "RightTroll") | (twitter_dat.account_category == "LeftTroll")]

twitter_dat['sentiment'] = (twitter_dat.account_category == "RightTroll").astype(int)




##RightTroll 1
##LeftTroll 0

twitter_dat.isna().sum().sum()

X = twitter_dat[['content']]



TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X_preprocess = []
tweets = list(twitter_dat['content'])
for tweet in tweets:
    X_preprocess.append(preprocess_text(tweet))


y = twitter_dat.sentiment.values


X_train, X_test, y_train, y_test = train_test_split(X_preprocess, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
##Word2Vec library
glove_file = open('glove/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#####
        
## Deep learning model with Gated Recurrent Units        
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(128),
    keras.layers.Dense(1, activation="sigmoid")
])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
plt.plot(y_pred,color="blue",label="predicted")
plt.plot(y_test,color="red",label = "actual")
plt.show()


def predict_input(str_input):
    instance = tokenizer.texts_to_sequences(str_input)
    
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)
            
    flat_list = [flat_list]
    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    return instance


import matplotlib.pyplot as plt

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


