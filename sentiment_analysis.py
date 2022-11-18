import webbrowser
# task 1 
# Sentiment Analysis

# I have written a Medium article related to this task. 
# I haven't written a readme.txt file, so please consider my Medium aritcle as a substitute.

# link to the Medium article:
webbrowser.open("https://medium.com/@ashvanth11/sentiment-analysis-on-tweets-915d470f3b92")

import numpy as np 
import pandas as pd 
import re    
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from keras import Sequential
from keras.layers import Bidirectional, Dense, LSTM, Conv1D, Embedding, Dropout, MaxPooling1D
from keras.metrics import Precision, Recall
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem.porter import *

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec


# Load Tweet dataset
df = pd.read_csv("Twitter_Data.csv")

# drop missing rows
df.dropna(axis=0, inplace=True)

STOP_WORDS = stopwords.words('english')

data = list(df['clean_text'])

# Generating Word Cloud
wc = WordCloud( max_words=500, width = 1600 , height = 800, collocations=False,
                stopwords = STOP_WORDS).generate(" ".join(data))

plt.figure(figsize = (20,20))
plt.imshow(wc)
plt.show()

# Data preprocessing
def preprocess(tweet):

    # Eliminating links and mentions
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tweet)
        tweet = re.sub("(@[A-Za-z0-9_]+)","", tweet)
        
        words = tweet.split()

    # remove stopwords
        words = [w for w in words if w not in STOP_WORDS]

        return words



X = list(map(preprocess, df['clean_text']))

X = np.array(X)

# get_dummies used to dummy-code 'category' column 
Y = pd.get_dummies(df['category'])

# splitting test, train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 0)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

# we use Word2Vec for Word Embeddings
# Creating Word2Vec training dataset
Word2vec_train_data = list(map(lambda x: x, X_train))
Embedding_dimensions = 100
# Defining the model and training it
word2vec_model = Word2Vec(Word2vec_train_data,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 min_count=5)


# Tokenization and Padding
input_length = 60  #length of each token 
vocab_length = 60000  #length of vocabulary set

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
tokenizer.num_words = vocab_length

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)
X_val  = pad_sequences(tokenizer.texts_to_sequences(X_val) , maxlen=input_length)

# Creating Embedding Matrix  
embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

# Creating a Bidirectional Long short-term memory (LSTM) model
def get_model():
    model = Sequential()
    model.add(Embedding(vocab_length, Embedding_dimensions, input_length=input_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    return model

model = get_model()

# Model summary
training_model = get_model()
training_model.summary()

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
               metrics=['accuracy', Precision(), Recall()])

# Specifying batch size and epoch size
batch_size = 64
epochs = 10

# Training the model
history = model.fit(X_train, y_train,
                    validation_data= (X_val, y_val),
                    batch_size=batch_size, epochs=epochs)

model.save("savemodel")

# Assesing the training of the model
# Plots for accuracy vs epoch and loss vs epoch
def plot_training_hist(history):
   
    # accuracy vs epoch
    plt.figure(figsize = (10,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

    # loss vs epoch
    plt.figure(figsize = (10,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

plot_training_hist(history)

# Evaluating the accuracy, precison and recall of the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)

print(" ")
print('Accuracy  : {:.2f} %'.format(accuracy*100))
print('Precision : {:.2f} %'.format(precision*100))
print('Recall    : {:.2f} %'.format(recall*100))
print(" ")

# Generating confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']

    # prediction on the test set
    y_pred = model.predict(X_test)

    # compute confusion matrix
    cm = confusion_matrix(np.argmax(np.array(y_test),axis=1), np.argmax(y_pred, axis=1))

    plt.figure(figsize=(10,10))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=sentiment_classes,
                yticklabels=sentiment_classes)
    plt.title('Confusion matrix')
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    plt.show()
    
plot_confusion_matrix(model, X_test, y_test)