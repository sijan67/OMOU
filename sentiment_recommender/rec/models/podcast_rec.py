import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from tensorflow.python.keras import backend
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

from . import ml_util
import seaborn as sns
import os
import numpy as np
import pandas as pd

df = pd.read_csv(os.path.join(ml_util.BASE_DIR,r'datasets\podcasts.csv'))

temp1=df[df["Genre"] == "Literature"]
temp2=df[df["Genre"] == "Business News"]
temp3=df[df["Genre"] == "Comedy"]
temp4=df[df["Genre"] == "History"]
temp5=df[df["Genre"] == "Places & Travel"]

merged_df = pd.concat([temp1, temp2, temp3,temp4,temp5]) #merge only 5 genres for better accuracy
df=merged_df.reset_index(drop=True) #reset the starting index from 0
nltk.download("stopwords")



df = df.reset_index(drop=True)
replace_by_space = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower() # lowercase text
    text = replace_by_space.sub(' ', text) # replace 'replace_by_space' symbols by space in text and substitute the matched string with space.
    text = bad_symbols.sub('', text) # remove symbols which are in 'bad_symbols' from text and substitute the matched string with nothing. 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text

df['Description'] = df['Description'].apply(clean_text)
df['Description'] = df['Description'].str.replace('\d+', '')



# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Description'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['Description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['Genre']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 7)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
  callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

def podcast_model(input_string):
  seq = tokenizer.texts_to_sequences([input_string])
  padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
  pred = model.predict(padded)
  labels = ['Literature', 'Business News', 'Comedy' ,'History', 'Places & Travel']
  print(pred, labels[np.argmax(pred)])
  pred_val = labels[np.argmax(pred)]
  print(pred_val)
  recs =df[df["Genre"] == pred_val].head(5)
  print(type(recs))
  recommended = []
  for rec in recs:
    print(rec["Name"])
    #recommended.append(pd.Series.to_string(rec["Name"])) #
  return recommended