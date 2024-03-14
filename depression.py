import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk as nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
import re

from math import exp
from numpy import sign

from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Reproducibility
np.random.seed(1234)

DEPRES_NROWS = 3200 # number of rows to read from DEPRESSIVE_TWEETS_CSV
RANDOM_NROWS = 12000 # number of rows to read from RANDOM_TWEETS_CSV
MAX_SEQUENCE_LENGTH = 140 # Max tweet size
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS= 10

df ='depressive_tweets_processed.csv'
RANDOM_TWEETS_CSV = 'Sentiment Analysis Dataset 2.csv'
depressive_tweets_df = pd.read_csv(df, sep = '|', header = None, usecols = range(0,9), nrows = DEPRES_NROWS)
random_tweets_df = pd.read_csv(RANDOM_TWEETS_CSV, encoding = "ISO-8859-1", usecols = range(0,4), nrows = RANDOM_NROWS)
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'

print (depressive_tweets_df.head(10))
print(random_tweets_df.head())
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            # remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())

            # fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)

            # expand contraction
            tweet = expandContractions(tweet)

            # remove punctuation
            tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

            # stop words
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(tweet)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            tweet = PorterStemmer().stem(tweet) 
            cleaned_tweets.append(tweet)
    return cleaned_tweets
depressive_tweets_arr = [x for x in depressive_tweets_df[5]]
random_tweets_arr = [x for x in random_tweets_df['SentimentText']]
X_d = clean_tweets(depressive_tweets_arr)
X_r = clean_tweets(random_tweets_arr)
# X_d = ["I love programming in Python", "Machine learning is fascinating"]
# X_r = ["Natural language processing is important", "Python is a versatile language"]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_d + X_r)
# word_index1 = tokenizer.word_index

# # Print the word index


sequences_d = tokenizer.texts_to_sequences(X_d)  #text into integer sequence
# sequences_d = [
#     [1, 2, 3, 4, 5],        # "I love programming in Python"
#     [6, 7, 8, 9]            # "Machine learning is fascinating"
# ]

sequences_r = tokenizer.texts_to_sequences(X_r)
# sequences_r = [
#     [10, 11, 12, 8, 13],    # "Natural language processing is important"
#     [5, 8, 14, 15, 11]      # "Python is a versatile language"
# ]


word_index = tokenizer.word_index
# word_index = {
#     'i': 1,
#     'love': 2,
#     'programming': 3,
#     'in': 4,
#     'python': 5,
#     'machine': 6,
#     'learning': 7,
#     'is': 8,
#     'fascinating': 9,
#     'natural': 10,
#     'language': 11,
#     'processing': 12,
#     'important': 13,
#     'a': 14,
#     'versatile': 15,
# } each word in all the tweets(X_d+X_r) are assigned with integer so that no of unique words are collected

word_list = [word for word, index in sorted(word_index.items(), key=lambda x: x[1])]

# Print the list of words
print(word_list[:100])
print('Found %s unique tokens' % len(word_index))
data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)#fitting the sequence as per max seq len if greater truncate to combine dataset
data_r = pad_sequences(sequences_r, maxlen=MAX_SEQUENCE_LENGTH)# cleaned tweet can be any size so we limiting that into max_no_words=140 
#so that greater sized tweets are trimmed
# data_d=[[0 0 0 0 0 0 0 0 0 0 1 2 3 4 5]
#  [0 0 0 0 0 0 0 0 0 0 0 6 7 8 9]] padding length=15
print(data_r[:10])
print(data_d[:10])

print('Shape of data_d tensor:', data_d.shape)
print('Shape of data_r tensor:', data_r.shape)
# Shape of data_d tensor: (2308, 140)
# Shape of data_r tensor: (11911, 140)
nb_words = min(MAX_NB_WORDS, len(word_index))
print(nb_words)

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
#creating embedding matrix for our vocabulary with 300 dimension as our embedding file has 300 property for ecery vocabulary

for (word, idx) in word_index.items():
    if word in word2vec and idx < MAX_NB_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)
#searching for the every word in our vocabulary in embedding file 
#if it is there the corresponding vectors is assigned with our embedding matrix


for idx, (word, _) in enumerate(sorted(word_index.items(), key=lambda x: x[1])[5:7]):
    embedding_vector = embedding_matrix[idx]
    print(f"{word}: {embedding_vector}")     
# for idx, (word, idx) in enumerate(word_index.items()):
#     if idx < 5:  # Stop after printing the first 5 elements
#         embedding_vector = embedding_matrix[idx]
#         # Format and print the word and its vector
#         print(f"{word}:{embedding_vector}")
# for(word, idx) in word_index.items():
#     if idx < MAX_NB_WORDS:
#         embedding_vector = embedding_matrix[idx]
#         print(f"{word}: {embedding_vector}")

# for idx, embedding_vector in enumerate(embedding_matrix):
#     if idx < MAX_NB_WORDS:
#         print(f"Index {idx}: {embedding_vector}")
# Assigning labels to the depressive tweets and random tweets data
labels_d = np.array([1] * DEPRES_NROWS) 
labels_r = np.array([0] * RANDOM_NROWS) 
# labels_r = np.array(x for x in random_tweets_df['Sentiment'])
# labels_r = random_tweets_df['Sentiment'].values

# label_d=[1 1 1 1 1 1 1 1 1 1]
# label_r=[0 0 0 0 0 0 0 0 0 0]

# Splitting the arrays into test (60%), validation (20%), and train data (20%)
perm_d = np.random.permutation(len(data_d))
idx_train_d = perm_d[:int(len(data_d)*(TRAIN_SPLIT))]#0%-60% as TRAIN_SPLIT =0.6
idx_test_d = perm_d[int(len(data_d)*(TRAIN_SPLIT)):int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT))]#60%-80% AS TEST_SPLIT=0.2
idx_val_d = perm_d[int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT)):]#80%-100% AS VALID_SPLIT=0.2

perm_r = np.random.permutation(len(data_r))
idx_train_r = perm_r[:int(len(data_r)*(TRAIN_SPLIT))]
idx_test_r = perm_r[int(len(data_r)*(TRAIN_SPLIT)):int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_r = perm_r[int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT)):]

# Combine depressive tweets and random tweets arrays
data_train = np.concatenate((data_d[idx_train_d], data_r[idx_train_r]))
labels_train = np.concatenate((labels_d[idx_train_d], labels_r[idx_train_r]))
data_test = np.concatenate((data_d[idx_test_d], data_r[idx_test_r]))
labels_test = np.concatenate((labels_d[idx_test_d], labels_r[idx_test_r]))
data_val = np.concatenate((data_d[idx_val_d], data_r[idx_val_r]))
labels_val = np.concatenate((labels_d[idx_val_d], labels_r[idx_val_r]))

# Shuffling
# perm_train = np.random.permutation(len(data_train))
# data_train = data_train[perm_train]
# labels_train = labels_train[perm_train]
# perm_test = np.random.permutation(len(data_test))
# data_test = data_test[perm_test]
# labels_test = labels_test[perm_test]
# perm_val = np.random.permutation(len(data_val))
# data_val = data_val[perm_val]
# labels_val = labels_val[perm_val]

model = Sequential()

model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))


model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
print(model.summary())


early_stop = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val),epochs=EPOCHS, batch_size=40, shuffle=True)
from keras.models import load_model
loaded_model = load_model('depression_detection_model.h5')
labels_pred = model.predict(data_test)
labels_pred = np.round(labels_pred.flatten())
accuracy = accuracy_score(labels_test, labels_pred)
prec= precision_score(labels_test, labels_pred,average='weighted')
recall = recall_score(labels_test, labels_pred,average='weighted')
f1_scor = f1_score(labels_test, labels_pred,average='weighted')
confo_matrix = confusion_matrix(labels_test, labels_pred)
print("Accuracy: %.2f%%" % (accuracy*100))
print("prec",prec)
print("recall",recall)
print("f1",f1_scor)
print("con",confo_matrix)
# Compute the confusion matrix
cm = confusion_matrix(labels_test, labels_pred)

# Define the class labels (if binary classification, it's [0, 1])
class_names = [0, 1]

# Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(labels_test, labels_pred))


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ... (Your previous code)

# Function to preprocess and classify input text
def classify_input_text(input_text, loaded_model, tokenizer):
    # Clean and preprocess the input text as you did for the training data
    cleaned_input_text = clean_tweets([input_text])
    
    # Tokenize and pad the input text
    input_sequences = tokenizer.texts_to_sequences(cleaned_input_text)
    input_data = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Make predictions using the model
    prediction = model.predict(input_data)

    # Determine if it's a depressive tweet based on the prediction threshold
    threshold = 0.5  # You can adjust this threshold as needed
    is_depressive = prediction[0][0] > threshold
     
    return is_depressive

# Example usage:
input_text = "I am not happy."
is_depressive = classify_input_text(input_text, model, tokenizer)

if is_depressive:
    print("The input text is classified as a depressive tweet.")
else:
    print("The input text is not classified as a depressive tweet.")




