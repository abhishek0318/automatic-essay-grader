from constants import GLOVE_DIR
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Sequential
import keras.regularizers
from util import tokenizer, load_embedding_matrix

def get_model(embedding_dimension, essay_length):
    """
    Returns compiled model.
    """
    vocabulary_size = len(tokenizer.word_index) + 1
    embedding_matrix = load_embedding_matrix(GLOVE_DIR, embedding_dimension)

    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_dimension, weights=[embedding_matrix], input_length=essay_length, trainable=False, mask_zero=True))
    model.add(Bidirectional(LSTM(150, dropout=0.4, recurrent_dropout=0.4)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.l2(0.0)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model