"""
Contains various utility functions useful for other modules.
"""

from constants import minimum_scores, maximum_scores, GLOVE_DIR, DATASET_DIR
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

tokenizer = Tokenizer()

def configure(use_cpu=False, gpu_memory_fraction=0.25, silence_warnings=True):
    """
    Configures tensorflow.
    """
    if silence_warnings:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if use_cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        set_session(tf.Session(config=config))

def create_folder(folder_path):
    """
    Creates folder with the given name.
    """
    try:
        os.makedirs(folder_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def load_data(dataset_directory, train_size=0.8, validation_size=0.2):
    """
    Loads data from csv file and divides it into test, train and validation.
    """
    essays = pd.read_csv(os.path.join(dataset_directory, 'training_set_rel3.tsv'), sep='\t',
                     encoding="ISO-8859-1")[['essay_id', 'essay_set', 'essay', 'domain1_score']]

    essays_training, essays_test = train_test_split(essays, train_size=train_size, random_state=0)
    essays_train, essays_cv = train_test_split(essays_training, test_size=validation_size)

    return essays_train, essays_cv, essays_test

def normalise_scores(essays):
    """
    Takes in unnormalised scores and normalises them in range 0-1.
    """
    normalised_scores = []

    for index, row in essays.iterrows():
        score = row['domain1_score']
        essay_set = row['essay_set']
        normalised_score = (score - minimum_scores[essay_set]) / (maximum_scores[essay_set] - minimum_scores[essay_set])
        normalised_scores.append(normalised_score)

    return np.array(normalised_scores)

def process_data(essays, essay_length=500):
    """
    Converts raw data to data ready to be feeded to neural network.
    """
    normalised_scores = normalise_scores(essays)

    tokenizer.fit_on_texts(essays['essay'])
    essay_sequences = tokenizer.texts_to_sequences(essays['essay'])

    X = pad_sequences(essay_sequences, maxlen=essay_length)
    y = normalised_scores

    return X, y

def load_embedding_matrix(glove_directory, embedding_dimension=50):
    """
    Loads pretrained word vectors for initialising the embedding layer.
    """

    with open(os.path.join(glove_directory, 'glove.6B.' + str(embedding_dimension) + 'd.txt')) as file:
        embeddings = {}
        for line in file:
            values = line.split()
            embeddings[values[0]] = np.array(values[1:], 'float64')

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dimension))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix