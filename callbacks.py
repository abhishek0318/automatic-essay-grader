"""
Contains custom callbacks.
"""

from constants import minimum_scores, maximum_scores
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import cohen_kappa_score
from util import process_data

class QWKScore(Callback):
    def __init__(self, essays):
        super()
        self.essays = essays

    def on_epoch_end(self, epoch, logs={}):
        qwk_scores = []
        number_essays = []
        print("\nQWK Scores")
        for essay_set in range(1, 9):
            essays_in_set = self.essays[self.essays['essay_set'] == essay_set]
            X, y = process_data(essays_in_set)
            y_true = essays_in_set['domain1_score'].values

            normalised_prediction = self.model.predict(X)
            y_pred = np.around((normalised_prediction * (maximum_scores[essay_set] - minimum_scores[essay_set])) + minimum_scores[essay_set])

            qwk_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            qwk_scores.append(qwk_score)
            number_essays.append(len(essays_in_set))
            print("Set {}: {:.2f}".format(essay_set, qwk_score), end=' ')

        qwk_scores = np.array(qwk_scores)
        number_essays = np.array(number_essays)

        weighted_qwk_score = np.sum(qwk_scores * number_essays) / np.sum(number_essays)
        print('\nWeighted QWK score: {:.2f}'.format(weighted_qwk_score))