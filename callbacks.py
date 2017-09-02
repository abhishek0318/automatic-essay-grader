"""
Contains custom callbacks.
"""

from constants import minimum_scores, maximum_scores
import constants
import datetime
import json
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score
from util import process_data, create_folder

class QWKScore(Callback):
    def __init__(self, essays, save_to_file=True, print_to_screen=True):
        super()
        self.essays = essays
        self.save_to_file = save_to_file
        self.print_to_screen = print_to_screen

    def on_epoch_end(self, epoch, logs={}):
        # for each essay set calculate the QWK scores
        qwk_scores = []
        number_essays = []

        if self.print_to_screen:
            print("\nQWK Scores")

        for essay_set in range(1, 9):
            essays_in_set = self.essays[self.essays['essay_set'] == essay_set]
            X, y = process_data(essays_in_set)
            y_true = essays_in_set['domain1_score'].values

            normalised_prediction = self.model.predict(X)
            normalised_prediction = np.array(normalised_prediction)
            y_pred = np.around((normalised_prediction * (maximum_scores[essay_set] - minimum_scores[essay_set])) + minimum_scores[essay_set])

            qwk_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            qwk_scores.append(qwk_score)
            number_essays.append(len(essays_in_set))

            if self.print_to_screen:
                print("Set {}: {:.2f}".format(essay_set, qwk_score), end=' ')

        qwk_scores = np.array(qwk_scores)
        number_essays = np.array(number_essays)

        weighted_qwk_score = np.sum(qwk_scores * number_essays) / np.sum(number_essays)
        if self.print_to_screen:
            print('\nWeighted QWK score: {:.2f}'.format(weighted_qwk_score))

        if self.save_to_file:
            summary = "Epoch " + str(epoch + 1)
            log_values = "\n"
            for key, value in logs.items():
                log_values += "{}: {:.4f} ".format(key, value)
            individual_qwk_scores = "\n"
            for essay_set in range(8):
                individual_qwk_scores += "Set {}: {:.2f} ".format(essay_set + 1, qwk_scores[essay_set])
            summary = summary + log_values + individual_qwk_scores
            summary += '\nWeighted QWK score: {:.2f}'.format(weighted_qwk_score)
            summary += '\n\n'
            with open(os.path.join(constants.SAVE_DIR, "scores.txt"), "a") as f:
                f.write(summary)

class SaveModel(ModelCheckpoint):
    """
    Wrapper of Model Checkpoint class.
    """
    def __init__(self, directory, filename, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
      
        # make folder with the current time as name
        now = datetime.datetime.now()
        current_time = "{}_{}_{}_{}_{}_{}".format(now.day, now.month, now.year, now.hour, now.minute, now.second)
        constants.SAVE_DIR = os.path.join(directory, current_time)

        create_folder(constants.SAVE_DIR)

        ModelCheckpoint.__init__(self, os.path.join(constants.SAVE_DIR, filename), monitor=monitor, save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode, period=period)

    def on_train_begin(self, logs=None):
        # save model architecture.
        parsed = json.loads(self.model.to_json())
        with open(os.path.join(constants.SAVE_DIR, 'model.txt'), 'w') as file:
            file.write(json.dumps(parsed, indent=4))
