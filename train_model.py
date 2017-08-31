from util import configure, load_data, process_data
configure(use_cpu=True)

from callbacks import QWKScore, SaveModel
from constants import DATASET_DIR, TRAINED_MODELS_DIR
from keras.callbacks import TensorBoard
from models.lstm import get_model

print('Loading data..')
essay_length = 500
essays_train, essays_cv, essays_test = load_data(DATASET_DIR, train_size=0.8, validation_size=0.2)
print("Training Examples: {}".format(len(essays_train)))
print("Cross Validation Data: {}".format(len(essays_cv)))
print("Testing Data: {}".format(len(essays_test)))
print('Data loaded.')

print()
print('Processing data..')

X_train, y_train = process_data(essays_train)
print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))

X_cv, y_cv = process_data(essays_cv)
print("X_cv.shape: {}, y_cv.shape: {}".format(X_cv.shape, y_cv.shape))

print('Processing done.')

print()
print('Loading model..')
model = get_model(embedding_dimension=50, essay_length=essay_length)
print(model.summary())
print('Model loaded.')

qwkscore = QWKScore(essays_cv)
savemodel = SaveModel(TRAINED_MODELS_DIR, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5', verbose=0)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=32, write_grads=False)
callbacks_list = [qwkscore, tensorboard, savemodel]

print()
print('Training model..')
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_cv, y_cv), callbacks=callbacks_list)
print('Model trained.')