import keras.backend as K
import numpy as np

def action_acc(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    return np.mean(np.array(y_true == y_pred, dtype='i'), axis=-1)
