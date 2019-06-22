import keras.backend as K

def action_acc(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true), K.argmax(y_pred)), axis=-1)