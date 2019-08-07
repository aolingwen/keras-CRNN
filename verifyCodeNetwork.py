#encoding=utf8


import utility
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Reshape,Lambda
from keras.layers import Input,Dense,Activation
from keras.layers.recurrent import GRU
from keras.models import Model
from keras import backend as K
from keras.layers.merge import add, concatenate



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_Model(training):
    input_shape = (utility.IMG_WIDTH, utility.IMG_HEIGHT, 1)
    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    x = inputs
    for i in range(3):
        x = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # CNN to RNN
    inner = Reshape(target_shape=((20, 7*128)), name='reshape')(x)
    inner = Dense(32, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

    # RNN layer
    gru_1 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)

    gru1_merged = add([gru_1, gru_1b])  # (None, 32, 512)

    gru_2 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    gru2_merged = concatenate([gru_2, gru_2b])  # (None, 32, 1024)

    # transforms RNN output to character activations:
    inner = Dense(len(utility.CHAR_SET)+1, kernel_initializer='he_normal', name='dense2')(gru2_merged)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[utility.CHAR_NUM], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])  # (None, 1)

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    else:
        return Model(inputs=[inputs], outputs=y_pred)
