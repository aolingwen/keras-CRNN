#encoding=utf8

from verifyCodeNetwork import get_Model
from keras.optimizers import Adam,SGD,Adadelta,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import utility
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

    train_filenames = []
    train_labels = []
    for file_name in os.listdir('train_data/'):
        train_filenames.append(os.path.join('./train_data', file_name))
        train_labels.append(file_name[:-4])

    val_filenames = []
    val_labels = []
    for file_name in os.listdir('valid_data/'):
        val_filenames.append(os.path.join('./valid_data', file_name))
        val_labels.append(file_name[:-4])


    model = get_Model(True)
    print(model.summary())

    try:
        model.load_weights('./model/mymodel.h5')
        print("...Previous weight data...")
    except:
        print("...New weight data...")
        pass

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='./model/mymodel.h5', save_best_only=True, monitor='val_loss', verbose=1, mode='min', period=1)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=2)
    tb = TensorBoard(log_dir='./log')

    adam = Adadelta()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    model.fit_generator(generator=utility.gen_train_batch(train_filenames, train_labels),
                        steps_per_epoch=int(4893/utility.BATCH_SIZE),
                        epochs=200,
                        callbacks=[checkpoint, tb, rl],
                        validation_data=utility.gen_train_batch(val_filenames, val_labels),
                        validation_steps=int(500/utility.BATCH_SIZE))





