import cv2
import itertools, os, time
import numpy as np
import argparse
from utility import *
from keras import backend as K
from verifyCodeNetwork import get_Model

def predict():
    model = get_Model(False)
    model.load_weights('./model/mymodel.h5')
    file_list = []
    X, Y = get_image_data_ctc('./valid_data', file_list)
    y_pred = model.predict(X)
    shape = y_pred[:, :, :].shape  # 2:
    out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :CHAR_NUM]  # 2:
    print()
    error_count = 0
    for i in range(len(X)):
        print(file_list[i])
        str_src = str(os.path.split(file_list[i])[-1]).split('.')[0]
        print(out[i])
        str_out = ''.join([str(CHAR_SET[x]) for x in out[i] if x != -1])
        print(str_src, str_out)
        if str_src != str_out:
            error_count += 1
            print('This is a error image---------------------------:', error_count)
    print((len(file_list) - error_count)/len(file_list))


predict()