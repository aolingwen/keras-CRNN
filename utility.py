#encoding=utf8

import random
import os
from captcha.image import ImageCaptcha
import cv2
import numpy as np

TRAIN_SIZE = 5000
VALID_SIZE = 500
CHAR_SET = '0123456789'
CHAR_NUM = 8
IMG_HEIGHT = 60
IMG_WIDTH = 160
FONT_SIZES = [40]
TRAIN_IMG_PATH = './train_data'
VALID_IMG_PATH = './valid_data'
LOG_DIR = './log/'
MODEL_DIR = './model/'
BATCH_SIZE = 32


def labels_to_text(labels):
    letters = [c for c in CHAR_SET]
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    letters = [c for c in CHAR_SET]
    return list(map(lambda x: letters.index(x), text))

def read_img(file_names):
    image = []
    for name in file_names:
        img = cv2.imread(name, 0)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.transpose(img, (IMG_HEIGHT, IMG_WIDTH))

        img = cv2.flip(img, 1)
        img = img/255.
        image.append(img)
    image = np.array(image)
    image = np.reshape(image, (-1, IMG_WIDTH, IMG_HEIGHT, 1))
    return image


def gen_train_batch(filenames, raw_labels, batch_size=BATCH_SIZE):
    steps = len(filenames) // batch_size
    filenames = np.array(filenames)
    raw_labels = np.array(raw_labels)
    while True:
        shuffle_idx = np.random.permutation(len(filenames))
        train_filenmaes_after_shullfle = filenames[shuffle_idx]
        train_labels_after_shullfle = raw_labels[shuffle_idx]

        for i in range(steps):
            labels = np.ones([batch_size, CHAR_NUM], dtype=np.int64)*(len(CHAR_SET))
            input_length = [20]*batch_size
            label_length = [CHAR_NUM]*batch_size
            input_length = np.array(input_length)
            label_length = np.array(label_length)

            filename_batch = train_filenmaes_after_shullfle[i * batch_size: (i + 1) * batch_size]
            label_batch = train_labels_after_shullfle[i * batch_size: (i + 1) * batch_size]
            X_data = read_img(filename_batch)

            for j, word in enumerate(label_batch):
                for p, c in enumerate(word):
                    labels[j, p] = int(CHAR_SET.find(c))

            inputs = {'the_input': X_data, 'the_labels': labels, 'input_length': input_length, 'label_length': label_length}
            outputs = {'ctc': labels}

            yield (inputs, outputs)



#生成不落地的验证码图片
def gen_a_verifycode():
    image = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT, font_sizes=FONT_SIZES)
    label = ''.join(random.sample(CHAR_SET, CHAR_NUM))
    img = image.generate_image(label)
    return np.asarray(img), label


#生成验证码图片
def gen_verifycode_img(gen_dir, total_size, chars_set, chars_num, img_height, img_width, font_sizes):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    image = ImageCaptcha(width=img_width, height=img_height, font_sizes=font_sizes)
    for i in range(total_size):
        num = np.random.randint(low=4, high=chars_num+1)
        label = ''.join(random.sample(chars_set, num))
        image.write(label, os.path.join(gen_dir, label+'.png'))


#生成名字列表和标签列表
def create_data_list(image_dir):
    if not os.path.exists(image_dir):
        return None, None
    images = []
    labels = []
    for file_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, file_name), 0)
        input_img = np.array(image, dtype='float32')
        label_name = os.path.basename(file_name).split('_')[0]
        images.append(input_img)
        labels.append(label_name)
    return images, labels




if __name__ == '__main__':
    print('在%s生成%d个验证码' % (TRAIN_IMG_PATH, TRAIN_SIZE))
    gen_verifycode_img(TRAIN_IMG_PATH, TRAIN_SIZE, CHAR_SET, CHAR_NUM, IMG_HEIGHT, IMG_WIDTH, FONT_SIZES)
    print('在%s生成%d个验证码' % (VALID_IMG_PATH, VALID_SIZE))
    gen_verifycode_img(VALID_IMG_PATH, VALID_SIZE, CHAR_SET, CHAR_NUM, IMG_HEIGHT, IMG_WIDTH, FONT_SIZES)
    print('生成完毕')
