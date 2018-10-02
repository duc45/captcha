import numpy as np
import keras
import os
import re
import cv2
import itertools
import keras.callback
from random import shuffle
from tqdm import tqdm
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed, Lambda
from keras.layers import Flatten, Reshape, Dense, Input, Activation
from keras.optimizers import *


train_data_dir = "/home/duc/Downloads/captcha/captcha1/train"
test_data_dir = "/home/duc/Downloads/captcha/captcha1/test"
out_put_dir = "test_captcha_model"
epochs = 10
regex = r'^[a-z A-Z]+$'
list_chars = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
list_chars_len = len(list_chars)
max_str_len = 16

def get_label(img):
  label = []
  for c in img:
    label.append(list_chars.find(c))
  return label

def get_text(label):
  text = []
  for c in label:
    if c == list_chars_len:
      text.append("")
    else:
      text.append(list_chars[c])
  return "".join(text)

def is_valid_str(str):
  search = re.compile(regex, re.UNICODE).search
  return bool((search(str)))

def data(data_dir):
  data = []
  for i in tqdm(os.listdir(data_dir)):
    path = os.path.join(data_dir, i)
    temp = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(temp, (100,32))
    label = get_label(i.split(".")[0])
    data.append([np.array(img),label])
  shuffle(data)
  return data

train_image = data(train_data_dir)
test_image = data(test_data_dir)

train_img_data = np.array([i[0] for i in train_image])
train_img_label = np.array([i[1] for i in train_image])
#train_img_label.shape = (-1,372)

test_img_data = np.array([i[0] for i in test_image])
test_img_label = np.array([i[1] for i in test_image])
#test_img_label.shape = (-1,372)

#CTC lambda function
def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  y_pred = y_pred[:, 2:, :]
  return K.ctc_batch_cost(lebels, y_pred, input_length, label_length)

def decode_batch(test_func, word_batch):
  out = test_func([word_batch])[0]
  ret = []
  for i in range(out.shape[0]):
    out_best = list(np.argmax(out[i, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    out_str = get_text(out_best)
    ret.append(out_str)
  return ret

# Call back class for visualize training process
'''class Visualize_callback(keras.callback.Callback):

  def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
    self.test_func = test_func
    self.output_dir = os.path.join(out_put_dir, run_name)
    self.text_img_gen = text_img_gen
    self.num_display_words = num_display_words
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)
   
  def show_edit_distance(self, num):
    num_left = num
    mean_norm_ed = 0.0
    mean_ed = 0.0
    while num_left > 0:
      word_batch = next(self.text_img_gen)[0]
'''







    # BUILD MODEL

#setup parameter
cnn_act = 'relu'
cnn_kernel = (3,3)
img_width, img_height = 100, 32
rnn_size = 512
kernel_init = 'he_normal'

if K.image_data_format() == 'channels_first':
  input_shape = (3, img_width, img_height)
else:
  input_shape = (img_width, img_height, 3)


# Input 150x50 color image, using 32 filter size 3x3
Input_data = Input(name='the_input', shape=input_shape, dtype='float32')
cnn = Conv2D(64, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv1')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2, name='max1')(cnn)
cnn = Dropout(0.25)(cnn)

cnn = Conv2D(128, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv2')(cnn)
cnn = MaxPooling2D(pool_size=(2,2), strides=2)(cnn)
cnn = Conv2D(256, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv3')(cnn)
cnn = Conv2D(256, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv4')(cnn)
cnn = MaxPooling2D(pool_size=(1,2), strides=2)(cnn)
cnn = Dropout(0.25)(cnn)
cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv5')(cnn)
cnn = BatchNormalization(axis=1)(cnn)
cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv6')(cnn)
cnn = BatchNormalization(axis=1)(cnn)
cnn = MaxPooling2D(pool_size=(1,2), strides=2)(cnn)
cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same'
                kernel_initializer=kernel_init, name='conv6')(cnn)
#cnn.add(Conv2D(512,(1,2), activation=cnn_act))
cnn = Dropout(0.25)(cnn)
cnn.summary()

conv_to_rnn_dims = (img_width//(1**2), (img_height // (2**2))*512)

rnn = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(cnn)
rnn = Dense(32, activation=cnn_act, name='dense1')(rnn)
rnn = Bidirectional(LSTM(rnn_size, return_sequences=True,
          kernel_initializer=kernel_init, name='lstm1'))(rnn)
rnn = Dropout(0.25)(rnn)
rnn = Bidirectional(LSTM(rnn_size, return_sequences=True,
          kernel_initializer=kernel_init, name='lstm2'))(rnn)
rnn = Dense(list_chars_len+1, kernel_initializer=kernel_init
          ,name='dense2')(rnn)
y_pred = Activation('softmax', name='softmax')
Model(inputs=Input_data, outputs=y_pred).summary()

labels = Input(name='the_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[Input_data, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizers=sgd)

epochs = 10
model.fit(train_img_data, train_img_label, batch_size=32, 
          epochs=epochs, verbose=1, validation_data=(test_img_data,test_img_label))

model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")