import numpy as np
import editdistance
import datetime
import matplotlib.pyplot as plt
import keras
import os
import re
import cv2
import itertools
import keras.callbacks
import pylab
from random import shuffle
from tqdm import tqdm
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import LSTM, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed, Lambda
from keras.layers import Flatten, Reshape, Dense, Input, Activation
from keras.optimizers import *


train_data_dir = "/home/duc/Downloads/captcha/captcha1/train"
test_data_dir = "/home/duc/Downloads/captcha/captcha1/test"
val_data_dir = "/home/duc/Downloads/captcha/captcha1/validation"

out_put_dirs = "test_captcha_model"
epochs = 10
regex = r'^[a-z A-Z]+$'
list_chars = u'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
list_chars_len = len(list_chars)

max_str_len = 16
img_width, img_height = 150, 50

def get_label(img):
  label = []
  for c in img:
    label.append(list_chars.find(c))
  #for c in range(max_str_len - len(label)):
  #	label.append(list_chars_len)
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
    img = cv2.resize(temp, (img_width,img_height))
    label = get_label(i.split(".")[0])
    data.append([np.array(img),label,len(i.split(".")[0])])
  shuffle(data)
  return data

'''train_image = data(train_data_dir)
=======
    data.append([img,label])
  shuffle(data)
  return data

train_image = data(train_data_dir)
>>>>>>> 228f807f0087345963fbd4e63c0501d445c9a168
test_image = data(test_data_dir)

train_img_data = np.array([i[0] for i in train_image])
train_img_label = np.array([i[1] for i in train_image])
#train_img_label.shape = (-1,372)

test_img_data = np.array([i[0] for i in test_image])
test_img_label = np.array([i[1] for i in test_image])
#test_img_label.shape = (-1,372)
<<<<<<< HEAD
'''

		#CTC lambda function


#CTC lambda function
def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  y_pred = y_pred[:, 2:, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(test_func, word_batch):
  out = test_func([word_batch])[0]
  ret = []
  for i in range(out.shape[0]):
    out_best = list(np.argmax(out[i, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    out_str = get_text(out_best)
    ret.append(out_str)
  return ret

#generator class 
class DataGenerator(keras.callbacks.Callback):
	def __init__(self, train_dir,
		im_h, im_w, ds_factor,train_size,
		val_dir, batch_size, val_size,
		n_channels=3, shuffle=True, max_str_leng=16):
		
		self.train_dir = train_dir
		self.val_dir = val_dir
		self.batch_size= batch_size
		self.train_size = train_size
		self.val_size =val_size
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.max_str_leng = max_str_leng
		self.ds_factor = ds_factor
		self.im_w = im_w
		self.im_h = im_h
		self.cur_train_index = 0
		self.cur_val_index = 0

	def get_output_size(self):
		return list_chars_len + 1

	def get_data(self):
		train_data = data(self.train_dir)
		train_X =[]
		train_Y = []
		train_Y_len = []
		for i in train_data:
			train_X.append(i[0])
			train_Y.append(i[1])
			train_Y_len.append(i[2])
		self.train_X = np.array(train_X)
		self.train_Y = np.array(train_Y)
		self.train_Y_len = np.array(train_Y_len)

		#self.train_X = np.array(i[0] for i in train_data)
		#self.train_Y = np.array(i[1] for i in train_data)
		#self.train_Y_len = np.array(i[2]for i in train_data)

		val_data = data(self.val_dir)
		val_X = []
		val_Y = []
		val_Y_len = []
		for i in val_data:
			val_X.append(i[0])
			val_Y.append(i[1])
			val_Y_len.append(i[2])
		self.val_X = np.array(val_X)
		self.val_Y = np.array(val_Y)
		self.val_Y_len = np.array(val_Y_len)

		#self.val_X = np.array(i[0] for i in val_data)
		#self.val_Y = np.array(i[1] for i in val_data)
		#self.val_Y_len = np.array(i[2] for i in val_data)

		'''
		if K.image_data_format() == 'channels_first':
			self.train_X.shape = (self.train_size, 3, self.im_w, self.im_h)
			self.val_X.shape = (self.val_size, 3, self.im_w, self.im_h)
		else:
			self.train_X.shape = (self.train_size, self.im_w, self.im_h, 3)
			self.val_X.shape = (self.val_size, self.im_w, self.im_h, 3)
			'''

	def get_batch(self, index, size, train):
		X_data = []
		labels = []
		input_length = np.ones([size, 1])
		label_length = np.ones([size, 1])
		source_str = []
		for i in range(size):
			if train:
				X_data.append(self.train_X[index + i])
				labels.append(self.train_Y[index + i])
				input_length[i] = self.im_w // self.ds_factor - 2
				label_length[i] = self.train_Y_len[index + i]
				source_str.append(get_text(labels[i]))
			else:
				X_data.append(self.val_X[index + i])
				labels.append(self.val_Y[index + i])
				input_length[i] = self.im_w // self.ds_factor - 2
				label_length[i] = self.val_Y_len[index + i]
				source_str.append(get_text(labels[i]))
		X_data = np.array(X_data)
		labels = np.array(labels)
		inputs = {
			'the_input': X_data,
			'the_labels': labels,
			'input_length': input_length,
			'label_length': label_length,
			'source_str': source_str
			}
		outputs = {'ctc': np.zeros([size])}
		return (inputs, outputs)

	def next_train(self):
		while 1:
			self.cur_train_index = 0
			self.cur_val_index = 0
			ret = self.get_batch(0, self.batch_size, train = True)
			self.cur_train_index += self.batch_size
			yield ret

	def next_val(self):
		while 1:
			self.cur_train_index = 0
			self.cur_val_index = 0
			ret = self.get_batch(0, self.batch_size, train=False)
			self.cur_val_index += self.batch_size
			yield ret

	def on_train_begin(self, logs={}):
		self.get_data()


# Call back class for visualize training process
class Visualize_callback(keras.callbacks.Callback):

	def __init__(self, run_name, test_func, data_gen, num_display_words=6):
		self.test_func = test_func
		self.output_dir = os.path.join(out_put_dirs, run_name)
		self.data_gen = data_gen
		self.num_display_words = num_display_words
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
   
	def show_edit_distance(self, num):
		num_left = num
		mean_norm_ed = 0.0
		mean_ed = 0.0
		while num_left > 0:
			word_batch = next(self.data_gen)[0]
			num_proc=min(word_batch['the_input'].shape[0],num_left)
			decoded_res= decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
			for j in range(num_proc):
				edit_dist =editdistance.eval(decoded_res[j], word_batch['source_str'][j])
				mean_ed += float(edit_dist)
				mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
			num_left -= num_proc
		mean_norm_ed = mean_norm_ed/num
		mean_ed = mean_ed / num
		print('\nOut of %d samples: Mean edit distance: %0.3f Mean normalized edit distance: %0.3f' % (num, mean_ed, mean_norm_ed))

	def on_epoch_end(self, epoch, logs={}):
		self.model.save_weights(os.path.join(self.output_dir, 'weights.h5'))
		self.show_edit_distance(256)
		word_batch = next(self.data_gen)[0]
		res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
		if word_batch['the_input'][0].shape[0] < 256:
			cols = 2
		else:
			cols = 1
		for i in range(self.num_display_words):
			plt.subplot(self.num_display_words // cols, cols, i+1)
			if K.image_data_format() == 'channels_first':
				the_input = word_batch['the_input'][i, 0, :, :]
			else:
				the_input = word_batch['the_input'][i, :, :, 0]
			plt.imshow(the_input, cmap='Greys_r')
			plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
		fig = pylab.gcf()
		fig.set_size_inches(10, 13)
		plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
		plt.close()










    # BUILD MODEL

#setup parameter

def train(run_name, start_epoch, stop_epoch):
	cnn_act = 'relu'
	cnn_kernel = (3,3)

	rnn_size = 512
	kernel_init = 'he_normal'
	batch_sizes = 32
	nb_train = os.listdir(train_data_dir)
	nb_val = os.listdir(val_data_dir)

	if K.image_data_format() == 'channels_first':
  		input_shape = (3, img_height, img_width)
	else:
  		input_shape = (img_height, img_width, 3)

	data_gen = DataGenerator(train_dir = train_data_dir,
 						  val_dir = val_data_dir,
 						  train_size = nb_train,
 						  val_size = nb_val,
 						  n_channels = 3,
 						  batch_size = batch_sizes,
 						  im_w = img_width,
 						  im_h = img_height,
 						  ds_factor = 2**2
 						  )

	# Input 150x50 color image, using 32 filter size 3x3
	Input_data = Input(name='the_input', shape=input_shape, dtype='float32')
	cnn = Conv2D(64, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv1')(Input_data)
	cnn = MaxPooling2D(pool_size=(2, 2), strides=2, name='max1')(cnn)
	#cnn = Dropout(0.25)(cnn)

	cnn = Conv2D(64, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv2')(cnn)
	cnn = MaxPooling2D(pool_size=(2,2), strides=2)(cnn)
	
	cnn = Conv2D(256, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv3')(cnn)
	cnn = Conv2D(256, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv4')(cnn)
	cnn = MaxPooling2D(pool_size=(1,2), strides=2)(cnn)
	#cnn = Dropout(0.25)(cnn)
	cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv5')(cnn)
	cnn = BatchNormalization(axis=1)(cnn)
	cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same',
                kernel_initializer=kernel_init, name='conv6')(cnn)
	cnn = BatchNormalization(axis=1)(cnn)
	cnn = MaxPooling2D(pool_size=(1,2), strides=2)(cnn)
	#cnn = Conv2D(512, cnn_kernel, activation=cnn_act, padding='same',
    #            kernel_initializer=kernel_init, name='conv7')(cnn)
	#cnn.add(Conv2D(512,(1,2), activation=cnn_act))
	#cnn = Dropout(0.25)(cnn)
	
	#print(conv_to_rnn_dims)
	print(cnn.shape)
	#conv_to_rnn_dims = (img_width//(2**2), (img_height // (2**2))*64)
	#rnn = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(cnn)
	rnn = Dense(32, activation=cnn_act, name='dense1')(cnn)
	print(rnn.shape)
	rnn = Reshape(target_shape=(27,32))(rnn)
	rnn = Bidirectional(LSTM(rnn_size, return_sequences=True, name='lstm1'))(rnn)
	#rnn = Dropout(0.25)(rnn)
	rnn = Bidirectional(LSTM(rnn_size, return_sequences=True, name='lstm2'))(rnn)
	rnn = Dense(list_chars_len+1, kernel_initializer=kernel_init
          ,name='dense2')(rnn)
	y_pred = Activation('softmax', name='softmax')(rnn)
	#Model(inputs=Input_data, outputs=y_pred).summary()

	labels = Input(name='the_labels', shape=[data_gen.max_str_leng-10], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='int64')
	label_length = Input(name='label_length', shape=[1], dtype='int64')

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

	sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

	model = Model(inputs=[Input_data, labels, input_length, label_length], outputs=loss_out)
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=['accuracy'])

	if start_epoch > 0:
		weights_file = os.path.join(out_put_dirs, os.path.join(run_name, 'weights.h5'))
		model.load_weights(weights_file)

	test_func = K.function([Input_data], [y_pred])

	viz_cb = Visualize_callback(run_name, test_func, data_gen.next_val())

	model.fit_generator(generator = data_gen.next_train(),
						steps_per_epoch = (len(nb_train)//batch_sizes - 1),
						epochs=stop_epoch,
						validation_data = data_gen.next_val(),
						validation_steps = (len(nb_val)//batch_sizes - 1),
						callbacks=[viz_cb, data_gen],
						initial_epoch=start_epoch)

run_name = "build_2"
train(run_name,0,10)

