import numpy as np
import keras
import os
import cv2
from random import shuffle
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Flatten, Reshape, Dense, Input
from keras.optimizers import Adagrad
img_width, img_height = 150, 50

train_data_dir = "/home/duc/Downloads/captcha/captcha1/train"
test_data_dir = "/home/duc/Downloads/captcha/captcha1/test"
epochs = 10
list_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
list_chars_len = len(list_chars)

def get_label(img):
	label = []
	for i in range(len(img.split(".")[0])):
		for j in range(list_chars_len):
			if img[i] == list_chars[j]:
				label.append(to_categorical(j,list_chars_len))
				break
	return np.array(label)

def data(data_dir):
	data = []
	for i in tqdm(os.listdir(data_dir)):
		path = os.path.join(data_dir, i)
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		label = get_label(i)
		data.append([np.array(img),label])
	shuffle(data)
	return data

train_image = data(train_data_dir)
test_image = data(test_data_dir)

train_img_data = np.array([i[0] for i in train_image])
train_img_label = np.array([i[1] for i in train_image])
train_img_label.shape = (-1,372)
print(train_img_data.shape)
print(train_img_label.shape)


test_img_data = np.array([i[0] for i in test_image])
test_img_label = np.array([i[1] for i in test_image])
test_img_label.shape = (-1,372)
print(test_img_data.shape)
print(test_img_label.shape)


		# BUILD MODEL

cnn = Sequential()
# Input 150x50 color image, using 32 filter size 3x3

cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_height,img_width,3)))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=2))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2), strides=2))
cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(1,2), strides=2))
cnn.add(BatchNormalization(axis=1))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(BatchNormalization(axis=1))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(1,2), strides=2))
cnn.add(Conv2D(512, (2, 2), activation='relu'))
cnn.add(Dropout(0.25))

model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(img_height,img_width,3)))
model.add(Reshape((16,512)))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Flatten())
#model.add(Reshape(10374,372))
model.add(Dense(372, activation='softmax'))

#ada_op = Adagrad(lr=0.015)
model.compile(loss='categorical_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

model.fit(train_img_data, train_img_label, batch_size=64, 
	epochs=epochs, validation_data=(test_img_data, test_img_label))

model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
